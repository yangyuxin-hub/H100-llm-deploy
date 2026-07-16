# 03 数据准备

本页讲"要做什么数据准备":数据管理规则、P0 普通数据集、LongBench v2 本地子集、P1 固定数据。环境配置见 [02](02-environment-setup.md),用什么参数见 [04](04-evaluation-protocol.md)。

## 数据管理规则

所有评测数据和外部 harness 放在 `.eval-deps/`,不要写入 `models/`。每项正式测试至少记录:

- 数据集名称、来源、license、revision/commit;
- subset、split、few-shot 来源;
- 原始题量、过滤题量、生成题量、实际计分题量;
- 本地文件路径和 SHA256;
- prompt/chat template、答案抽取和聚合方式;
- 被排除样本及原因。

EvalScope adapter 数据通常在第一次 smoke 时自动下载并缓存。因此第一次运行需要数据源可访问;后续复现应复用同一缓存,并在 `frozen_config/` 和报告中记录实际数据信息。

## P0 普通数据集

P0 的 IFEval、IFBench、GPQA Diamond、MMLU-Pro、MMLU-Redux 和 C-Eval 由 EvalScope adapter 加载。运行前先核对:

```bash
column -s '|' -t benchmarks/modelcard/core_tasks.tsv | less -S
```

第一次 smoke 会同时完成数据下载、加载和 schema 验证。不要直接从 full 开始,因为数据或 adapter 问题可能在运行数小时后才暴露。

> node1 无公网,数据下载必须在本地完成后再同步缓存到 node1(见 [02](02-environment-setup.md)的"同步数据集缓存")。不要直接在 node1 上等 adapter 联网下载,会卡住。

## LongBench v2

LongBench v2 必须按真实 Qwen chat template 计算输入长度,不能依靠字符数或静默截断。

> 以下命令在**本地**执行(需要联网下载数据)。生成的子集再同步到 node1,见 [02](02-environment-setup.md)的"同步 LongBench 子集"。

先用与 EvalScope 相同的 ModelScope loader 下载并校验原始 503 题;这一步不访问模型端点:

```bash
.eval-deps/evalscope-1.6.1/bin/python - <<'PY'
from modelscope.msdatasets import MsDataset
dataset = MsDataset.load(
    'ZhipuAI/LongBench-v2',
    subset_name='default',
    split='train',
)
assert len(dataset) == 503, len(dataset)
print('LongBench v2 raw rows:', len(dataset))
PY
```

准备本地可运行子集:

```bash
.eval-deps/evalscope-1.6.1/bin/python \
  benchmarks/modelcard/prepare_longbench_inlimit.py \
  --tokenizer .eval-deps/tokenizers/Qwen3.6-27B-FP8 \
  --max-model-len 262144 \
  --max-output-tokens 16384
```

脚本会生成:

```text
.eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/
├── default_train.jsonl
├── skipped_context_limit.tsv
└── sample_counts.json
```

验收:

```bash
jq . .eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/sample_counts.json
wc -l .eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/default_train.jsonl
wc -l .eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/skipped_context_limit.tsv
```

规则是:

```text
模板化输入 tokens + 16,384 输出预算 <= 262,144
```

当前冻结结果为原始 503 题、可运行 400 题、超限 103 题。若 tokenizer、chat template、上下文或输出预算变化,必须重新生成,不能继续沿用 400 题。

如果本机缓存中存在多个 LongBench revision,自动查找会拒绝继续。此时先列出实际文件,再用 `--source-arrow` 明确选择已冻结的 revision:

```bash
find ~/.cache/modelscope/hub/datasets -path '*ZhipuAI_LongBench-v2-*' -name 'data-*.arrow'
```

## P1 固定数据

> 以下命令在**本地**执行(需要联网下载 HMMT)。生成的 JSONL 由 P1 远程入口自动同步到 node1。

```bash
bash benchmarks/modelcard/prepare_p1_data.sh
```

该脚本会:

- 下载固定 revision 的 HMMT Nov 2025;
- 校验 SHA256、schema、题量和题号唯一性;
- 生成本地 JSONL 和 `dataset_metadata.json`;
- 不访问模型服务,也不写入 `models/`。

验证:

```bash
jq . .eval-deps/data/modelcard-p1/hmmt_nov_2025/dataset_metadata.json
wc -l .eval-deps/data/modelcard-p1/hmmt_nov_2025/default_train.jsonl
```

其余任务以 `benchmarks/evalscope/reference_benchmarks.tsv` 为参考清单。若缺数据、judge、sandbox 或外部 harness,应保留 `BLOCKED_*`,不能用近似数据冒充正式结果。

## P1 smoke fixture

> 以下命令在**本地**执行。fixture 生成后由 P1 远程入口自动同步到 node1。

P1 smoke 可自动准备其余 8 项的单样本 fixture:

```bash
source config/evaluation.env
"${EVAL_PYTHON}" benchmarks/modelcard/prepare_p1_smoke_data.py \
  --output-dir "${P1_SMOKE_DATA_ROOT}"
```

fixture 只用于验证链路,不代替正式数据集;`dataset_metadata.json` 会记录来源、subset、行数和 SHA256。远程入口会自动执行这一步并把 fixture 同步到 node1 NVMe。

## P1 full 完整数据集(离线预下载)

> 以下操作在**本地**执行(需要联网)。数据下载后由手动 rsync 到 node1,P1 远程入口不会自动同步 full 数据集。

node1 无公网,evalscope 在 full 模式下默认会尝试联网拉取 HF/ModelScope 数据。必须把 6 个数据集预先下载到本地,再传到 node1 的 `${P1_FULL_DATA_ROOT}`(默认 `/root/llm-deploy/.eval-deps/data/p1-full-data/`)。`run_reference_suite.sh` 的 `dataset_args_for()` 已修改为 smoke/full 两种模式都传 `local_path`,只要目录存在就强制本地加载,不联网。

涉及的 6 个数据集:

| Benchmark | 子目录 | 关键文件 | 来源 |
|---|---|---|---|
| HLE w/ CoT | `hle/` | `data/test-00000-of-00001.parquet` | HuggingFace `HuggingFaceH4/hle` |
| SuperGPQA | `super_gpqa/` | `SuperGPQA-all.jsonl` | ModelScope `modelscope/SuperGPQA` |
| HMMT Feb 25 | `hmmt_feb_2025/` | `data/train-00000-of-00001.parquet` | HuggingFace `MathArena/hmmt-feb-2025` |
| MMMLU | `mmmlu/` | `test/mmlu_*.csv`(14 种语言) | HuggingFace `lighteval/M-MMMLU` |
| PolyMATH | `poly_math/` | 18 语言子目录,各含 `low/medium/high/top.parquet` | ModelScope `PolyMath/PolyMATH` |
| AA-LCR | `aa_lcr/` | `AA-LCR_Dataset.csv`、`extracted_text/lcr/` | 官方 release |

同步到 node1:

```bash
ssh root@10.16.11.24 'mkdir -p /root/llm-deploy/.eval-deps/data/p1-full-data'
rsync -az .eval-deps/data/p1-full-data/ \
  root@10.16.11.24:/root/llm-deploy/.eval-deps/data/p1-full-data/
```

验证:

```bash
ssh root@10.16.11.24 'ls /root/llm-deploy/.eval-deps/data/p1-full-data/'
# 应列出:aa_lcr  hle  hmmt_feb_2025  mmmlu  poly_math  super_gpqa
```

### BFCL memory 任务的 MiniLM 模型

BFCL 的 `memory_vector`、`memory_rec_sum` 子集需要 `all-MiniLM-L6-v2` 做 vector encoding。node1 无公网时,`SentenceTransformer` 默认构造函数会尝试联网下载并指数退避重试,导致 `memory snapshot prereq` 阶段卡数十分钟最终报 `RuntimeError: Cannot send a request, as the client has been closed.`。

本地下载(约 90MB)后上传到 node1:

```bash
# 本地下载
hf-mirror.com 或 huggingface.co 下载 all-MiniLM-L6-v2 整个目录

# 上传
rsync -az all-MiniLM-L6-v2/ \
  root@10.16.11.24:/mnt/nvme0/models/all-MiniLM-L6-v2/
```

patch node1 上的 `memory_vector.py:26`:

```python
# 原始
ENCODER = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
# 修改为
ENCODER = SentenceTransformer("/mnt/nvme0/models/all-MiniLM-L6-v2", device="cpu")
```

验证 patch 已生效:

```bash
ssh root@10.16.11.24 'grep -n "all-MiniLM-L6-v2" /root/llm-deploy/.eval-deps/evalscope-1.6.1/lib/python3.10/site-packages/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_vector.py'
# 应输出:26:ENCODER = SentenceTransformer("/mnt/nvme0/models/all-MiniLM-L6-v2", device="cpu")
```

该 patch 位置在 site-packages 中,重装 EvalScope venv 会丢失;重新同步 venv 后需要再次应用。
