# 模型卡测试完整流程(已归档)

> **本文档已废弃。** 内容已拆分到 [01 测试集总表](../01-benchmark-inventory.md) 至 [07 常见问题](../07-troubleshooting.md)。请从 [README](../README.md) 进入。本文件仅作历史追溯保留。

本文是一份可以直接照着执行的模型卡评测手册，目标是参考 [Qwen3.5-122B-A10B 模型卡](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)，对 `Qwen3.6-27B-FP8` 和实验模型进行同题、同配置、可恢复的能力评测。

本文只讲“如何复现”。实时分数见[跑分对比表](../benchmark-comparison.md)，82 项完整优先级和环境状态见[模型卡评测优先级汇总](evaluation-planning/2026-07-14-model-card-evaluation-priority-summary.md)。

## 1. 最终要完成什么

一次完整测试要依次完成：

```text
选择测试集
  → 检查模型服务
  → 配置评测环境和 SSH 隧道
  → 准备并冻结数据
  → 冻结 prompt、生成参数和评分协议
  → dry-run / preflight
  → smoke
  → full
  → 监控和断点续跑
  → 核对题量、错误、截断和评分
  → 写入模型卡
```

最终必须留下四类证据：

1. 环境：Python、EvalScope、模型端点和运行参数。
2. 配置：数据 split、few-shot、采样参数、输出预算和评分口径。
3. 原始结果：prediction、review、report、日志和状态表。
4. 结论：主分数、异常、限制以及与 122B 官方分数的可比性。

## 2. Qwen 122B 参考边界

Qwen3.5-122B-A10B 模型卡给出了 Language 和 Vision Language 测试项目及参考分数，本仓库整理为 82 项：37 项 Language、45 项 Vision Language。

官方没有完整公开每一项的 prompt、数据 revision、seed、harness、judge 和输出预算。因此：

- 122B 官方成绩是参考线，不自动构成严格 A/B。
- 两个本地模型必须使用同题、同配置和同评分协议。
- 每个结果都要标记可比性等级。

| 等级 | 含义 |
|---|---|
| `Strict` | 数据、prompt、harness、参数和评分协议均一致 |
| `Aligned` | 主要协议一致，少量官方细节无法确认 |
| `Approximate` | 同名 benchmark，但数据范围或协议有明确差异 |
| `Reference only` | 只引用官方分数，不做直接胜负判断 |
| `Not reproducible` | 数据或协议未公开，当前无法复现 |

## 3. 测试集 P0–P4 分层

P0–P4 表示模型卡业务优先级，不等于测试难度，也不等于当前环境已经可运行。环境是否就绪另用 `READY`、`BLOCKED_ENV`、`BLOCKED_DATA`、`BLOCKED_PROTOCOL` 等状态表示。

| 层级 | 数量 | 目的 | 主要能力 | 启动条件 |
|---|---:|---|---|---|
| P0 | 7 | 形成第一版可信模型卡 | 指令、知识、中文、推理、长上下文 | 基础 EvalScope 和两个端点可用 |
| P1 | 10 | 补齐高价值能力 | 数学、代码、工具、Agent、多语言 | sandbox、TAU2、judge 等按项就绪 |
| P2 | 13 | 验证视觉和中等复杂环境 | 视觉、软件工程、终端、多语言 | 视觉、sandbox、judge gate 通过 |
| P3 | 12 | 扩大视觉覆盖 | 通用视觉、文档、幻觉、grounding | P2-A 至少三项产生有效结果 |
| P4 | 40 | 专项基础设施任务池 | 搜索、视频、3D、GUI、Android、医疗 | 单独确认需求、预算、许可和隔离 |

### 3.1 P0：第一版模型卡

| Benchmark | 能力 | 计划题量 | 主要指标 |
|---|---|---:|---|
| IFEval | 指令遵循 | 541 | Prompt/Instruction strict、loose |
| IFBench | 复杂指令遵循 | 300 | Prompt/Instruction strict、loose |
| GPQA Diamond | 高难科学推理 | 198 | accuracy |
| MMLU-Pro | 综合知识与推理 | 12,032 | 14 subsets mean accuracy |
| MMLU-Redux | 综合知识复核 | 5,700 | inclusion-aware accuracy |
| C-Eval | 中文知识与推理 | 1,346 | 52 subsets mean accuracy |
| LongBench v2 | 长上下文 | 原始 503 | accuracy |

P0 不覆盖代码执行、Agent、多语言和视觉；这些能力从 P1 开始补充。

### 3.2 P1：高价值扩展

| Benchmark | 能力 | 特殊前置条件 |
|---|---|---|
| BFCL-V4 | Function Calling | BFCL extra；本地版本不含两类网页搜索 |
| TAU2-Bench | 多轮 Agent | TAU2 包、v0.2.0 数据、统一用户模拟器 |
| LiveCodeBench v6 | 代码生成 | Docker sandbox |
| HMMT Nov 25 | 高难数学 | 固定 revision 和 SHA256 的本地数据 |
| HMMT Feb 25 | 高难数学 | 固定 30 题 |
| PolyMATH | 多语言数学 | 18 种语言、4 个难度层级 |
| SuperGPQA | 专业知识 | 72 subsets |
| MMMLU | 多语言知识 | 14 种语言宏平均 |
| HLE w/ CoT | 高难推理 | 独立 judge；当前仅 self-judge smoke |
| AA-LCR | 长上下文 | 独立 judge，不允许静默截断 |

该表同时是默认执行顺序：先测最能区分 Agent/工具调用能力的 BFCL 与 TAU2，再测代码、高难数学、知识/多语言；缺独立 judge 的两项放在最后，避免阻塞已就绪任务。

P2–P4 的逐项清单、依赖和升级条件见 `benchmarks/evalscope/reference_benchmarks.tsv` 和上文链接的优先级汇总。

## 4. 评测架构和目录

```text
本地评测机
├── EvalScope 1.6.1
├── 数据缓存和外部 harness：.eval-deps/
├── 自动化 runner：benchmarks/modelcard/、benchmarks/evalscope/
└── 原始结果：logs/eval/
        │
        ├── SSH tunnel 18000 → node1:8000
        └── SSH tunnel 18001 → node1:8001

node1
├── Qwen3.6-27B-FP8：GPU 0,1，TP=2，端口 8000
└── 实验模型：GPU 2,3，TP=2，端口 8001
```

评测脚本只请求已有 OpenAI-compatible API，不启动、停止或重启模型服务，也不修改模型权重。

关键文件：

| 文件 | 用途 |
|---|---|
| `config/serving.env` | node1 上的 vLLM 配置 |
| `config/evaluation.env` | 评测环境、端点、数据、judge 路径 |
| `benchmarks/modelcard/core_tasks.tsv` | P0 任务和冻结参数 |
| `benchmarks/modelcard/p1_tasks.tsv` | P1 任务、环境和协议状态 |
| `benchmarks/evalscope/reference_benchmarks.tsv` | 82 项统一清单 |
| `run_core_evaluations.sh` | P0 双模型 smoke/full/resume |
| `run_p1_evaluations.sh` | P1 preflight/smoke/full/resume |
| `run_p1_ranked_evaluations.sh` | P1 十项按能力优先级统一调度、后台运行和去重状态 |
| `scripts/run_remote_p1_ranked_tests.sh` | 本地一键准备数据、同步 runner 并在 node1 自动执行 |
| `run_reference_suite.sh` | 82 项单模型预检和调度 |

## 5. 第一步：检查模型服务

先在项目根目录阅读当前目标和配置：

```bash
cd /home/yangyuxin/llm-deploy
sed -n '1,120p' PROJECT_LOG.md
sed -n '1,260p' config/serving.env
```

本地 `scripts/status.sh` 只反映本机，不能证明 node1 上的状态。远端检查应使用 SSH：

```bash
ssh node1 'cd /root/llm-deploy && bash scripts/status.sh'
ssh node1 'docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'
ssh node1 'docker logs --tail 30 qwen3.6-27b-fp8'
ssh node1 'docker logs --tail 30 thinkingcap-qwen3.6-27b-fp8'
```

通过标准：

- 两个目标容器均为 running/healthy。
- 8000 返回 `qwen3.6-27b-fp8`。
- 8001 返回当前实验模型的 served name。
- 端点模型名与 `config/evaluation.env` 一致。

这里只做只读检查。若服务未运行，应先停止评测并按部署流程处理，不要由评测脚本自动重启服务。

## 6. 第二步：建立 SSH 隧道并加载评测配置

在单独终端保持隧道运行：

```bash
ssh -N \
  -L 18000:127.0.0.1:8000 \
  -L 18001:127.0.0.1:8001 \
  node1
```

在评测终端初始化环境。这里显式导出 API 地址，因为 P0 runner 有自己的默认地址：

```bash
cd /home/yangyuxin/llm-deploy
export QWEN_API_URL=http://127.0.0.1:18000/v1
export THINKINGCAP_API_URL=http://127.0.0.1:18001/v1
export QWEN_SERVED_MODEL=qwen3.6-27b-fp8
export THINKINGCAP_SERVED_MODEL=thinkingcap-qwen3.6-27b-fp8
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
```

验证隧道和模型名：

```bash
curl --noproxy '*' -fsS "${QWEN_API_URL}/models" | jq .
curl --noproxy '*' -fsS "${THINKINGCAP_API_URL}/models" | jq .
```

如果第二个端点实际运行的不是 ThinkingCap，必须修改 `THINKINGCAP_SERVED_MODEL` 或停止本轮测试；不能把请求误发给同端口的其他模型。

## 7. 第三步：配置 EvalScope 环境

### 7.1 创建基础环境

当前冻结环境为 Python 3.10.20、EvalScope 1.6.1：

```bash
command -v uv
uv python install 3.10.20
uv venv --python 3.10.20 .eval-deps/evalscope-1.6.1
uv pip install --python .eval-deps/evalscope-1.6.1/bin/python \
  'evalscope[perf,app]==1.6.1'
```

如果 `uv` 尚未安装，应先按 uv 官方方式安装，不要改用系统 Python 污染项目环境。

验证基础环境：

```bash
.eval-deps/evalscope-1.6.1/bin/python --version
.eval-deps/evalscope-1.6.1/bin/evalscope --version
.eval-deps/evalscope-1.6.1/bin/python -c 'import evalscope; print(evalscope.__version__)'
```

### 7.2 安装 P0 依赖

P0 的 IFEval 和 IFBench 需要额外评分依赖：

```bash
uv pip install --python .eval-deps/evalscope-1.6.1/bin/python \
  'evalscope[ifeval,ifbench]==1.6.1' \
  'setuptools<81'
```

验证：

```bash
.eval-deps/evalscope-1.6.1/bin/python - <<'PY'
import emoji
import langdetect
import nltk
import syllapy
print('P0 dependencies: OK')
PY
```

### 7.3 按需安装 P1–P3 依赖

```bash
# Language、BFCL、代码 sandbox、SWE-bench
bash benchmarks/evalscope/setup_eval_environments.sh install-language

# OCR、文档、RefCOCO
bash benchmarks/evalscope/setup_eval_environments.sh install-vision

# WMT/COMET：独立 Python 3.10 环境
bash benchmarks/evalscope/setup_eval_environments.sh install-translation

# Terminal-Bench：独立 Python 3.12 环境
bash benchmarks/evalscope/setup_eval_environments.sh install-terminal
```

全量环境检查：

```bash
bash benchmarks/evalscope/setup_eval_environments.sh check
```

这个检查覆盖 Language、Vision、WMT、Terminal、Docker 和 TAU2；如果当前只跑 P0，P1–P4 的 `MISSING` 可以暂时存在，不能误判为 P0 不可运行。

### 7.4 sandbox、judge 和 Agent 环境

代码任务只有同时满足以下条件才允许运行：

```bash
docker info
.eval-deps/evalscope-1.6.1/bin/python -c 'import docker, ms_enclave'
```

- runner 显式传入 `--enable-sandbox`；
- Docker daemon 和当前用户权限可用；
- 代码执行环境与 H100 模型服务隔离。

需要 judge 的正式任务必须配置独立端点：

```bash
export EVALSCOPE_JUDGE_MODEL=<judge-served-name>
export EVALSCOPE_JUDGE_API_URL=http://127.0.0.1:<judge-port>/v1
```

`--allow-self-judge` 只用于 smoke，结果不能写入正式模型卡。

TAU2 要求包和数据同时存在：

```bash
test -d .eval-deps/tau2-data-v0.2.0/tau2/domains
.eval-deps/evalscope-1.6.1/bin/python -c 'import tau2; print("TAU2: OK")'
```

两个被测模型必须共用 `config/evaluation.env` 中同一个 `P1_TAU2_USER_MODEL`，否则用户模拟行为会污染 A/B。

## 8. 第四步：准备和冻结数据集

### 8.1 数据管理规则

所有评测数据和外部 harness 放在 `.eval-deps/`，不要写入 `models/`。每项正式测试至少记录：

- 数据集名称、来源、license、revision/commit；
- subset、split、few-shot 来源；
- 原始题量、过滤题量、生成题量、实际计分题量；
- 本地文件路径和 SHA256；
- prompt/chat template、答案抽取和聚合方式；
- 被排除样本及原因。

EvalScope adapter 数据通常在第一次 smoke 时自动下载并缓存。因此第一次运行需要数据源可访问；后续复现应复用同一缓存，并在 `frozen_config/` 和报告中记录实际数据信息。

### 8.2 P0 普通数据集

P0 的 IFEval、IFBench、GPQA Diamond、MMLU-Pro、MMLU-Redux 和 C-Eval 由 EvalScope adapter 加载。运行前先核对：

```bash
column -s '|' -t benchmarks/modelcard/core_tasks.tsv | less -S
```

第一次 smoke 会同时完成数据下载、加载和 schema 验证。不要直接从 full 开始，因为数据或 adapter 问题可能在运行数小时后才暴露。

### 8.3 LongBench v2

LongBench v2 必须按真实 Qwen chat template 计算输入长度，不能依靠字符数或静默截断。

先用与 EvalScope 相同的 ModelScope loader 下载并校验原始 503 题；这一步不访问模型端点：

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

准备本地可运行子集：

```bash
.eval-deps/evalscope-1.6.1/bin/python \
  benchmarks/modelcard/prepare_longbench_inlimit.py \
  --tokenizer .eval-deps/tokenizers/Qwen3.6-27B-FP8 \
  --max-model-len 262144 \
  --max-output-tokens 16384
```

脚本会生成：

```text
.eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/
├── default_train.jsonl
├── skipped_context_limit.tsv
└── sample_counts.json
```

验收：

```bash
jq . .eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/sample_counts.json
wc -l .eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/default_train.jsonl
wc -l .eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/skipped_context_limit.tsv
```

规则是：

```text
模板化输入 tokens + 16,384 输出预算 <= 262,144
```

当前冻结结果为原始 503 题、可运行 400 题、超限 103 题。若 tokenizer、chat template、上下文或输出预算变化，必须重新生成，不能继续沿用 400 题。

如果本机缓存中存在多个 LongBench revision，自动查找会拒绝继续。此时先列出实际文件，再用 `--source-arrow` 明确选择已冻结的 revision：

```bash
find ~/.cache/modelscope/hub/datasets -path '*ZhipuAI_LongBench-v2-*' -name 'data-*.arrow'
```

### 8.4 P1 固定数据

```bash
bash benchmarks/modelcard/prepare_p1_data.sh
```

该脚本会：

- 下载固定 revision 的 HMMT Nov 2025；
- 校验 SHA256、schema、题量和题号唯一性；
- 生成本地 JSONL 和 `dataset_metadata.json`；
- 不访问模型服务，也不写入 `models/`。

验证：

```bash
jq . .eval-deps/data/modelcard-p1/hmmt_nov_2025/dataset_metadata.json
wc -l .eval-deps/data/modelcard-p1/hmmt_nov_2025/default_train.jsonl
```

其余任务以 `benchmarks/modelcard/tasks/<task>/protocol.yaml` 为协议入口。若缺数据、judge、sandbox 或外部 harness，应保留 `BLOCKED_*`，不能用近似数据冒充正式结果。

P1 smoke 可自动准备其余 8 项的单样本 fixture：

```bash
source config/evaluation.env
"${EVAL_PYTHON}" benchmarks/modelcard/prepare_p1_smoke_data.py \
  --output-dir "${P1_SMOKE_DATA_ROOT}"
```

fixture 只用于验证链路，不代替正式数据集；`dataset_metadata.json` 会记录来源、subset、行数和 SHA256。远程入口会自动执行这一步并把 fixture 同步到 node1 NVMe。

## 9. 第五步：冻结评测协议

P0 的完整冻结配置在 `benchmarks/modelcard/core_tasks.tsv`：

| Benchmark | few-shot / split | temperature | max_tokens | seed |
|---|---|---:|---:|---:|
| IFEval | 0-shot / train | 0 | 8,192 | 42 |
| IFBench | 0-shot / train | 0 | 8,192 | 42 |
| GPQA Diamond | 0-shot CoT / train | 0.6 | 16,384 | 42 |
| MMLU-Pro | validation 5-shot CoT / test | 0.6 | 8,192 | 42 |
| MMLU-Redux | 0-shot CoT / test | 0.6 | 8,192 | 42 |
| C-Eval | dev 5-shot / val | 0.6 | 8,192 | 42 |
| LongBench v2 | 0-shot / train | 0.6 | 16,384 | 42 |

公共设置：

- `top_p=0.95`、`top_k=20`；确定性指令任务为 `top_p=1.0`、`top_k=-1`。
- `enable_thinking=true`。
- 单请求 timeout 900 秒，流式输出。
- 正式 P0 默认每个模型并发 4。
- 同一 benchmark 的两个模型并行，benchmark 之间串行。

每次运行时 runner 会把实际配置写入：

```text
<run-dir>/<benchmark>/<model>/frozen_config/
├── generation_config.json
├── dataset_args.json
├── expected_total.txt
├── protocol_note.txt
└── sample_counts.json        # LongBench v2
```

修改任何冻结项后应创建新的 run id。不能把不同参数的结果写入同一个 run 目录。

P1 的任务型配置在 `benchmarks/modelcard/p1_tasks.tsv`，当前冻结值如下：

| 顺序 | Benchmark | 类型 | temperature | top_p / top_k | max_tokens | 额外设置 |
|---:|---|---|---:|---|---:|---|
| 1 | BFCL-V4 | Function Calling | 0.7 | 0.8 / 20 | 81,920 | 两模型均 non-thinking / presence 1.5；ThinkingCap 另有 strict tool template |
| 2 | TAU2-Bench | 多轮 Agent | Qwen 1.0；ThinkingCap 0.7 | Qwen 0.95 / 20；ThinkingCap 0.8 / 20 | 81,920 | Qwen thinking；ThinkingCap non-thinking、presence 0；user simulator 为 0、关闭 thinking、无输出上限 |
| 3 | LiveCodeBench v6 | 精确代码 | 0.6 | 0.95 / 20 | 81,920 | thinking |
| 4 | HMMT Nov 25 | 竞赛数学 | 1.0 | 0.95 / 20 | 81,920 | thinking |
| 5 | HMMT Feb 25 | 竞赛数学 | 1.0 | 0.95 / 20 | 81,920 | thinking |
| 6 | PolyMATH | 多语言数学 | 1.0 | 0.95 / 20 | 81,920 | thinking |
| 7 | SuperGPQA | 专业知识推理 | 1.0 | 0.95 / 20 | 81,920 | thinking |
| 8 | MMMLU | 多语言知识推理 | 1.0 | 0.95 / 20 | 81,920 | thinking |
| 9 | HLE w/ CoT | 高难通用推理 | 1.0 | 0.95 / 20 | 81,920 | thinking；正式分数缺 judge |
| 10 | AA-LCR | 长上下文推理 | 1.0 | 0.95 / 20 | 81,920 | thinking；正式分数缺 judge |

配置依据与取舍：

- 两个模型都沿用 Qwen3.6 sampling；ThinkingCap 模型卡也用同一套参数。通用 thinking 任务采用官方推荐 `temperature=1.0`，精确代码采用 `0.6`。
- BFCL 按功能调用任务单独优化：两模型都使用 Qwen 官方 non-thinking 参数，避免 BFCL memory 中复现的 reasoning loop 和伪 XML；ThinkingCap 仍保留服务层严格工具模板。TAU2 的 Qwen 保持 thinking 并用保守 `presence_penalty=0.5` 防循环；ThinkingCap 使用 `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=0` 并关闭 thinking。原因是原始审计既多次捕获到 thinking-only 空响应，也捕获到原生 `temperature=1.0` 单轮持续生成；`presence_penalty=0` 避免惩罚从上下文复制工具必填 ID。两套 harness 都不传 `preserve_thinking`，不强制 `tool_choice`，也不做正文转工具调用的后处理。
- TAU2 user simulator 固定为 harness 默认 `temperature=0`，显式关闭 thinking，不传 `max_tokens`，保证两模型面对同一交互环境且不会因低预算产生空 UserMessage。
- 除上述 Qwen Agent 防循环配置外，采样型任务使用 `min_p=0.0`、`presence_penalty=0.0`、`repetition_penalty=1.0`、`seed=42`。高温单 seed 只能复现一次 pass@1；正式稳定性结论应补多个 seed 并报告均值与波动，不能挑最好 seed。
- `max_tokens=81920` 是所有被测模型任务的完整输出预算，不要求模型填满；自动化另检查 `finish_reason`，不能把达到上限的未完成输出当作正常答案。
- 2026-07-15 的十项双模型 smoke 证明旧参数下链路已跑通；本表优化参数落地后必须用新 run id 重跑 smoke，验证端点、sandbox、tool parser 和截断 gate。

## 10. 第六步：dry-run 和 preflight

### 10.1 82 项离线 dry-run

只生成计划，不访问端点、不下载数据、不请求模型：

```bash
bash benchmarks/evalscope/run_reference_suite.sh \
  --model qwen \
  --modality all \
  --priority all \
  --dry-run \
  --run-id modelcard-reference-dry-run
```

输出位于：

```text
logs/eval/reference-82/modelcard-reference-dry-run/qwen/
├── status.tsv
└── commands.log
```

状态含义：

| 状态 | 含义 |
|---|---|
| `DRY_RUN` | runner 和命令可以生成 |
| `BLOCKED_REVIEW` | adapter 存在，但协议仍需核对 |
| `BLOCKED_ENV` | 缺依赖、judge、sandbox 或数据 |
| `BLOCKED_EXTERNAL` | 需要外部 harness/worker |
| `BLOCKED_PROTOCOL` | adapter 或模型卡协议未准备好 |

### 10.2 P0 dry-run

P0 runner 的 dry-run 会冻结命令和配置，但不检查端点、不请求模型：

```bash
RUN_ID="modelcard-p0-dry-run-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode smoke \
  --run-id "${RUN_ID}" \
  --dry-run
```

检查：

```bash
cat "logs/eval/modelcard/${RUN_ID}/commands.log"
column -s $'\t' -t "logs/eval/modelcard/${RUN_ID}/suite_status.tsv" | less -S
```

### 10.3 P1 preflight

```bash
bash benchmarks/modelcard/run_p1_evaluations.sh \
  --mode preflight \
  --run-id "modelcard-p1-preflight-$(date +%Y%m%d-%H%M%S)"
```

Preflight 只检查端点、adapter、环境、数据、judge 和 sandbox，不发送正式题目。先处理 `BLOCKED_DATA` 和 `BLOCKED_ENV`；`BLOCKED_EXTERNAL` 不应通过安装普通 Python 包强行解除。

## 11. 第七步：运行 smoke

Smoke 的目标是验证完整链路，不评价模型能力。必须检查：

```text
数据加载 → prompt 构造 → API 请求 → 流式响应
→ 答案抽取 → 逐题 review → 聚合报告 → 状态落盘
```

### 11.1 P0 双模型自动 smoke

```bash
RUN_ID="modelcard-p0-smoke-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode smoke \
  --run-id "${RUN_ID}" \
  --eval-batch-size 2
```

`--limit 1` 由 runner 自动添加。注意它表示每个 subset 一题：MMLU-Pro、MMLU-Redux 和 C-Eval 会产生多条请求，不是整个 benchmark 只有一题。

也可以先单跑一项：

```bash
RUN_ID="modelcard-mmlu-pro-smoke-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode smoke \
  --benchmark mmlu_pro \
  --run-id "${RUN_ID}" \
  --eval-batch-size 1
```

### 11.2 P1 双模型自动 smoke

```bash
RUN_ID="p1-ranked-smoke-$(date +%Y%m%d-%H%M%S)"
bash scripts/run_remote_p1_ranked_tests.sh \
  --mode smoke \
  --run-id "${RUN_ID}" \
  --eval-batch-size 1 \
  --background
```

该入口自动准备 fixture、同步 runner，并在 node1 按 `p1_tasks.tsv` 的顺序跑十项：BFCL、TAU2、LiveCodeBench、HMMT Nov、HMMT Feb、PolyMATH、SuperGPQA、MMMLU、HLE、AA-LCR。同一 benchmark 的两个模型并行，benchmark 之间串行。HLE w/ CoT 和 AA-LCR 在 smoke 中允许被测模型自评，只验证 judge 链路；正式运行仍要求独立 judge。

状态统一写入 node1 NVMe：

```bash
ssh node1 "tail -n 30 /mnt/nvme0/llm-deploy-eval-logs/modelcard/p1-ranked/${RUN_ID}/suite_status.tsv"
```

其中 `suite_status.tsv` 保留所有尝试，`latest_status.tsv` 对同一 benchmark/model/mode 取最后状态并按优先级排序。同一 run id 再次启动会跳过 `SUCCESS`，重新检查 `FAILED` 与 `BLOCKED_*`；dry-run 也不会导致真实运行被跳过。

如只需单项，可传 benchmark 显示名：

```bash
bash scripts/run_remote_p1_ranked_tests.sh \
  --mode smoke \
  --benchmark "BFCL-V4" \
  --run-id "p1-bfcl-smoke-$(date +%Y%m%d-%H%M%S)"
```

该入口只同步配置、runner 和 smoke fixture，不同步 `models/`、历史日志或整个 Python 环境。node1 首次使用前仍需准备 EvalScope 1.6.1、BFCL scorer、TAU2 v0.2.0 数据和 LiveCodeBench 的 `python:3.11-slim` Docker 镜像，不得把“sandbox pool 为空”当成成功。

HMMT Nov/Feb 在旧 smoke 中已证明 16,384 会截断，新参数把竞赛数学预算提高到 81,920。runner 会把已知 sandbox 错误和“所有 prediction 均以 `max_tokens`/`length` 结束”强制转为失败，即使 EvalScope 自身退出码为 0。

### 11.3 Smoke 通过标准

每个“模型 + benchmark”必须满足：

- runner 退出码为 0，状态为 `SUCCESS`；
- `predictions/`、`reviews/`、`reports/` 和 `progress.json` 存在；
- 没有未解释的 HTTP 错误、超时、空输出或解析失败；
- 输出没有全部停在 `max_tokens`；
- 多模态、代码、judge、Agent 任务通过自己的 gate；
- 分数只标记为 smoke，不写入正式模型卡。

检查命令：

```bash
RUN_DIR="logs/eval/modelcard/${RUN_ID}"
column -s $'\t' -t "${RUN_DIR}/suite_status.tsv" | less -S
find "${RUN_DIR}" -type f \
  \( -name 'progress.json' -o -name '*.jsonl' -o -path '*/reports/*.json' \) \
  | sort
```

## 12. 第八步：运行正式全量测试

### 12.1 P0 全量

只有 P0 smoke 验收通过后才启动：

```bash
RUN_ID="modelcard-p0-full-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode full \
  --run-id "${RUN_ID}" \
  --eval-batch-size 4 \
  --benchmark-parallel 1
```

调度方式：

- 同一 benchmark 的 Qwen 与 ThinkingCap 同时运行。
- 每个端点并发 4，总请求并发 8。
- benchmark 之间串行，避免多个任务争抢端点和混淆日志。

只跑单项：

```bash
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode full \
  --benchmark gpqa_diamond \
  --run-id "${RUN_ID}" \
  --eval-batch-size 4
```

### 12.2 P1 全量

```bash
RUN_ID="p1-ranked-full-$(date +%Y%m%d-%H%M%S)"
bash scripts/run_remote_p1_ranked_tests.sh \
  --mode full \
  --run-id "${RUN_ID}" \
  --eval-batch-size 1 \
  --background
```

未配置独立 judge、sandbox、TAU2 或外部环境的任务会保留 `BLOCKED_*`，不会生成伪正式分数。

P1 的总状态位于 node1 的 `/mnt/nvme0/llm-deploy-eval-logs/modelcard/p1-ranked/<run-id>/suite_status.tsv`，每个任务的原始输出位于该 run id 下对应的任务和模型目录。

只运行远端 Agent 重点项：

```bash
RUN_ID="p1-agent-full-$(date +%Y%m%d-%H%M%S)"
REMOTE_HOST=root@10.16.11.24 bash scripts/run_remote_p1_agent_tests.sh \
  --mode full --task all --eval-batch-size 1 \
  --run-id "${RUN_ID}" --background
```

输出位于 node1 的 `logs/eval/modelcard/p1-agent/<run-id>/`。BFCL full 是 20 个无凭据子集，不包含 `web_search_base` 和 `web_search_no_snippet`，必须标记为 partial；TAU2 full 是三个领域共 269 题。若模型把工具调用写进普通 `content` 而没有 OpenAI `tool_calls`，应记录为协议失败，不在评测后处理阶段静默改写。ThinkingCap 已知的长 system prompt 顺序冲突应在服务层使用受审计的 `config/chat_templates/thinkingcap_agent.jinja` 修复，并通过 `scripts/start_thinkingcap_docker.sh` 加载；修改模板或重启容器后必须先重跑 TAU2 smoke，不能直接沿用修复前的失败/成功状态。

### 12.3 BFCL / TAU2 生成参数与原始响应审计

生成参数按 benchmark 和模型分别冻结。BFCL 两模型都使用 non-thinking Function Calling 配置；TAU2 的 Qwen 使用 thinking，ThinkingCap 使用较低随机性的 Function Calling 采样并关闭 thinking。Function Calling 不设 `preserve_thinking=true`：BFCL 上游对 FC 模型省略历史 reasoning，实测强制回灌后 ThinkingCap 会把 JSON、`<core_memory_replace>` 或残缺 `<tool_call>` 写进普通 content，反而损害原生 OpenAI tool protocol。评测端不再注入过小的 4,096-token 上限，而是显式使用 Qwen 官方 Chat Completions 示例的 `max_tokens=81920`；它保留模型完整输出预算，同时避免循环响应一直扩展到整个 262,144-token 上下文窗口。

81,920 是模型卡预算，不是 BFCL 官方硬规则。如果样本达到该上限，必须在审计中标记为 `finish_reason=length`，不能当成正常完成。不能使用“完全不设预算”代替合理高预算：vLLM 0.24 的服务端 `override_generation_config` 只识别 `max_new_tokens`作为缺省上限，现有 `max_tokens` 字段并未成为缺省值；请求不传上限时，循环响应会持续到剩余上下文长度或 HTTP timeout。

TAU2 的用户模拟器固定复用稳定 Qwen 端点，使用 `temperature=0, enable_thinking=false`，且不传 `max_tokens`。用户模拟器只生成简短对话，关闭 thinking 可避免只产生 reasoning 而用户正文为空；不传输出上限则避免在共享交互环境里再注入 4K 人为预算。被测 Qwen 保持 thinking；ThinkingCap 因相同的 reasoning-only 空响应证据关闭 thinking，但两者都保留 81K 输出预算。不能为了某一被测模型单独调整 user simulator。

BFCL 通过 `benchmarks/modelcard/run_evalscope_with_bfcl_patch.py` 安装兼容补丁，原始审计位于每个模型任务目录下的 `audit/openai_responses.jsonl`。每条至少保存 `finish_reason`、`content`、`reasoning_content`、`tool_calls`、请求参数和用量；新正式轮次应记录 `temperature=0.7`、`max_tokens_source=request` 且 `max_tokens=81920`。补丁不强制 `tool_choice=required`，也不把普通正文改写为工具调用。

`Failed to decode the model response. Proceed to next turn.` 是 BFCL 上游将当前响应判定为非函数调用时的通用日志；在 Agentic 任务中，最后一条自然语言回答本来就是评分对象。因此不能用该日志行数代替原始响应审计，也不能等同于“ThinkingCap 没有调工具”。

ThinkingCap 的服务端 `thinkingcap_agent.jinja` 必须在 system 末尾重申原生 Qwen tool protocol。除了禁止 `<invoke>` 和 JSON code block，还必须禁止实测出现的 `<tool_code>`、Python 函数写法和其他伪 XML，并明确工具调用不能与用户正文混在同一轮。该修复仅约束“已决定调工具”时的输出格式，不强制每轮调用。

## 13. 第九步：监控运行

P0 输出目录：

```text
logs/eval/modelcard/<run-id>/
├── environment.json
├── commands.log
├── suite_status.tsv
├── issue_log.tsv
└── <benchmark>/<model>/
    ├── command.sh
    ├── runner.log
    ├── progress.json
    ├── frozen_config/
    ├── predictions/
    ├── reviews/
    └── reports/
```

查看状态和进度：

```bash
RUN_DIR="logs/eval/modelcard/${RUN_ID}"
tail -n 30 "${RUN_DIR}/suite_status.tsv"
tail -f "${RUN_DIR}/mmlu_pro/qwen/runner.log"
jq . "${RUN_DIR}/mmlu_pro/qwen/progress.json"
```

查看 node1 服务是否收到请求：

```bash
ssh node1 'docker logs --since 1m -f qwen3.6-27b-fp8 2>&1' \
  | grep --line-buffered -E 'Avg prompt throughput|SpecDecoding metrics|Running|Waiting'
```

判断原则：

- `progress.json` 在更新：评测仍在推进。
- 远端 `Running=0` 且本地进度长时间不更新：优先检查本地 runner/PID。
- `progress=100%` 不等于结果已验收，仍要检查 report、题量和异常。
- `suite_status.tsv` 只在整项结束后写入 `SUCCESS` 或 `FAILED`。

## 14. 第十步：中断和续跑

不要删除原 run 目录。使用相同 run id 和完全相同的冻结配置：

```bash
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode full \
  --run-id "${RUN_ID}" \
  --eval-batch-size 4 \
  --resume
```

续跑规则：

- 已有 `SUCCESS` 的“模型 + benchmark”直接跳过。
- 未完成任务若存在 prediction/review，会先比较 `frozen_config/`。
- 配置一致时传给 EvalScope `--use-cache`，只处理缺失样本。
- 配置不一致时拒绝混用；应创建新 run id。

当前 P0 剩余任务的后台入口：

```bash
bash benchmarks/modelcard/run_remaining_evaluations.sh \
  --run-id modelcard-core-full-20260714 \
  --background
```

默认顺序为 MMLU-Redux、C-Eval、LongBench v2。后台日志：

```bash
tail -f logs/eval/modelcard/modelcard-core-full-20260714/resume-launcher.log
```

不要重复启动同一个 run id。`run_remaining_evaluations.sh` 会做重复启动检查，但启动前仍应查看 PID、日志更新时间和端点请求数。

## 15. 第十一步：验收正式结果

一个 benchmark 只有通过以下检查才能写入模型卡。

### 15.1 状态和文件

- Qwen 和 ThinkingCap 都有对应 full phase 的 `SUCCESS`。
- prediction、review、report、runner.log、command.sh 和 frozen_config 完整。
- 实际执行命令与计划一致。

### 15.2 题量

同时记录：

```text
官方/原始题量
计划题量
生成题量
实际计分题量
排除题量及原因
```

不能把生成题量当成计分题量。例如 IFBench 可能生成 300 条但汇总只计分 294 条，必须同时保留两个分母。

查找报告：

```bash
find "${RUN_DIR}" -path '*/reports/*' -type f -name '*.json' | sort
find "${RUN_DIR}" -path '*/predictions/*' -type f -name '*.jsonl' | sort
find "${RUN_DIR}" -path '*/reviews/*' -type f -name '*.jsonl' | sort
```

### 15.3 配置一致性

两模型应具有相同的：

- dataset args、subset、split、few-shot；
- generation config、seed、timeout 和输出预算；
- prompt/chat template、答案抽取和评分器；
- judge、sandbox、用户模拟器和外部 worker 版本。

### 15.4 异常和截断

必须核对：

- HTTP 失败、超时和重试；
- 空答案、解析失败和未评分样本；
- 达到 `max_tokens` 的样本；
- 上下文超限和被过滤样本；
- judge 失败、sandbox 失败和工具调用失败。

达到 `max_tokens` 的错误不能直接解释为模型基础能力差异。LongBench v2 只验收预算内样本，同时保留超限清单。

### 15.5 可比性结论

- 本地两模型：只有同配置、同题量后才可做 A/B。
- Qwen 122B 官方分：根据已公开协议标记 `Strict` 到 `Not reproducible`。
- smoke、自评 judge、近似数据和部分 subset 不能混入正式主分数。

## 16. 第十二步：写入模型卡

更新顺序：

1. 在 [跑分对比表](../benchmark-comparison.md) 写主分数、实际计分题量和差值。
2. 在本手册只补充可复用的新流程或限制，不维护实时进度。
3. 在 `PROJECT_LOG.md` 和当日项目日志记录运行、错误、修复和结论。
4. 原始结果继续保留在 `logs/eval/`，不复制进报告目录。

单项结果至少记录：

```text
Benchmark：
Run ID / 完成时间：
模型与 served name：
Dataset revision / subset / split：
Few-shot / prompt / chat template：
temperature / top_p / top_k / max_tokens / seed：
计划 / 生成 / 计分 / 排除题量：
两个模型主分数和绝对差：
失败 / 超时 / 解析失败 / 截断：
是否复用 cache：
122B 官方参考分和可比性等级：
prediction / review / report / frozen_config 路径：
结论与限制：
```

## 17. 常见问题

| 现象 | 优先判断 | 处理 |
|---|---|---|
| `/v1/models` 模型名不符 | 8001 当前运行了其他模型 | 停止本轮，不要误发请求 |
| 两模型进度同时停止 | 本地 runner/终端退出 | 检查 PID、runner.log，再使用同 run id `--resume` |
| 远端容器正常但 `Running=0` | 当前没有评测请求 | 检查本地进程和 progress 更新时间 |
| 冻结配置不一致 | 同 run id 参数变化 | 新建 run id，不强制复用 cache |
| IFBench 生成数与计分数不同 | adapter 有样本未进入指标 | 同时报告两个分母并列出差异 |
| LongBench 上下文超限 | 输入加输出超过 262,144 | 重新预处理，不静默裁剪 |
| judge 任务被阻塞 | 未配置独立 judge | smoke 可自评；full 必须独立 judge |
| 代码任务被阻塞 | 未启用或无法访问 Docker | 修复隔离 sandbox 后显式启用 |
| `BLOCKED_EXTERNAL` | 缺搜索、视频、GUI 等 worker | 单独建设环境，不用普通 adapter 替代 |
| TAU2 报空 `AssistantMessage` | 上一轮工具调用落在普通 `content` | 查看原始轨迹；ThinkingCap 检查服务端 `thinkingcap_agent.jinja` 是否加载并先重跑 smoke，不在评测端改写正文 |

调试时记录：执行命令、原始错误、初步原因、尝试的修复、最终结果、可比性影响和证据路径。

## 18. 最小可执行顺序

新环境首次跑 P0，可以按下面的顺序执行：

```bash
# 1. 建 SSH 隧道（单独终端）
ssh -N -L 18000:127.0.0.1:8000 -L 18001:127.0.0.1:8001 node1

# 2. 进入项目并设置端点
cd /home/yangyuxin/llm-deploy
export QWEN_API_URL=http://127.0.0.1:18000/v1
export THINKINGCAP_API_URL=http://127.0.0.1:18001/v1
export QWEN_SERVED_MODEL=qwen3.6-27b-fp8
export THINKINGCAP_SERVED_MODEL=thinkingcap-qwen3.6-27b-fp8

# 3. 创建环境并安装 P0 依赖
uv python install 3.10.20
uv venv --python 3.10.20 .eval-deps/evalscope-1.6.1
uv pip install --python .eval-deps/evalscope-1.6.1/bin/python \
  'evalscope[perf,app,ifeval,ifbench]==1.6.1' 'setuptools<81'

# 4. dry-run
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode smoke --run-id modelcard-p0-dry-run --dry-run

# 5. 下载并准备 LongBench 本地长度安全子集
.eval-deps/evalscope-1.6.1/bin/python - <<'PY'
from modelscope.msdatasets import MsDataset
dataset = MsDataset.load('ZhipuAI/LongBench-v2', subset_name='default', split='train')
assert len(dataset) == 503, len(dataset)
PY
.eval-deps/evalscope-1.6.1/bin/python \
  benchmarks/modelcard/prepare_longbench_inlimit.py

# 6. smoke
RUN_ID="modelcard-p0-smoke-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode smoke --run-id "${RUN_ID}" --eval-batch-size 2

# 7. full
RUN_ID="modelcard-p0-full-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode full --run-id "${RUN_ID}" \
  --eval-batch-size 4 --benchmark-parallel 1
```

第 5 步 smoke 未通过时，不进入第 7 步 full。

## 19. 安全边界

- 绝对不重启 H100 宿主机；只允许按部署流程操作容器。
- 评测脚本本身不得启动、停止或重启模型服务。
- 不删除、移动或重写 `models/` 下的模型权重。
- 不保存 SSH key、token 或外部 judge 凭据。
- 代码、搜索、GUI、Android 和 Agent worker 与模型宿主机隔离。
- `BLOCKED` 表示环境或协议缺口，不表示模型能力失败。
