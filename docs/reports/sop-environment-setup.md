# SOP-A:评测环境从零搭建手册

本手册覆盖从一台干净的本地工作机和一台已部署模型服务的 node1(10.16.11.24)出发,搭建出可复现评测能力并跑出第一版模型卡的完整流程。所有命令均为可直接执行的精确命令,不是样例。

被测对象固定为两个:
- `qwen3.6-27b-fp8`(端口 8000,GPU 0,1,TP=2)
- `thinkingcap-qwen3.6-27b-fp8`(端口 8001,GPU 2,3,TP=2)

## 0. 前置条件

开始前必须满足以下条件,否则本手册无法继续。

### 0.1 本地工作机

```bash
# 确认本地工作目录
ls /home/yangyuxin/llm-deploy/AGENTS.md

# 确认 uv 已安装
command -v uv && uv --version

# 确认 rsync、ssh、docker、jq 可用
command -v rsync ssh docker jq

# 确认能 ssh 到 node1(无密码密钥登录)
ssh -o BatchMode=yes -o ConnectTimeout=5 root@10.16.11.24 'hostname && uname -a'
```

### 0.2 node1 模型服务(只读检查,不重启)

```bash
# 在 node1 上检查两个目标容器
ssh root@10.16.11.24 'docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "qwen3.6-27b-fp8|thinkingcap-qwen3.6-27b-fp8"'

# 在 node1 上验证 8000 端点返回的 served name
ssh root@10.16.11.24 'curl -fsS --noproxy "*" http://127.0.0.1:8000/v1/models | jq -r ".data[].id"'

# 在 node1 上验证 8001 端点返回的 served name
ssh root@10.16.11.24 'curl -fsS --noproxy "*" http://127.0.0.1:8001/v1/models | jq -r ".data[].id"'
```

通过标准:
- 两个容器均为 `Up` 状态。
- 8000 返回 `qwen3.6-27b-fp8`。
- 8001 返回 `thinkingcap-qwen3.6-27b-fp8`。

若服务未运行或 served name 不符,停止本手册,先按部署流程处理。**绝对不重启 H100 宿主机**。

## 1. 在本地创建 EvalScope venv

node1 无公网,所有 Python 环境必须在本地建好后 rsync 过去。以下命令在**本地工作机**执行。

```bash
cd /home/yangyuxin/llm-deploy

# 安装 Python 3.10.20(EvalScope 1.6.1 冻结版本)
uv python install 3.10.20

# 创建 venv(路径固定,后续所有脚本依赖此路径)
uv venv --python 3.10.20 .eval-deps/evalscope-1.6.1

# 安装 EvalScope 基础包
uv pip install --python .eval-deps/evalscope-1.6.1/bin/python \
  'evalscope[perf,app]==1.6.1'

# 安装 P0 评分依赖(IFEval/IFBench 需要)
uv pip install --python .eval-deps/evalscope-1.6.1/bin/python \
  'evalscope[ifeval,ifbench]==1.6.1' \
  'setuptools<81'
```

验证本地 venv:

```bash
.eval-deps/evalscope-1.6.1/bin/python --version
.eval-deps/evalscope-1.6.1/bin/evalscope --version
.eval-deps/evalscope-1.6.1/bin/python -c 'import evalscope; print(evalscope.__version__)'

# 验证 P0 评分依赖
.eval-deps/evalscope-1.6.1/bin/python - <<'PY'
import emoji, langdetect, nltk, syllapy
print('P0 dependencies: OK')
PY
```

## 2. 在本地预下载 P0 数据集缓存

P0 七项(IFEval/IFBench/GPQA/MMLU-Pro/MMLU-Redux/C-Eval/LongBench v2)由 EvalScope adapter 通过 ModelScope/HuggingFace 加载。第一次加载会下载到本地缓存目录,后续同步到 node1。

```bash
cd /home/yangyuxin/llm-deploy

# 准备本地缓存目录(若已有可跳过)
mkdir -p ~/.cache/modelscope ~/.cache/huggingface

# 触发 P0 数据集下载并写入本地缓存
# 注:此步骤需要公网,且只下载数据、不请求模型端点
export EVALSCOPE_CACHE="${HOME}/.cache/evalscope"
mkdir -p "${EVALSCOPE_CACHE}"

# 逐项 dry-run 触发 adapter 下载(只下数据,不请求模型)
for benchmark in ifeval ifbench gpqa_diamond mmlu_pro mmlu_redux ceval; do
  .eval-deps/evalscope-1.6.1/bin/evalscope eval \
    --model qwen3.6-27b-fp8 \
    --model-id qwen3.6-27b-fp8 \
    --api-url http://127.0.0.1:18000/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets "${benchmark}" \
    --eval-batch-size 1 \
    --seed 42 \
    --generation-config '{"temperature":0,"top_p":1.0,"top_k":-1,"max_tokens":8192,"seed":42,"timeout":900,"stream":true,"extra_body":{"chat_template_kwargs":{"enable_thinking":true}}}' \
    --work-dir /tmp/p0-data-prefetch/${benchmark} \
    --no-timestamp \
    --limit 1 \
    --enable-progress-tracker || true
done
```

验证缓存:

```bash
ls ~/.cache/modelscope/hub/datasets/ | head
ls ~/.cache/huggingface/datasets/ | head
```

## 3. 在本地生成 LongBench v2 长度安全子集

LongBench v2 必须按真实 Qwen chat template 计算输入长度,不能用字符数或静默截断。规则:`模板化输入 tokens + 16384 输出预算 ≤ 262144`。

```bash
cd /home/yangyuxin/llm-deploy

# 下载 LongBench v2 原始 503 题并校验(需要公网,不访问模型端点)
.eval-deps/evalscope-1.6.1/bin/python - <<'PY'
from modelscope.msdatasets import MsDataset
dataset = MsDataset.load('ZhipuAI/LongBench-v2', subset_name='default', split='train')
assert len(dataset) == 503, len(dataset)
print('LongBench v2 raw rows:', len(dataset))
PY

# 生成可运行子集(用 Qwen3.6-27B-FP8 tokenizer 真实计算长度)
.eval-deps/evalscope-1.6.1/bin/python \
  benchmarks/modelcard/prepare_longbench_inlimit.py \
  --tokenizer .eval-deps/tokenizers/Qwen3.6-27B-FP8 \
  --max-model-len 262144 \
  --max-output-tokens 16384
```

验证:

```bash
jq . .eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/sample_counts.json
wc -l .eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/default_train.jsonl
wc -l .eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/skipped_context_limit.tsv
```

预期:`expected_to_evaluate=400`、`skipped_context_limit=103`、`default_train.jsonl` 共 400 行。若 tokenizer/chat template/预算变化必须重新生成。

## 4. 在本地准备 P1 固定数据

P1 的 HMMT Nov 25 必须固定 revision,其他数据由 smoke fixture 自动准备。以下在本地执行(需要公网下载 HMMT)。

```bash
cd /home/yangyuxin/llm-deploy

# 下载并校验 HMMT Nov 2025 固定 revision
bash benchmarks/modelcard/prepare_p1_data.sh

# 验证 HMMT Nov 元数据与题量
jq . .eval-deps/data/modelcard-p1/hmmt_nov_2025/dataset_metadata.json
wc -l .eval-deps/data/modelcard-p1/hmmt_nov_2025/default_train.jsonl
```

预期:`default_train.jsonl` 共 30 行,SHA256 与脚本内固定值一致。

## 5. 同步 EvalScope venv 到 node1

uv 创建的 venv 内 python 符号链接和 shebang 是本地绝对路径(`/home/yangyuxin/...`),node1 上是 `/root/...`,必须修复。

```bash
cd /home/yangyuxin/llm-deploy

# 第一步:同步 uv 管理的 Python 3.10(venv 的 python 指向它)
rsync -az ~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/ \
  root@10.16.11.24:/root/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/

# 第二步:同步 venv(保持符号链接,不解引用)
rsync -a .eval-deps/evalscope-1.6.1/ \
  root@10.16.11.24:/root/llm-deploy/.eval-deps/evalscope-1.6.1/

# 第三步:在 node1 上修复 venv 内的绝对路径
ssh root@10.16.11.24 '
  cd /root/llm-deploy/.eval-deps/evalscope-1.6.1
  ln -sf /root/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 bin/python
  grep -rl "/home/yangyuxin/llm-deploy" bin/ \
    | xargs -r sed -i "s|/home/yangyuxin/llm-deploy|/root/llm-deploy|g"
  sed -i "s|/home/yangyuxin/.local|/root/.local|g" pyvenv.cfg
'
```

验证 node1 上 venv 可用:

```bash
ssh root@10.16.11.24 '/root/llm-deploy/.eval-deps/evalscope-1.6.1/bin/evalscope --version'
ssh root@10.16.11.24 '/root/llm-deploy/.eval-deps/evalscope-1.6.1/bin/python -c "import evalscope; print(evalscope.__version__)"'
```

## 6. 同步数据集缓存到 node1

把本地下载的 ModelScope/HF 缓存同步到 node1 NVMe。

```bash
# ModelScope 缓存(P0 大部分数据集)
ssh root@10.16.11.24 'mkdir -p /mnt/nvme0/llm-deploy-eval-deps/modelscope-cache'
rsync -az ~/.cache/modelscope/ \
  root@10.16.11.24:/mnt/nvme0/llm-deploy-eval-deps/modelscope-cache/

# HuggingFace 缓存
ssh root@10.16.11.24 'mkdir -p /mnt/nvme0/llm-deploy-eval-deps/huggingface-cache'
rsync -az ~/.cache/huggingface/ \
  root@10.16.11.24:/mnt/nvme0/llm-deploy-eval-deps/huggingface-cache/

# EvalScope 自身缓存
ssh root@10.16.11.24 'mkdir -p /mnt/nvme0/llm-deploy-eval-deps/evalscope-cache'
rsync -az ~/.cache/evalscope/ \
  root@10.16.11.24:/mnt/nvme0/llm-deploy-eval-deps/evalscope-cache/ 2>/dev/null || true
```

## 7. 同步 LongBench 子集和 P1 数据到 node1

```bash
cd /home/yangyuxin/llm-deploy

# LongBench 子集
ssh root@10.16.11.24 'mkdir -p /root/llm-deploy/.eval-deps/data/longbench-v2-qwen3.6-inlimit-245760'
rsync -az \
  .eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/ \
  root@10.16.11.24:/root/llm-deploy/.eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/

# P1 HMMT Nov 数据(目录结构保持一致)
ssh root@10.16.11.24 'mkdir -p /root/llm-deploy/.eval-deps/data/modelcard-p1/hmmt_nov_2025'
rsync -az \
  .eval-deps/data/modelcard-p1/hmmt_nov_2025/ \
  root@10.16.11.24:/root/llm-deploy/.eval-deps/data/modelcard-p1/hmmt_nov_2025/
```

## 8. 同步 TAU2 数据到 node1

TAU2-Bench 需要 v0.2.0 数据。

```bash
cd /home/yangyuxin/llm-deploy

# 确认本地已有 TAU2 数据
ls .eval-deps/tau2-data-v0.2.0/tau2/domains

# 同步到 node1
ssh root@10.16.11.24 'mkdir -p /mnt/nvme0/llm-deploy-eval-deps/tau2-data-v0.2.0'
rsync -az .eval-deps/tau2-data-v0.2.0/ \
  root@10.16.11.24:/mnt/nvme0/llm-deploy-eval-deps/tau2-data-v0.2.0/
```

## 9. 在 node1 上安装 BFCL MiniLM 模型并打 patch

BFCL 的 `memory_vector`、`memory_rec_sum` 子集需要 `all-MiniLM-L6-v2`。node1 无公网时 `SentenceTransformer` 默认构造函数会尝试联网下载并卡数十分钟。必须本地下载后上传并 patch。

```bash
cd /home/yangyuxin/llm-deploy

# 本地下载 all-MiniLM-L6-v2(约 90MB)
mkdir -p /tmp/all-MiniLM-L6-v2
.eval-deps/evalscope-1.6.1/bin/python - <<'PY'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("/tmp/all-MiniLM-L6-v2")
print("saved to /tmp/all-MiniLM-L6-v2")
PY

# 上传到 node1
ssh root@10.16.11.24 'mkdir -p /mnt/nvme0/models/all-MiniLM-L6-v2'
rsync -az /tmp/all-MiniLM-L6-v2/ \
  root@10.16.11.24:/mnt/nvme0/models/all-MiniLM-L6-v2/

# patch node1 上的 memory_vector.py:26
ssh root@10.16.11.24 'sed -i "s|SentenceTransformer(\"all-MiniLM-L6-v2\"|SentenceTransformer(\"/mnt/nvme0/models/all-MiniLM-L6-v2\"|g" /root/llm-deploy/.eval-deps/evalscope-1.6.1/lib/python3.10/site-packages/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_vector.py'
```

验证 patch 已生效:

```bash
ssh root@10.16.11.24 'grep -n "all-MiniLM-L6-v2" /root/llm-deploy/.eval-deps/evalscope-1.6.1/lib/python3.10/site-packages/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_vector.py'
```

预期输出:`26:ENCODER = SentenceTransformer("/mnt/nvme0/models/all-MiniLM-L6-v2", device="cpu")`。

> 此 patch 位置在 site-packages 中,**重装 EvalScope venv 会丢失**。重新同步 venv 后必须再次应用本步骤。

## 10. 同步项目代码到 node1

P0 在 node1 上手动跑,需要把项目代码同步到 node1 的 `/root/llm-deploy`。

```bash
cd /home/yangyuxin/llm-deploy

rsync -az \
  --exclude 'logs/' \
  --exclude 'models/' \
  --exclude '.git/' \
  --exclude '.eval-deps/wheelhouse' \
  config/ benchmarks/ scripts/ \
  root@10.16.11.24:/root/llm-deploy/
```

代码变更后重跑前应重新同步。

## 11. 全量验证

在本地通过 ssh 验证 node1 上所有资源就绪:

```bash
# venv
ssh root@10.16.11.24 '/root/llm-deploy/.eval-deps/evalscope-1.6.1/bin/evalscope --version'

# 数据集缓存
ssh root@10.16.11.24 'ls /mnt/nvme0/llm-deploy-eval-deps/modelscope-cache/hub/datasets/ | head'

# LongBench
ssh root@10.16.11.24 'wc -l /root/llm-deploy/.eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/default_train.jsonl'

# TAU2
ssh root@10.16.11.24 'test -d /mnt/nvme0/llm-deploy-eval-deps/tau2-data-v0.2.0/tau2/domains && echo TAU2_OK'

# BFCL MiniLM 模型 + patch
ssh root@10.16.11.24 'test -d /mnt/nvme0/models/all-MiniLM-L6-v2 && grep -q "/mnt/nvme0/models/all-MiniLM-L6-v2" /root/llm-deploy/.eval-deps/evalscope-1.6.1/lib/python3.10/site-packages/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_vector.py && echo MINILM_OK'

# HMMT Nov
ssh root@10.16.11.24 'wc -l /root/llm-deploy/.eval-deps/data/modelcard-p1/hmmt_nov_2025/default_train.jsonl'

# 两个端点
ssh root@10.16.11.24 'curl -fsS --noproxy "*" http://127.0.0.1:8000/v1/models | jq -r ".data[].id"'
ssh root@10.16.11.24 'curl -fsS --noproxy "*" http://127.0.0.1:8001/v1/models | jq -r ".data[].id"'
```

全部通过后,环境搭建完成,可进入 SOP-B 执行评测。

## 12. 搭建完成后的首次 dry-run

环境搭好后,先用 dry-run 验证命令和配置能正确生成,不访问端点。

```bash
ssh root@10.16.11.24
cd /root/llm-deploy
export QWEN_API_URL=http://127.0.0.1:8000/v1
export THINKINGCAP_API_URL=http://127.0.0.1:8001/v1
export QWEN_SERVED_MODEL=qwen3.6-27b-fp8
export THINKINGCAP_SERVED_MODEL=thinkingcap-qwen3.6-27b-fp8
export MODELSCOPE_CACHE=/mnt/nvme0/llm-deploy-eval-deps/modelscope-cache
export HF_HOME=/mnt/nvme0/llm-deploy-eval-deps/huggingface-cache
export EVALSCOPE_CACHE=/mnt/nvme0/llm-deploy-eval-deps/evalscope-cache
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

RUN_ID="modelcard-p0-dry-run-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode smoke \
  --run-id "${RUN_ID}" \
  --dry-run

# 检查 dry-run 输出
cat "logs/eval/modelcard/${RUN_ID}/commands.log"
column -s $'\t' -t "logs/eval/modelcard/${RUN_ID}/suite_status.tsv"
```

通过标准:`suite_status.tsv` 中所有任务为 `DRY_RUN`,无意外报错。环境搭建阶段结束。

## 安全红线

- 绝对不重启 H100 宿主机,只允许按部署流程操作容器。
- 评测脚本只请求已有 OpenAI-compatible API,不启动、停止或重启模型服务。
- 不删除、移动或重写 `models/` 下的模型权重。
- 不保存 SSH key、token 或外部 judge 凭据。
- 修改 `thinkingcap_agent.jinja` 或重启 ThinkingCap 容器后,必须先重跑 TAU2 smoke,不能直接沿用修复前的状态。
