# 02 环境配置

本页讲"环境怎么配":评测架构、复现前要齐到什么程度、模型服务检查、端点访问、EvalScope 环境、node1 离线时的环境与数据同步、按需依赖。数据准备见 [03](03-data-preparation.md),跑起来见 [05](05-execution-guide.md)。

## 评测架构

**所有评测都在 node1 上跑**(本地内存不够会死机)。区别只是触发方式:

```text
方式 A:P0,ssh 进 node1 手动跑
┌─────────────────┐  ssh    ┌──────────────────────┐
│  本地电脑        │ ──────→ │  node1 (10.16.11.24) │
│  ssh root@10.16.11.24│      │  评测脚本 + 模型服务  │
│  看结果          │ ←────── │  都在这台机器上      │
└─────────────────┘         └──────────────────────┘

方式 B:P1,本地一键触发,远程自动执行
┌─────────────────┐  ssh    ┌──────────────────────┐
│  本地电脑        │ ──────→ │  node1 (10.16.11.24) │
│  bash scripts/   │         │  评测脚本 + 模型服务  │
│  run_remote_p1_* │ ←────── │  都在这台机器上      │
└─────────────────┘         └──────────────────────┘
```

- **P0**(`run_core_evaluations.sh`):`ssh root@10.16.11.24` 进去,在 node1 上手动执行。端点 `127.0.0.1:8000`(本机访问),不需要 SSH 隧道。
- **P1**(`scripts/run_remote_p1_*.sh`):本地一键触发,脚本内部用 `ssh root@10.16.11.24` 把评测命令发到 node1 上执行。端点同样是 `127.0.0.1:8000`,不需要 SSH 隧道。

两种方式的评测进程、模型服务、结果文件都在 node1 上。本地电脑只负责触发和看结果,不跑评测进程,不会占本地内存。

评测脚本只请求已有 OpenAI-compatible API,不启动、停止或重启模型服务,也不修改模型权重。

## 关键文件

| 文件 | 用途 |
|---|---|
| `config/serving.env` | node1 上的 vLLM 配置 |
| `config/evaluation.env` | 评测环境、端点、数据、judge 路径 |
| `benchmarks/modelcard/core_tasks.tsv` | P0 任务和冻结参数 |
| `benchmarks/modelcard/p1_tasks.tsv` | P1 任务、环境和协议状态 |
| `benchmarks/evalscope/reference_benchmarks.tsv` | 82 项统一清单 |
| `run_core_evaluations.sh` | P0 双模型 smoke/full/resume |
| `run_p1_evaluations.sh` | P1 preflight/smoke/full/resume |
| `run_p1_ranked_evaluations.sh` | P1 十项按能力优先级统一调度 |
| `scripts/run_remote_p1_ranked_tests.sh` | 本地一键准备数据、同步 runner 并在 node1 执行 |
| `run_reference_suite.sh` | 82 项单模型预检和调度 |

## 复现前准备 Checklist

开始前应确认以下条件齐备(各项细节见后续小节):

- [ ] **node1 服务健康**:两个目标容器 running/healthy,8000 返回 `qwen3.6-27b-fp8`,8001 返回当前实验模型的 served name
- [ ] **SSH 到 node1 可用**:`ssh root@10.16.11.24` 能登录(所有评测都通过 ssh 在 node1 上执行)
- [ ] **node1 上端点可达**:在 node1 上 `curl http://127.0.0.1:8000/v1/models` 能返回(P0 和 P1 都用本机访问,不需要隧道)
- [ ] **node1 上 EvalScope 1.6.1 venv**:Python 3.10.20 + `evalscope[perf,app]==1.6.1`(环境在 node1 的 `.eval-deps/` 下;node1 离线时从本地同步,见下方"node1 离线时的环境与数据同步")
- [ ] **node1 上 P0 依赖**:`evalscope[ifeval,ifbench]==1.6.1`、`setuptools<81`
- [ ] **node1 上数据集缓存**:ModelScope/HuggingFace 缓存已同步到 node1 NVMe(node1 离线时必须,否则 adapter 无法下载数据)
- [ ] **node1 上 P1 按需依赖**:sandbox(Docker)、TAU2 数据、独立 judge(按要跑的任务准备,见 [01 状态列](01-benchmark-inventory.md))
- [ ] **数据冻结**:LongBench 本地长度安全子集、HMMT 固定 revision(见 [03](03-data-preparation.md))
- [ ] **协议冻结**:`core_tasks.tsv` / `p1_tasks.tsv` 已确认(见 [04](04-evaluation-protocol.md))

只跑 P0 时,P1 的 `BLOCKED_*` 可暂时存在,不能误判为 P0 不可运行。

## 第一步:检查模型服务

先 ssh 进 node1,阅读当前目标和配置:

```bash
ssh root@10.16.11.24
cd /root/llm-deploy
sed -n '1,120p' PROJECT_LOG.md
sed -n '1,260p' config/serving.env
```

检查容器状态(在 node1 上执行):

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
docker logs --tail 30 qwen3.6-27b-fp8
docker logs --tail 30 thinkingcap-qwen3.6-27b-fp8
```

通过标准:

- 两个目标容器均为 running/healthy。
- 8000 返回 `qwen3.6-27b-fp8`。
- 8001 返回当前实验模型的 served name。
- 端点模型名与 `config/evaluation.env` 一致。

这里只做只读检查。若服务未运行,应先停止评测并按部署流程处理,不要由评测脚本自动重启服务。

## 第二步:在 node1 上配置端点访问

所有评测都在 node1 上跑,端点是 `127.0.0.1:8000`(本机访问),不需要 SSH 隧道。

### P0:ssh 进 node1 手动执行

```bash
# 从本地 ssh 进 node1
ssh root@10.16.11.24

# 在 node1 上进入项目目录
cd /root/llm-deploy
export QWEN_API_URL=http://127.0.0.1:8000/v1
export THINKINGCAP_API_URL=http://127.0.0.1:8001/v1
export QWEN_SERVED_MODEL=qwen3.6-27b-fp8
export THINKINGCAP_SERVED_MODEL=thinkingcap-qwen3.6-27b-fp8
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
```

验证端点:

```bash
curl -fsS "${QWEN_API_URL}/models" | jq .
curl -fsS "${THINKINGCAP_API_URL}/models" | jq .
```

如果第二个端点实际运行的不是 ThinkingCap,必须修改 `THINKINGCAP_SERVED_MODEL` 或停止本轮测试;不能把请求误发给同端口的其他模型。

### P1:本地一键触发,不需要配端点

P1 的三个远程入口(`scripts/run_remote_p1_*.sh`)内部用 `ssh root@10.16.11.24` 把评测命令发到 node1 上执行,端点直接是 `127.0.0.1:8000`(node1 本机访问)。你本地只需要 `ssh root@10.16.11.24` 能登录,不需要手动 ssh 进去,也不需要 export 端点变量。

脚本会自动设置远端环境变量,包括:

```text
QWEN_API_URL=http://127.0.0.1:8000/v1
THINKINGCAP_API_URL=http://127.0.0.1:8001/v1
P1_TAU2_USER_API_URL=http://127.0.0.1:8000/v1
```

详见 [05 执行流程](05-execution-guide.md)的脚本速查表。

## 第三步:在 node1 上配置 EvalScope 环境

以下命令都在 node1 上执行(先 `ssh root@10.16.11.24` 进去)。

> **node1 无公网。** 若 venv 尚未同步到 node1,不要直接在 node1 上 `uv pip install`,会卡住。应先在本地建好 venv,按本页"node1 离线时的环境与数据同步"同步过来;数据缓存同理。

### 创建基础环境

当前冻结环境为 Python 3.10.20、EvalScope 1.6.1:

```bash
command -v uv
uv python install 3.10.20
uv venv --python 3.10.20 .eval-deps/evalscope-1.6.1
uv pip install --python .eval-deps/evalscope-1.6.1/bin/python \
  'evalscope[perf,app]==1.6.1'
```

如果 `uv` 尚未安装,应先按 uv 官方方式安装,不要改用系统 Python 污染项目环境。

验证基础环境:

```bash
.eval-deps/evalscope-1.6.1/bin/python --version
.eval-deps/evalscope-1.6.1/bin/evalscope --version
.eval-deps/evalscope-1.6.1/bin/python -c 'import evalscope; print(evalscope.__version__)'
```

### 安装 P0 依赖

P0 的 IFEval 和 IFBench 需要额外评分依赖:

```bash
uv pip install --python .eval-deps/evalscope-1.6.1/bin/python \
  'evalscope[ifeval,ifbench]==1.6.1' \
  'setuptools<81'
```

验证:

```bash
.eval-deps/evalscope-1.6.1/bin/python - <<'PY'
import emoji
import langdetect
import nltk
import syllapy
print('P0 dependencies: OK')
PY
```

### 按需安装 P1–P3 依赖

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

全量环境检查:

```bash
bash benchmarks/evalscope/setup_eval_environments.sh check
```

这个检查覆盖 Language、Vision、WMT、Terminal、Docker 和 TAU2;如果当前只跑 P0,P1–P4 的 `MISSING` 可以暂时存在,不能误判为 P0 不可运行。

## node1 离线时的环境与数据同步

node1 无法访问公网。以下资源必须在本地准备好后传到 node1。P1 远程入口脚本只自动同步代码、smoke fixture 和 HMMT 数据,其余需要手动同步。

### 资源清单

| 资源 | 本地路径 | node1 路径 | 同步方式 |
|---|---|---|---|
| uv 管理的 Python 3.10 | `~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/` | `/root/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/` | rsync |
| EvalScope venv | `.eval-deps/evalscope-1.6.1/` | `/root/llm-deploy/.eval-deps/evalscope-1.6.1/` | rsync + 修复路径 |
| ModelScope 缓存 | `~/.cache/modelscope/` | `/mnt/nvme0/llm-deploy-eval-deps/modelscope-cache/` | rsync |
| HF 缓存 | `~/.cache/huggingface/` | `/mnt/nvme0/llm-deploy-eval-deps/huggingface-cache/` | rsync |
| LongBench 子集 | `.eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/` | `/root/llm-deploy/.eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/` | rsync |
| TAU2 数据 | `.eval-deps/tau2-data-v0.2.0/` | `/mnt/nvme0/llm-deploy-eval-deps/tau2-data-v0.2.0/` | rsync |
| HMMT Nov 数据 | `.eval-deps/data/modelcard-p1/hmmt_nov_2025/` | `${REMOTE_DATA_ROOT}/hmmt_nov_2025.jsonl` | P1 入口自动 |
| P1 smoke fixture | `.eval-deps/data/modelcard-p1/smoke/` | `${REMOTE_DATA_ROOT}/` | P1 入口自动 |
| **P1 full 数据集** | `.eval-deps/data/p1-full-data/` | `/root/llm-deploy/.eval-deps/data/p1-full-data/` | **手动 rsync**(6 个数据集,见 [03](03-data-preparation.md)) |
| **BFCL MiniLM 模型** | 本地下载的 `all-MiniLM-L6-v2/` | `/mnt/nvme0/models/all-MiniLM-L6-v2/` | **手动 rsync + patch memory_vector.py**(见 [03](03-data-preparation.md)) |
| 项目代码 | `config/`、`benchmarks/`、`scripts/` | `/root/llm-deploy/` | P1 自动;P0 手动 |
| Docker 镜像 | 本地 `docker images` | node1 `docker load` | save/load |

模型权重(`QWEN_TOKENIZER_PATH=/mnt/nvme0/models/Qwen3.6-27B-FP8`、`THINKINGCAP_TOKENIZER_PATH=/mnt/nvme0/models/ThinkingCap-Qwen3.6-27B-FP8`)由部署阶段准备,不在评测同步范围。

### 同步 EvalScope venv

uv 创建的 venv 里 python 符号链接和脚本 shebang 都是本地绝对路径(`/home/yangyuxin/...`),node1 上路径是 `/root/...`。直接 rsync 后要修复路径才能用。

第一步:传 uv 管理的 Python 3.10(venv 的 python 指向它):

```bash
rsync -az ~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/ \
  root@10.16.11.24:/root/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/
```

第二步:传 venv(保持符号链接,不解引用):

```bash
rsync -a .eval-deps/evalscope-1.6.1/ \
  root@10.16.11.24:/root/llm-deploy/.eval-deps/evalscope-1.6.1/
```

第三步:在 node1 上修复路径:

```bash
ssh root@10.16.11.24 '
  cd /root/llm-deploy/.eval-deps/evalscope-1.6.1
  ln -sf /root/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 bin/python
  grep -rl "/home/yangyuxin/llm-deploy" bin/ \
    | xargs -r sed -i "s|/home/yangyuxin/llm-deploy|/root/llm-deploy|g"
  sed -i "s|/home/yangyuxin/.local|/root/.local|g" pyvenv.cfg
'
```

验证:

```bash
ssh root@10.16.11.24 '/root/llm-deploy/.eval-deps/evalscope-1.6.1/bin/evalscope --version'
ssh root@10.16.11.24 '/root/llm-deploy/.eval-deps/evalscope-1.6.1/bin/python -c "import evalscope; print(evalscope.__version__)"'
```

P1 按需环境(语言、视觉、翻译、Terminal)如果本地也装了,用同样方式 rsync + 修复路径;或者只传 wheel 到 node1,在 node1 上 `pip install --no-index` 离线装。

### 同步数据集缓存

P0 的 IFEval、IFBench、GPQA、MMLU-Pro、MMLU-Redux、C-Eval 和 LongBench v2 由 EvalScope adapter 通过 ModelScope/HuggingFace 加载。第一次 smoke 时本地会下载并缓存,把缓存传到 node1 后直接复用,不再联网:

```bash
# ModelScope 缓存(P0 大部分数据集)
ssh root@10.16.11.24 'mkdir -p /mnt/nvme0/llm-deploy-eval-deps/modelscope-cache'
rsync -az ~/.cache/modelscope/ \
  root@10.16.11.24:/mnt/nvme0/llm-deploy-eval-deps/modelscope-cache/

# HuggingFace 缓存
ssh root@10.16.11.24 'mkdir -p /mnt/nvme0/llm-deploy-eval-deps/huggingface-cache'
rsync -az ~/.cache/huggingface/ \
  root@10.16.11.24:/mnt/nvme0/llm-deploy-eval-deps/huggingface-cache/
```

P1 远程入口已自动设置 `MODELSCOPE_CACHE`、`HF_HOME`、`EVALSCOPE_CACHE` 指向这些路径。P0 在 node1 上手动跑时需要先 export:

```bash
# ssh root@10.16.11.24 后,跑 P0 前 export
export MODELSCOPE_CACHE=/mnt/nvme0/llm-deploy-eval-deps/modelscope-cache
export HF_HOME=/mnt/nvme0/llm-deploy-eval-deps/huggingface-cache
export EVALSCOPE_CACHE=/mnt/nvme0/llm-deploy-eval-deps/evalscope-cache
```

### 同步 LongBench 子集

LongBench 子集是本地预处理的 JSONL,直接 rsync 到 node1 同路径:

```bash
ssh root@10.16.11.24 'mkdir -p /root/llm-deploy/.eval-deps/data/longbench-v2-qwen3.6-inlimit-245760'
rsync -az \
  .eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/ \
  root@10.16.11.24:/root/llm-deploy/.eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/
```

### 同步 TAU2 数据

```bash
ssh root@10.16.11.24 'mkdir -p /mnt/nvme0/llm-deploy-eval-deps/tau2-data-v0.2.0'
rsync -az .eval-deps/tau2-data-v0.2.0/ \
  root@10.16.11.24:/mnt/nvme0/llm-deploy-eval-deps/tau2-data-v0.2.0/
```

### 同步 P0 代码

P0 在 node1 上手动跑,需要先把项目代码同步到 node1 的 `/root/llm-deploy`(P1 远程入口会自动 rsync 所需代码,P0 不经过远程入口):

```bash
rsync -az \
  --exclude 'logs/' \
  --exclude 'models/' \
  --exclude '.git/' \
  --exclude '.eval-deps/wheelhouse' \
  config/ benchmarks/ scripts/ \
  root@10.16.11.24:/root/llm-deploy/
```

代码变更后重跑前也应重新同步。`config/evaluation.env` 里的端点默认值是 SSH 隧道端口(`127.0.0.1:18000`),P0 在 node1 上跑时应显式覆盖为本机直连:

```bash
# ssh root@10.16.11.24 后
export QWEN_API_URL=http://127.0.0.1:8000/v1
export THINKINGCAP_API_URL=http://127.0.0.1:8001/v1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

### Docker 镜像(代码 sandbox / LiveCodeBench)

代码任务需要 `python:3.11-slim` 等镜像。node1 连不上网时,在本地 save 后传过去 load:

```bash
# 本地:save 镜像
docker save python:3.11-slim -o /tmp/python-3.11-slim.tar

# 传到 node1
scp /tmp/python-3.11-slim.tar root@10.16.11.24:/tmp/

# node1 上 load
ssh root@10.16.11.24 'docker load -i /tmp/python-3.11-slim.tar && rm /tmp/python-3.11-slim.tar'
```

### 全量验证

```bash
# venv
ssh root@10.16.11.24 '/root/llm-deploy/.eval-deps/evalscope-1.6.1/bin/evalscope --version'
# 数据集缓存
ssh root@10.16.11.24 'ls /mnt/nvme0/llm-deploy-eval-deps/modelscope-cache/hub/datasets/ | head'
# LongBench
ssh root@10.16.11.24 'wc -l /root/llm-deploy/.eval-deps/data/longbench-v2-qwen3.6-inlimit-245760/default_train.jsonl'
# TAU2
ssh root@10.16.11.24 'test -d /mnt/nvme0/llm-deploy-eval-deps/tau2-data-v0.2.0/tau2/domains && echo TAU2_OK'
# Docker
ssh root@10.16.11.24 'docker images python:3.11-slim'
# P1 full 数据集
ssh root@10.16.11.24 'ls /root/llm-deploy/.eval-deps/data/p1-full-data/'
# BFCL MiniLM 模型 + patch
ssh root@10.16.11.24 'test -d /mnt/nvme0/models/all-MiniLM-L6-v2 && grep -q "/mnt/nvme0/models/all-MiniLM-L6-v2" /root/llm-deploy/.eval-deps/evalscope-1.6.1/lib/python3.10/site-packages/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_vector.py && echo MINILM_OK'
```

## 第四步:sandbox、judge 和 Agent 环境(按需)

只有要跑对应任务时才需要配置。要跑什么见 [01 状态列](01-benchmark-inventory.md)。

### 代码 sandbox

代码任务只有同时满足以下条件才允许运行:

```bash
docker info
.eval-deps/evalscope-1.6.1/bin/python -c 'import docker, ms_enclave'
```

- runner 显式传入 `--enable-sandbox`;
- Docker daemon 和当前用户权限可用;
- 代码执行环境与 H100 模型服务隔离。

### 独立 judge

需要 judge 的正式任务(HLE w/ CoT、AA-LCR)必须配置独立端点:

```bash
export EVALSCOPE_JUDGE_MODEL=<judge-served-name>
export EVALSCOPE_JUDGE_API_URL=http://127.0.0.1:<judge-port>/v1
```

`--allow-self-judge` 只用于 smoke,结果不能写入正式模型卡。未配置独立 judge 时,runner 会把 judge 任务标为 `BLOCKED_JUDGE`,不会生成伪正式分数。

### TAU2

TAU2 要求包和数据同时存在:

```bash
test -d .eval-deps/tau2-data-v0.2.0/tau2/domains
.eval-deps/evalscope-1.6.1/bin/python -c 'import tau2; print("TAU2: OK")'
```

两个被测模型必须共用 `config/evaluation.env` 中同一个 `P1_TAU2_USER_MODEL`,否则用户模拟行为会污染 A/B。
