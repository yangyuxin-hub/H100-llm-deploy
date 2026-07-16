# SOP-B:环境就绪后的自动化测评执行手册

本手册假设 SOP-A 已完成:node1 上 EvalScope 1.6.1 venv、数据集缓存、LongBench 子集、TAU2 数据、BFCL MiniLM patch、HMMT 数据和项目代码均已就绪。本手册只讲如何触发评测、监控进度、续跑和验收。

被测对象固定为两个:
- `qwen3.6-27b-fp8`(端口 8000,GPU 0,1,TP=2)
- `thinkingcap-qwen3.6-27b-fp8`(端口 8001,GPU 2,3,TP=2)

所有评测进程都在 node1(10.16.11.24)上执行。P0 手动 ssh 进 node1 跑;P1 本地一键触发。

## 0. 执行前只读检查

每次启动评测前,先确认服务健康,不重启任何容器。

```bash
# 检查两个目标容器状态
ssh root@10.16.11.24 'docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "qwen3.6-27b-fp8|thinkingcap-qwen3.6-27b-fp8"'

# 验证 8000 端点
ssh root@10.16.11.24 'curl -fsS --noproxy "*" http://127.0.0.1:8000/v1/models | jq -r ".data[].id"'

# 验证 8001 端点
ssh root@10.16.11.24 'curl -fsS --noproxy "*" http://127.0.0.1:8001/v1/models | jq -r ".data[].id"'
```

通过标准:两个容器 `Up`;8000 返回 `qwen3.6-27b-fp8`;8001 返回 `thinkingcap-qwen3.6-27b-fp8`。不符则停止,先处理服务。

## 1. P0 评测(ssh 进 node1 手动执行)

P0 包含 7 项核心任务:IFEval、IFBench、GPQA Diamond、MMLU-Pro、MMLU-Redux、C-Eval、LongBench v2。顺序执行流程:dry-run → smoke → full → 续跑。

### 1.1 ssh 进 node1 并配置环境

每次开新会话都要执行以下 export:

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
```

### 1.2 P0 dry-run

只生成命令和冻结配置,不访问端点。

```bash
RUN_ID="modelcard-p0-dry-run-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode smoke \
  --run-id "${RUN_ID}" \
  --dry-run

# 检查
cat "logs/eval/modelcard/${RUN_ID}/commands.log"
column -s $'\t' -t "logs/eval/modelcard/${RUN_ID}/suite_status.tsv"
```

通过标准:所有任务状态为 `DRY_RUN`,无报错。

### 1.3 P0 smoke

每个 subset 跑 1 题,验证完整链路。

```bash
RUN_ID="modelcard-p0-smoke-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode smoke \
  --run-id "${RUN_ID}" \
  --eval-batch-size 2

# 检查状态
column -s $'\t' -t "logs/eval/modelcard/${RUN_ID}/suite_status.tsv"

# 检查单项进度与日志
jq . "logs/eval/modelcard/${RUN_ID}/mmlu_pro/qwen/progress.json"
tail -50 "logs/eval/modelcard/${RUN_ID}/mmlu_pro/qwen/runner.log"
```

通过标准:每个"模型 + benchmark"状态为 `SUCCESS`,无 HTTP 错误、空输出或全部停在 `max_tokens`。

也可单跑一项验证:

```bash
RUN_ID="modelcard-mmlu-pro-smoke-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode smoke \
  --benchmark mmlu_pro \
  --run-id "${RUN_ID}" \
  --eval-batch-size 1
```

### 1.4 P0 full

smoke 通过后才启动 full。两模型并行,每端点并发 4,benchmark 间串行。

```bash
RUN_ID="modelcard-p0-full-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode full \
  --run-id "${RUN_ID}" \
  --eval-batch-size 4 \
  --benchmark-parallel 1
```

只跑单项 full:

```bash
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode full \
  --benchmark gpqa_diamond \
  --run-id "${RUN_ID}" \
  --eval-batch-size 4
```

### 1.5 P0 续跑(同 run id)

中断后用相同 run id 和相同冻结配置续跑。已有 `SUCCESS` 自动跳过,配置一致时复用 prediction/review cache。

```bash
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode full \
  --run-id "${RUN_ID}" \
  --eval-batch-size 4 \
  --benchmark-parallel 1 \
  --resume
```

### 1.6 P0 续跑剩余三项(后台)

`run_remaining_evaluations.sh` 串行跑 MMLU-Redux、C-Eval、LongBench v2,默认复用 `modelcard-core-full-20260714` run id。

```bash
# 后台启动
bash benchmarks/modelcard/run_remaining_evaluations.sh \
  --run-id modelcard-core-full-20260714 \
  --background

# 查看后台日志
tail -f logs/eval/modelcard/modelcard-core-full-20260714/resume-launcher.log
```

> 不要重复启动同一 run id。启动前先检查 PID 和日志更新时间。

## 2. P1 评测(本地一键触发)

P1 共 10 项,按能力优先级排序:BFCL-V4、TAU2-Bench、LiveCodeBench v6、HMMT Nov 25、HMMT Feb 25、PolyMATH、SuperGPQA、MMMLU、HLE w/ CoT、AA-LCR(后两项 BLOCKED_JUDGE)。

三个远程入口脚本内部用 `ssh root@10.16.11.24` 把评测命令发到 node1 上执行,端点为 `127.0.0.1:8000`(本机访问),不需要 SSH 隧道。

### 2.1 P1 十项统一调度(推荐)

从本地直接触发,脚本自动同步 runner、配置和 smoke fixture 到 node1。

```bash
cd /home/yangyuxin/llm-deploy

# preflight:只检查端点/adapter/环境/数据/judge/sandbox,不发正式题
bash scripts/run_remote_p1_ranked_tests.sh \
  --mode preflight \
  --run-id "p1-preflight-$(date +%Y%m%d-%H%M%S)" \
  --background
```

处理 preflight 暴露的 `BLOCKED_DATA` / `BLOCKED_ENV` 后再继续。`BLOCKED_EXTERNAL` 不强行解除。

### 2.2 P1 smoke

```bash
cd /home/yangyuxin/llm-deploy

bash scripts/run_remote_p1_ranked_tests.sh \
  --mode smoke \
  --run-id "p1-ranked-smoke-$(date +%Y%m%d-%H%M%S)" \
  --eval-batch-size 1 \
  --background
```

smoke 中 HLE 和 AA-LCR 允许被测模型自评,只验证 judge 链路,不能当正式分数。

### 2.3 P1 full

smoke 通过后才启动 full。

```bash
cd /home/yangyuxin/llm-deploy

bash scripts/run_remote_p1_ranked_tests.sh \
  --mode full \
  --run-id "p1-full-$(date +%Y%m%d-%H%M%S)" \
  --eval-batch-size 1 \
  --background
```

未配置独立 judge 的 HLE/AA-LCR 会保留 `BLOCKED_JUDGE`,不生成伪正式分数。要解除需配置独立 judge:

```bash
export EVALSCOPE_JUDGE_MODEL=<judge-served-name>
export EVALSCOPE_JUDGE_API_URL=http://127.0.0.1:<judge-port>/v1
bash scripts/run_remote_p1_ranked_tests.sh \
  --mode full \
  --run-id "p1-full-$(date +%Y%m%d-%H%M%S)" \
  --eval-batch-size 1 \
  --background
```

### 2.4 P1 续跑(同 run id)

```bash
cd /home/yangyuxin/llm-deploy

bash scripts/run_remote_p1_ranked_tests.sh \
  --mode full \
  --run-id "p1-full-20260716" \
  --eval-batch-size 1 \
  --background \
  --no-prepare-data
```

`--no-prepare-data` 复用已生成的 fixture。同 run id 再次启动会跳过 `SUCCESS`,重新检查 `FAILED` 与 `BLOCKED_*`。

### 2.5 P1 单项筛选

单项筛选统一用 ranked 入口的 `--benchmark`(benchmark 名称必须与 `p1_tasks.tsv` 一致):

```bash
cd /home/yangyuxin/llm-deploy

bash scripts/run_remote_p1_ranked_tests.sh \
  --mode smoke \
  --benchmark "BFCL-V4" \
  --run-id "p1-bfcl-smoke-$(date +%Y%m%d-%H%M%S)"
```

## 3. 监控进度

### 3.1 P0 监控(在 node1 上)

```bash
# 整体状态(ssh 进 node1 后)
RUN_ID="modelcard-p0-full-20260714-1934"
column -s $'\t' -t "logs/eval/modelcard/${RUN_ID}/suite_status.tsv"

# 单任务进度
jq . "logs/eval/modelcard/${RUN_ID}/mmlu_pro/qwen/progress.json"

# 单任务日志(实时)
tail -f "logs/eval/modelcard/${RUN_ID}/mmlu_pro/qwen/runner.log"

# 续跑后台日志
tail -f "logs/eval/modelcard/${RUN_ID}/resume-launcher.log"
```

### 3.2 P1 监控(本地 ssh 远程看)

```bash
# 实时日志
ssh root@10.16.11.24 'tail -f /mnt/nvme0/llm-deploy-eval-logs/modelcard/p1-ranked/p1-full-20260716/launcher.log'

# 状态汇总(去重,按 benchmark/model/mode 保留最后一次)
ssh root@10.16.11.24 'column -s $"\t" -t /mnt/nvme0/llm-deploy-eval-logs/modelcard/p1-ranked/p1-full-20260716/latest_status.tsv'

# 全部尝试记录(审计)
ssh root@10.16.11.24 'column -s $"\t" -t /mnt/nvme0/llm-deploy-eval-logs/modelcard/p1-ranked/p1-full-20260716/suite_status.tsv'

# 进程状态
ssh root@10.16.11.24 'pgrep -af run_p1_ranked_evaluations.sh'
```

### 3.3 查看 node1 服务是否收到请求

```bash
ssh root@10.16.11.24 'docker logs --since 1m qwen3.6-27b-fp8 2>&1' \
  | grep --line-buffered -E 'Avg prompt throughput|SpecDecoding metrics|Running|Waiting'
```

### 3.4 判断卡住的信号

| 现象 | 判断 | 处理 |
|---|---|---|
| `progress.json` 长时间不更新 | runner/PID 可能退出 | 检查 PID 和 `runner.log` 末尾,用同 run id `--resume` |
| node1 `Running=0` 且进度不更新 | 当前没有评测请求 | 检查 node1 上的评测进程和 progress 更新时间 |
| node1 `Running=0` 但进度在更新 | 评测在请求间隙 | 正常,继续等 |
| `progress=100%` | 不等于已验收 | 仍要检查 report、题量和异常 |
| `suite_status.tsv` 长时间无新行 | 整项未结束或卡住 | 看 `runner.log` 和 `issue_log.tsv` |

## 4. 验收正式结果

一个 benchmark 只有通过以下检查才能写入模型卡。

### 4.1 状态和文件检查

```bash
# P0
RUN_DIR="logs/eval/modelcard/${RUN_ID}"
column -s $'\t' -t "${RUN_DIR}/suite_status.tsv"
find "${RUN_DIR}" -path '*/reports/*' -type f -name '*.json' | sort
find "${RUN_DIR}" -path '*/predictions/*' -type f -name '*.jsonl' | sort
find "${RUN_DIR}" -path '*/reviews/*' -type f -name '*.jsonl' | sort

# P1(远程)
ssh root@10.16.11.24 'find /mnt/nvme0/llm-deploy-eval-logs/modelcard/p1-ranked/<run-id>/ -path "*/reports/*" -type f -name "*.json" | sort'
```

通过标准:
- Qwen 和 ThinkingCap 都有对应 full phase 的 `SUCCESS`。
- `predictions/`、`reviews/`、`reports/`、`runner.log`、`command.sh`、`frozen_config/` 完整。
- 实际执行命令与计划一致(查 `command.sh` 或 `commands.log`)。

### 4.2 题量核对

必须同时记录:官方/原始题量、计划题量、生成题量、实际计分题量、排除题量及原因。不能把生成题量当计分题量(如 IFBench 生成 300 条但计分 294 条)。

```bash
# 查看报告
jq . "${RUN_DIR}/mmlu_pro/qwen/reports/"*.json | head -50
```

### 4.3 配置一致性核对

两模型应具有相同的 dataset args、subset、split、few-shot、generation config、seed、timeout、输出预算、prompt/chat template、答案抽取和评分器。BFCL/TAU2 的 MODEL_SPECIFIC 差异除外。

```bash
# 对比两模型的冻结配置
diff "${RUN_DIR}/mmlu_pro/qwen/frozen_config/generation_config.json" \
     "${RUN_DIR}/mmlu_pro/thinkingcap/frozen_config/generation_config.json"
diff "${RUN_DIR}/mmlu_pro/qwen/frozen_config/dataset_args.json" \
     "${RUN_DIR}/mmlu_pro/thinkingcap/frozen_config/dataset_args.json"
```

### 4.4 异常核对

必须检查:HTTP 失败、超时、空答案、解析失败、达到 `max_tokens` 的样本、上下文超限、judge/sandbox/工具调用失败。达到 `max_tokens` 不能直接解释为能力差异。

## 5. 写入模型卡

验收通过后更新 `docs/reports/benchmark-comparison.md`:

1. 写主分数、实际计分题量和差值。
2. 在 `PROJECT_LOG.md` 记录运行、错误、修复和结论。
3. 原始结果保留在 node1 的 `logs/eval/` 或 `/mnt/nvme0/llm-deploy-eval-logs/`,不复制进报告目录。

单项结果至少记录:

```text
Benchmark:
Run ID / 完成时间:
模型与 served name:
Dataset revision / subset / split:
Few-shot / prompt / chat template:
temperature / top_p / top_k / max_tokens / seed:
计划 / 生成 / 计分 / 排除题量:
两个模型主分数和绝对差:
失败 / 超时 / 解析失败 / 截断:
是否复用 cache:
122B 官方参考分和可比性等级:
prediction / review / report / frozen_config 路径:
结论与限制:
```

## 6. 常见故障处理

| 现象 | 处理 |
|---|---|
| `/v1/models` 模型名不符 | 停止本轮,不要误发请求 |
| 两模型进度同时停止 | 检查 node1 上的 PID、`runner.log`,再用同 run id `--resume` |
| 冻结配置不一致 | 新建 run id,不强制复用 cache |
| IFBench 生成数与计分数不同 | adapter 有样本未进入指标,同时报告两个分母 |
| LongBench 上下文超限 | 重新预处理,不静默裁剪 |
| judge 任务被阻塞 | smoke 可自评;full 必须独立 judge |
| 代码任务被阻塞 | 修复隔离 sandbox 后显式 `--enable-sandbox` |
| TAU2 报空 `AssistantMessage` | 查看原始轨迹;ThinkingCap 检查服务端 `thinkingcap_agent.jinja` 是否加载并先重跑 smoke |
| BFCL 卡在 `memory snapshot prereq` | 检查 `all-MiniLM-L6-v2` patch 是否生效(重装 venv 会丢失,见 SOP-A 第 9 步) |
| HMMT 数学输出被截断 | 检查是否用了旧 tsv,P1 已用 `max_tokens=81920` |
| 请求被代理拦截(连接 127.0.0.1:7890) | `export NO_PROXY=...,127.0.0.1,localhost,10.16.11.24` |

## 7. 安全红线

- 绝对不重启 H100 宿主机,只允许按部署流程操作容器。
- 评测脚本只请求已有 OpenAI-compatible API,不启动、停止或重启模型服务,不修改模型权重。
- 不删除、移动或重写 `models/` 下的模型权重。
- 不保存 SSH key、token 或外部 judge 凭据。
- `BLOCKED` 表示环境或协议缺口,不表示模型能力失败;不能用近似数据或自评 judge 冒充正式结果。
- 修改 `thinkingcap_agent.jinja` 或重启 ThinkingCap 容器后,必须先重跑 TAU2 smoke,不能直接沿用修复前的状态。
