# 05 执行流程

本页讲"怎么跑起来":脚本速查表(先看用哪个脚本)、dry-run、preflight、smoke、full、续跑。参数见 [04](04-evaluation-protocol.md),跑的时候看进度见 [06](06-monitoring-and-verification.md)。

## 脚本速查表

先确定用哪个入口,再去看对应小节的详细命令。**所有评测都在 node1 上跑**(本地内存不够会死机)。

| 我想跑 | 执行位置 | 入口脚本 | 典型命令 |
|---|---|---|---|
| P0 dry-run | node1(ssh 进去手动跑) | `benchmarks/modelcard/run_core_evaluations.sh` | `--mode smoke --dry-run` |
| P0 smoke / full / 续跑 | node1(ssh 进去手动跑) | `benchmarks/modelcard/run_core_evaluations.sh` | `--mode smoke\|full [--resume]` |
| P0 续跑剩余 3 项(MMLU-Redux/C-Eval/LongBench) | node1(ssh 进去手动跑) | `benchmarks/modelcard/run_remaining_evaluations.sh` | `--background` |
| P1 全十项 smoke / full / 续跑 | node1(本地一键触发) | `scripts/run_remote_p1_ranked_tests.sh` | `--mode smoke\|full [--background]` |
| P1 单项筛选 | node1(本地一键触发) | `scripts/run_remote_p1_ranked_tests.sh` | `--benchmark BFCL-V4\|TAU2-Bench\|...` |
| P1 本地 preflight / smoke | node1(ssh 进去手动跑) | `benchmarks/modelcard/run_p1_evaluations.sh` | `--mode preflight\|smoke` |
| 82 项离线 dry-run | node1(ssh 进去手动跑) | `benchmarks/evalscope/run_reference_suite.sh` | `--dry-run` |

**执行位置说明**:

- **node1(ssh 进去手动跑)**:P0 和 82 项 dry-run 用这种方式。先 `ssh root@10.16.11.24` 登录,`cd /root/llm-deploy`,然后在 node1 上直接 `bash` 调脚本。端点是 `127.0.0.1:8000`(本机访问),不需要 SSH 隧道。
- **node1(本地一键触发)**:P1 用这种方式。你在本地 `bash scripts/run_remote_p1_*.sh`,脚本内部用 `ssh root@10.16.11.24` 把评测命令发到 node1 上执行。评测进程、模型服务、结果文件都在 node1 上。端点同样是 `127.0.0.1:8000`,不需要 SSH 隧道。

两种方式的评测进程都在 node1 上,本地电脑只负责触发和看结果,不跑评测进程,不会占本地内存。P1 远程入口会自动同步 runner、fixture 和配置到 node1,并设置远端环境变量(PATH、API URL、缓存路径、NO_PROXY 等),你只需要 `ssh root@10.16.11.24` 能登录。

## 第一步:dry-run

只生成计划,不访问端点、不下载数据、不请求模型。以下命令在 node1 上执行(先 `ssh root@10.16.11.24` 进去,`cd /root/llm-deploy`)。

### 82 项离线 dry-run

```bash
bash benchmarks/evalscope/run_reference_suite.sh \
  --model qwen \
  --modality all \
  --priority all \
  --dry-run \
  --run-id modelcard-reference-dry-run
```

输出位于:

```text
logs/eval/reference-82/modelcard-reference-dry-run/qwen/
├── status.tsv
└── commands.log
```

状态含义:

| 状态 | 含义 |
|---|---|
| `DRY_RUN` | runner 和命令可以生成 |
| `BLOCKED_REVIEW` | adapter 存在,但协议仍需核对 |
| `BLOCKED_ENV` | 缺依赖、judge、sandbox 或数据 |
| `BLOCKED_EXTERNAL` | 需要外部 harness/worker |
| `BLOCKED_PROTOCOL` | adapter 或模型卡协议未准备好 |

### P0 dry-run

P0 runner 的 dry-run 会冻结命令和配置,但不检查端点、不请求模型:

```bash
RUN_ID="modelcard-p0-dry-run-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode smoke \
  --run-id "${RUN_ID}" \
  --dry-run
```

检查:

```bash
cat "logs/eval/modelcard/${RUN_ID}/commands.log"
column -s $'\t' -t "logs/eval/modelcard/${RUN_ID}/suite_status.tsv" | less -S
```

## 第二步:preflight(P1)

```bash
bash benchmarks/modelcard/run_p1_evaluations.sh \
  --mode preflight \
  --run-id "modelcard-p1-preflight-$(date +%Y%m%d-%H%M%S)"
```

Preflight 只检查端点、adapter、环境、数据、judge 和 sandbox,不发送正式题目。先处理 `BLOCKED_DATA` 和 `BLOCKED_ENV`;`BLOCKED_EXTERNAL` 不应通过安装普通 Python 包强行解除。

## 第三步:smoke

Smoke 的目标是验证完整链路,不评价模型能力。必须检查:

```text
数据加载 → prompt 构造 → API 请求 → 流式响应
 答案抽取 → 逐题 review → 聚合报告 → 状态落盘
```

P0 命令在 node1 上执行(先 `ssh root@10.16.11.24` 进去,`cd /root/llm-deploy`)。P1 命令在本地触发。

### P0 双模型自动 smoke

```bash
RUN_ID="modelcard-p0-smoke-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode smoke \
  --run-id "${RUN_ID}" \
  --eval-batch-size 2
```

`--limit 1` 由 runner 自动添加。注意它表示每个 subset 一题:MMLU-Pro、MMLU-Redux 和 C-Eval 会产生多条请求,不是整个 benchmark 只有一题。

也可以先单跑一项:

```bash
RUN_ID="modelcard-mmlu-pro-smoke-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode smoke \
  --benchmark mmlu_pro \
  --run-id "${RUN_ID}" \
  --eval-batch-size 1
```

### P1 双模型自动 smoke

```bash
RUN_ID="p1-ranked-smoke-$(date +%Y%m%d-%H%M%S)"
bash scripts/run_remote_p1_ranked_tests.sh \
  --mode smoke \
  --run-id "${RUN_ID}" \
  --eval-batch-size 1 \
  --background
```

该入口自动准备 fixture、同步 runner,并在 node1 按 `p1_tasks.tsv` 的顺序跑十项:BFCL、TAU2、LiveCodeBench、HMMT Nov、HMMT Feb、PolyMATH、SuperGPQA、MMMLU、HLE、AA-LCR。同一 benchmark 的两个模型并行,benchmark 之间串行。HLE w/ CoT 和 AA-LCR 在 smoke 中允许被测模型自评,只验证 judge 链路;正式运行仍要求独立 judge。

如只需单项,可传 benchmark 显示名:

```bash
bash scripts/run_remote_p1_ranked_tests.sh \
  --mode smoke \
  --benchmark "BFCL-V4" \
  --run-id "p1-bfcl-smoke-$(date +%Y%m%d-%H%M%S)"
```

该入口只同步配置、runner 和 smoke fixture,不同步 `models/`、历史日志或整个 Python 环境。node1 首次使用前仍需准备 EvalScope 1.6.1、BFCL scorer、TAU2 v0.2.0 数据和 LiveCodeBench 的 `python:3.11-slim` Docker 镜像,不得把"sandbox pool 为空"当成成功。

HMMT Nov/Feb 在旧 smoke 中已证明 16,384 会截断,新参数把竞赛数学预算提高到 81,920。runner 会把已知 sandbox 错误和"所有 prediction 均以 `max_tokens`/`length` 结束"强制转为失败,即使 EvalScope 自身退出码为 0。

### Smoke 通过标准

每个"模型 + benchmark"必须满足:

- runner 退出码为 0,状态为 `SUCCESS`;
- `predictions/`、`reviews/`、`reports/` 和 `progress.json` 存在;
- 没有未解释的 HTTP 错误、超时、空输出或解析失败;
- 输出没有全部停在 `max_tokens`;
- 多模态、代码、judge、Agent 任务通过自己的 gate;
- 分数只标记为 smoke,不写入正式模型卡。

检查命令:

```bash
RUN_DIR="logs/eval/modelcard/${RUN_ID}"
column -s $'\t' -t "${RUN_DIR}/suite_status.tsv" | less -S
find "${RUN_DIR}" -type f \
  \( -name 'progress.json' -o -name '*.jsonl' -o -path '*/reports/*.json' \) \
  | sort
```

## 第四步:full

只有 smoke 验收通过后才启动。P0 命令在 node1 上执行(先 `ssh root@10.16.11.24` 进去,`cd /root/llm-deploy`)。P1 命令在本地触发。

### P0 全量

```bash
RUN_ID="modelcard-p0-full-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode full \
  --run-id "${RUN_ID}" \
  --eval-batch-size 4 \
  --benchmark-parallel 1
```

调度方式:

- 同一 benchmark 的 Qwen 与 ThinkingCap 同时运行。
- 每个端点并发 4,总请求并发 8。
- benchmark 之间串行,避免多个任务争抢端点和混淆日志。

只跑单项:

```bash
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode full \
  --benchmark gpqa_diamond \
  --run-id "${RUN_ID}" \
  --eval-batch-size 4
```

### P1 全量

```bash
RUN_ID="p1-ranked-full-$(date +%Y%m%d-%H%M%S)"
bash scripts/run_remote_p1_ranked_tests.sh \
  --mode full \
  --run-id "${RUN_ID}" \
  --eval-batch-size 1 \
  --background
```

未配置独立 judge、sandbox、TAU2 或外部环境的任务会保留 `BLOCKED_*`,不会生成伪正式分数。

P1 的总状态位于 node1 的 `/mnt/nvme0/llm-deploy-eval-logs/modelcard/p1-ranked/<run-id>/suite_status.tsv`,每个任务的原始输出位于该 run id 下对应的任务和模型目录。

只运行单项(如 BFCL-V4)用 ranked 入口的 `--benchmark` 筛选:

```bash
bash scripts/run_remote_p1_ranked_tests.sh \
  --mode full --benchmark "BFCL-V4" \
  --run-id "p1-bfcl-full-$(date +%Y%m%d-%H%M%S)" --background
```

BFCL full 是 20 个无凭据子集,不包含 `web_search_base` 和 `web_search_no_snippet`,必须标记为 partial;TAU2 full 是三个领域共 269 题。若模型把工具调用写进普通 `content` 而没有 OpenAI `tool_calls`,应记录为协议失败,不在评测后处理阶段静默改写。ThinkingCap 已知的长 system prompt 顺序冲突应在服务层使用受审计的 `config/chat_templates/thinkingcap_agent.jinja` 修复,并通过 `scripts/start_thinkingcap_docker.sh` 加载;修改模板或重启容器后必须先重跑 TAU2 smoke,不能直接沿用修复前的失败/成功状态。

BFCL/TAU2 的生成参数设计依据(为什么这样设)见[附录 A](appendix-a-bfcl-tau2-rationale.md)。

## 第五步:中断和续跑

不要删除原 run 目录。P0 续跑在 node1 上执行(先 `ssh root@10.16.11.24` 进去,`cd /root/llm-deploy`);P1 续跑在本地触发。使用相同 run id 和完全相同的冻结配置:

```bash
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode full \
  --run-id "${RUN_ID}" \
  --eval-batch-size 4 \
  --resume
```

续跑规则:

- 已有 `SUCCESS` 的"模型 + benchmark"直接跳过。
- 未完成任务若存在 prediction/review,会先比较 `frozen_config/`。
- 配置一致时传给 EvalScope `--use-cache`,只处理缺失样本。
- 配置不一致时拒绝混用;应创建新 run id。

### P0 续跑剩余任务

```bash
bash benchmarks/modelcard/run_remaining_evaluations.sh \
  --run-id modelcard-core-full-20260714 \
  --background
```

默认顺序为 MMLU-Redux、C-Eval、LongBench v2。后台日志:

```bash
tail -f logs/eval/modelcard/modelcard-core-full-20260714/resume-launcher.log
```

不要重复启动同一个 run id。`run_remaining_evaluations.sh` 会做重复启动检查,但启动前仍应查看 PID、日志更新时间和端点请求数。

### P1 续跑

同一 run id 再次启动会跳过 `SUCCESS`,重新检查 `FAILED` 与 `BLOCKED_*`;dry-run 也不会导致真实运行被跳过。配置好 judge 或修复环境后无需换 run id。

```bash
bash scripts/run_remote_p1_ranked_tests.sh \
  --mode full \
  --run-id "${RUN_ID}" \
  --background \
  --no-prepare-data
```

`latest_status.tsv` 会按 benchmark/model/mode 保留最后一次状态并恢复优先级顺序,适合自动验收;`suite_status.tsv` 保留所有尝试,适合审计。
