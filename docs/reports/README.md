# 评测复现手册

本目录是一份可以直接照着执行的模型卡评测手册,目标是参考 [Qwen3.5-122B-A10B 模型卡](https://huggingface.co/Qwen/Qwen3.5-122B-A10B),对 `Qwen3.6-27B-FP8` 和 `ThinkingCap-Qwen3.6-27B-FP8` 进行同题、同配置、可恢复的能力评测。

只讲"如何复现"。实时分数见[跑分对比表](benchmark-comparison.md)。

## 两份可执行 SOP(直接照着跑)

| SOP | 适用场景 | 文档 |
|---|---|---|
| SOP-A 评测环境从零搭建 | 从干净的本地工作机 + 已部署模型服务的 node1 出发,搭出可复现评测能力 | [sop-environment-setup.md](sop-environment-setup.md) |
| SOP-B 环境就绪后的自动化测评执行 | venv/数据/协议已就绪,只想触发评测、监控进度、续跑和验收 | [sop-evaluation-execution.md](sop-evaluation-execution.md) |

两份 SOP 内的命令均为可直接执行的精确命令,不是样例。SOP-A 完成后可直接进入 SOP-B。

## 参考文档清单

按功能拆分的参考文档,各司其职(背景与原理,执行时按 SOP-A/SOP-B):

| 文档 | 回答什么问题 |
|---|---|
| [01 测试集总表](01-benchmark-inventory.md) | 有哪些测试集?各自多少题、测什么能力、什么前置条件、当前状态 |
| [02 环境配置](02-environment-setup.md) | 环境怎么配?模型服务怎么检查、端点访问、EvalScope、按需依赖 |
| [03 数据准备](03-data-preparation.md) | 要做什么数据准备?P0/P1 数据冻结、LongBench 子集、SHA256 |
| [04 评测协议](04-evaluation-protocol.md) | 用什么参数?P0/P1 冻结参数表、可比性等级、配置一致性 |
| [05 执行流程](05-execution-guide.md) | 怎么跑起来?脚本速查表、dry-run/preflight/smoke/full/续跑 |
| [06 观测与验收](06-monitoring-and-verification.md) | 怎么观测?日志路径、判断卡住、性能观测、验收清单、写卡模板 |
| [07 常见问题与安全边界](07-troubleshooting.md) | 出问题怎么办?故障排查表、安全红线 |
| [附录 A: BFCL/TAU2 设计依据](appendix-a-bfcl-tau2-rationale.md) | 为什么 BFCL/TAU2 用那套参数?生成参数的原理与取舍 |
| [附录 B: 微调数据测评流程](appendix-b-finetuned-model-eval.md) | 微调后模型怎么回归评测?训练→部署→回归对比的流程骨架 |
| [跑分对比表](benchmark-comparison.md) | 当前分数是多少?双模型主表、补充指标、可比性说明 |

## 按"我想做什么"快速跳转

- **第一次搭环境** → SOP-A(直接执行)或 02 → 03 → 04(参考)
- **环境已就绪,要跑评测** → SOP-B(直接执行)
- **想知道测什么** → 01
- **跑的时候看进度/排错** → SOP-B 第 3、6 节 或 06 → 07
- **跑完写卡** → SOP-B 第 5 节 或 06 的"验收与写卡"小节
- **理解 BFCL/TAU2 参数为什么这样设** → 附录 A
- **要做微调数据测评** → 附录 B

## 最小可执行顺序

> **node1 无公网。** EvalScope venv、ModelScope/HF 数据缓存、LongBench 子集、项目代码都必须先在本地准备好再 rsync 到 node1;直接在 node1 上 `uv pip install` 或 `MsDataset.load` 会卡住。同步细节见 [02 的"node1 离线时的环境与数据同步"](02-environment-setup.md)。

### A. 本地准备(只做一次)

在本地完成以下准备,每步细节见 02/03:

1. 创建 `.eval-deps/evalscope-1.6.1` venv 并装 P0 依赖(`evalscope[perf,app,ifeval,ifbench]==1.6.1`、`setuptools<81`)—— 见 02。
2. 下载 LongBench v2 原始 503 题并跑 `prepare_longbench_inlimit.py` 生成本地长度安全子集 —— 见 03。
3. 让 P0 其余数据集(IFEval/IFBench/GPQA/MMLU-Pro/MMLU-Redux/C-Eval)下载到本地 `~/.cache/modelscope` 和 `~/.cache/huggingface`(本地有端点就跑一次 smoke;没有就用 `run_reference_suite.sh --preflight-only --skip-endpoint` 触发 adapter 下载)—— 见 02。
4. 按 02 的"node1 离线时的环境与数据同步"把 venv、两个缓存、LongBench 子集、`config/`/`benchmarks/`/`scripts/` 同步到 node1,并修复 venv 内的绝对路径。

### B. node1 上执行

```bash
# 1. ssh 进 node1,进入项目目录,export 端点和缓存路径 —— 见 02
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
# 端点是 node1 本机访问,不需要 SSH 隧道

# 2. dry-run —— 见 05
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode smoke --run-id modelcard-p0-dry-run --dry-run

# 3. smoke —— 见 05
RUN_ID="modelcard-p0-smoke-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode smoke --run-id "${RUN_ID}" --eval-batch-size 2

# 4. full —— 见 05(smoke 未通过不进入此步)
RUN_ID="modelcard-p0-full-$(date +%Y%m%d-%H%M%S)"
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode full --run-id "${RUN_ID}" \
  --eval-batch-size 4 --benchmark-parallel 1
```

## 原始数据与归档

- 原始预测与评分结果在 `logs/eval/`,runner 和任务清单在 `benchmarks/modelcard/`,不复制进报告目录。
- `archive/` 保存早期 pilot、单项详报、模型卡调研、覆盖规划和性能分析,仅在需要追溯细节时查阅,不作为最终汇报入口。原 923 行单文档 `benchmark-test-process.md` 已拆分到上述 01–07,旧版归档至 `archive/`。
