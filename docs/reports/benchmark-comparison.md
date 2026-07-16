# 核心模型跑分对比表

更新时间：2026-07-16

P0 正式 Run ID：`modelcard-core-full-20260714`（7/7 完成）

P1 正式 Run ID：`p1-full-20260716`（运行中，8/10 可跑，2 项 BLOCKED_JUDGE）

## P0 核心跑分（已完成）

当前进度：7/7 全部完成；两模型共 14 个任务状态均为 `SUCCESS`。

## 本地测试条件

| 类别 | 条件 |
|---|---|
| 测试时间 | 2026-07-14 19:34 至 2026-07-15 15:15（Asia/Shanghai） |
| 被测服务 | node1 上的两个 OpenAI-compatible 端点：Qwen3.6-27B-FP8 使用 GPU 0,1、TP=2、端口 8000；ThinkingCap-Qwen3.6-27B-FP8 使用 GPU 2,3、TP=2、端口 8001 |
| 硬件 | 4× NVIDIA H100 80GB HBM3；每个模型独占 2 张 GPU |
| 推理服务配置 | vLLM 0.24.0，CUDA 12.9 镜像；FP8 权重、FP8 KV cache、FLASH_ATTN、MTP 3、最大上下文 262,144 tokens、`max-num-seqs=16` |
| 评测端 | 本地评测器访问 node1 远端端点；EvalScope 1.6.1、Python 3.10.20 |
| 调度方式 | 每个模型请求并发 4；同一 benchmark 的两个模型并行，总请求并发 8；benchmark 之间串行 |
| 公共请求设置 | `enable_thinking=true`、`seed=42`、流式输出、单请求 timeout 900 秒；不允许静默截断 |
| 主要评分口径 | IFEval、IFBench 使用 `Prompt strict`；其余任务使用 accuracy 或 mean accuracy；分数单位为百分比 |

服务端参数来自本次评测所用项目配置；评测版本、端点、并发、seed 和生成参数由 Run 目录中的 `environment.json`、`commands.log` 与各任务 `frozen_config/` 记录。正式复现前仍应通过远端 Docker inspect/API 重新验收实际服务参数。

### 各任务冻结协议

| Benchmark | Prompt / split | temperature | top_p / top_k | max_tokens | 题量与特殊条件 |
|---|---|---:|---:|---:|---|
| IFEval | 0-shot / train | 0 | 1.0 / -1 | 8,192 | 541 题；Prompt/Instruction strict、loose |
| IFBench | 0-shot / train | 0 | 1.0 / -1 | 8,192 | 300 条生成，294 条计分 |
| GPQA Diamond | 0-shot CoT / train | 0.6 | 0.95 / 20 | 16,384 | 198 题 |
| MMLU-Pro | validation 5-shot CoT / test | 0.6 | 0.95 / 20 | 8,192 | 14 个 subset，共 12,032 题 |
| MMLU-Redux | 0-shot CoT / test | 0.6 | 0.95 / 20 | 8,192 | 57 个 subset，共 5,700 题；中断后按相同冻结配置复用缓存续跑 |
| C-Eval | dev 5-shot / val | 0.6 | 0.95 / 20 | 8,192 | 52 个 subset，共 1,346 题 |
| LongBench v2 | 0-shot / train | 0.6 | 0.95 / 20 | 16,384 | 原始 503 题；按真实 chat template 预计算长度，只评测上下文预算内的 400 题，跳过 103 题且不截断 |

> **可比性边界：** 下表中的严格 A/B 仅限两个本地 27B 模型。Qwen3.5-122B-A10B 分数来自官方模型卡；其 prompt、数据 revision、seed、harness、judge 和输出预算未完整公开，因此只能作为背景参考，不能与本地结果视为同测试条件。

## 主表

| Benchmark | 计划题量 / 实际计分 | Qwen3.5-122B-A10B 官方参考 | Qwen3.6-27B-FP8 本地 | ThinkingCap-Qwen3.6-27B-FP8 本地 | ThinkingCap - Qwen | 状态 |
|---|---|---:|---:|---:|---:|---|
| IFEval | 541 / 541 | 93.4 | 84.84 | 87.80 | +2.96pp | 完成 |
| IFBench | 300 生成 / 294 计分 | 76.1 | 50.00 | 54.08 | +4.08pp | 完成 |
| GPQA Diamond | 198 / 198 | 86.6 | 83.33 | 83.84 | +0.51pp | 完成 |
| MMLU-Pro | 12,032 / 12,032 | 86.7 | 84.72 | 85.14 | +0.42pp | 完成 |
| MMLU-Redux | 5,700 / 5,700 | 94.0 | 93.19 | 93.19 | 0.00pp | 完成；复用中断缓存续跑 |
| C-Eval | 1,346 / 1,346 | 91.9 | 91.09 | 90.79 | -0.30pp | 完成 |
| LongBench v2 | 原始 503；可运行 400 / 400 | 60.2 | 61.75 | 63.25 | +1.50pp | 完成；103 题超上下文预算，Approximate |

分数单位均为百分比。IFEval、IFBench 使用 `Prompt strict`，其余任务使用 accuracy 或 mean accuracy。中断任务的完成数量只是进度，不能当作跑分。

## 已完成任务的补充指标

| Benchmark | 指标 | Qwen3.6-27B-FP8 | ThinkingCap-Qwen3.6-27B-FP8 |
|---|---|---:|---:|
| IFEval | Instruction strict | 88.02 | 90.63 |
| IFEval | Prompt loose | 87.43 | 90.20 |
| IFEval | Instruction loose | 89.99 | 92.51 |
| IFBench | Instruction strict | 50.85 | 56.29 |
| IFBench | Prompt loose | 58.50 | 63.61 |
| IFBench | Instruction loose | 59.52 | 65.65 |

## 结果说明

- 七项均已完成：ThinkingCap 在 IFEval、IFBench、GPQA Diamond、MMLU-Pro 和 LongBench v2 上高 0.42pp 至 4.08pp，在 MMLU-Redux 上同为 93.19%，在 C-Eval 上低 0.30pp。差值较小的任务仍需结合统计不确定性判断，不能仅凭点估计断言稳定优势。
- Qwen3.5-122B-A10B 一列来自官方模型卡，只作参考。官方未完整披露每项 prompt、seed、harness 和输出预算，不能视为与本地结果严格同配置的 A/B。
- IFEval 全量差为 +2.96pp，但剔除任一模型达到 8,192-token 上限的 39 条提示后差距缩小到 +1.20pp，95% 区间跨 0；汇报时应同时说明截断影响。
- LongBench v2 的本地分数只覆盖模型上下文预算内的 400/503 题；103 题被明确跳过且未静默截断，因此协议等级为 Approximate，不能与官方 503 题分数直接横比。
- 题量、生成参数、运行方法和证据路径见[评测复现手册](README.md)。IFEval 的配对统计细节保存在[归档详报](archive/evaluation-results/2026-07-14-ifeval-comparison/report.html)。

## P1 ranked 全量（运行中）

- **Run ID**：`p1-full-20260716`
- **启动时间**：2026-07-16
- **调度方式**：按 `p1_tasks.tsv` 能力优先级串行；同一 benchmark 的两模型并行；每个端点并发 1；benchmark 之间串行
- **输出根目录**：`/mnt/nvme0/llm-deploy-eval-logs/modelcard/p1-ranked/p1-full-20260716/`
- **状态文件**：`suite_status.tsv`（append-only，审计用）、`latest_status.tsv`（按 benchmark/model/mode 去重，自动验收用）

### 任务清单与当前状态

| 顺序 | Benchmark | 题量/状态 | 备注 |
|---:|---|---|---|
| 1 | BFCL-V4 | 运行中（`memory_kv_prereq`） | 20 个无凭据子集，不含 `web_search_*`；需 MiniLM 本地模型 |
| 2 | TAU2-Bench | 待运行 | airline/retail/telecom 共 269 题；用户模拟器为 Qwen `temperature=0` |
| 3 | LiveCodeBench v6 | 待运行 | release_v6；Docker sandbox；pass@1 |
| 4 | HMMT Nov 25 | 待运行 | 30 题；固定 revision |
| 5 | HMMT Feb 25 | 待运行 | 30 题；本地 parquet |
| 6 | PolyMATH | 待运行 | 18 语言 × 4 levels；9000 题 |
| 7 | SuperGPQA | 待运行 | 72 子集；macro mean accuracy |
| 8 | MMMLU | 待运行 | 14 语言；unweighted macro mean |
| 9 | HLE w/ CoT | **BLOCKED_JUDGE** | 2158 题；需独立 judge，未配置 |
| 10 | AA-LCR | **BLOCKED_JUDGE** | 100 题；需独立 judge，未配置 |

### 启动前修复的离线依赖问题

node1 无公网，P1 full 首次启动时暴露两个离线依赖问题，已在本次启动前修复：

1. **BFCL memory 任务卡在 `memory snapshot prereq` 数十分钟**

   原因：`SentenceTransformer("all-MiniLM-L6-v2")` 在无网环境下尝试联网下载并指数退避重试，最终报 `RuntimeError: Cannot send a request, as the client has been closed.`

   修复：本地下载 `all-MiniLM-L6-v2`（90MB）上传到 `/mnt/nvme0/models/all-MiniLM-L6-v2/`，patch `bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_vector.py:26` 改用本地绝对路径。重启后 BFCL `memory_kv_prereq` 已顺利推进。

2. **6 个数据集在 full 模式下未传 `local_path` 导致 evalscope 联网拉取失败**

   原因：`run_reference_suite.sh` 的 `dataset_args_for()` 只在 `P1_SMOKE=1` 时传 `local_path`，full 模式下返回空 JSON，evalscope 尝试从 HF/ModelScope 联网下载。

   修复：HLE、SuperGPQA、HMMT Feb 25、MMMLU、PolyMATH、AA-LCR 六个分支改为统一通过 Python 脚本根据 `P1_SMOKE` 标志选择 `smoke_dir` 或 `full_dir`，只要目录存在就传 `local_path`。完整数据集已预下载到 `/root/llm-deploy/.eval-deps/data/p1-full-data/`。

详细修复步骤见 [03 数据准备](03-data-preparation.md)和 [07 故障处理](07-troubleshooting.md)。

### 监控命令

```bash
# 实时日志
ssh root@10.16.11.24 'tail -f /mnt/nvme0/llm-deploy-eval-logs/modelcard/p1-ranked/p1-full-20260716/launcher.log'

# 状态汇总（去重）
ssh root@10.16.11.24 'column -s $"\t" -t /mnt/nvme0/llm-deploy-eval-logs/modelcard/p1-ranked/p1-full-20260716/latest_status.tsv'

# 全部尝试记录（审计）
ssh root@10.16.11.24 'column -s $"\t" -t /mnt/nvme0/llm-deploy-eval-logs/modelcard/p1-ranked/p1-full-20260716/suite_status.tsv'

# 进程状态
ssh root@10.16.11.24 'pgrep -af run_p1_ranked_evaluations.sh'
```
