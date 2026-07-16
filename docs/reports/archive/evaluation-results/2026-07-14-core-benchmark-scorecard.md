# 七项核心跑分对比与测试配置记录

## 跑分对比

记录时间：2026-07-15 10:18（本地 runner 已停止，待续跑）

Run ID：`modelcard-core-full-20260714`

| Benchmark | 本地题量 | Qwen3.5-122B-A10B 官方 | Qwen3.6-27B-FP8 本地 | ThinkingCap-Qwen3.6-27B-FP8 本地 | 状态 |
|---|---:|---:|---:|---:|---|
| IFEval | 541 | 93.4 | 84.84 | 87.80 | 完成 |
| IFBench | 300 生成 / 294 计分 | 76.1 | 50.00 | 54.08 | 完成 |
| GPQA Diamond | 198 | 86.6 | 83.33 | 83.84 | 完成 |
| MMLU-Pro | 12,032 | 86.7 | 84.72 | 85.14 | 完成 |
| MMLU-Redux | 5,700 | 94.0 | 31.56% 进度 | 53.77% 进度 | 已停止，待续跑 |
| C-Eval | 1,346 | 91.9 | 待跑 | 待跑 | 待跑 |
| LongBench v2 | 本地可运行 400 | 60.2 | 待跑 | 待跑 | 待跑 |

主表统一使用各任务的主指标：IFEval、IFBench 使用 `Prompt strict`，其余任务使用 accuracy。Qwen3.5-122B-A10B 一列来自[官方模型卡](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)，但官方没有完整披露每项评测的全部参数，因此只作为模型卡参考分，不视为严格同配置 A/B。

### 已完成任务的详细分数

| Benchmark | 指标 | Qwen3.6-27B-FP8 | ThinkingCap-Qwen3.6-27B-FP8 |
|---|---|---:|---:|
| IFEval | Prompt strict | 84.84 | 87.80 |
| IFEval | Instruction strict | 88.02 | 90.63 |
| IFEval | Prompt loose | 87.43 | 90.20 |
| IFEval | Instruction loose | 89.99 | 92.51 |
| IFBench | Prompt strict | 50.00 | 54.08 |
| IFBench | Instruction strict | 50.85 | 56.29 |
| IFBench | Prompt loose | 58.50 | 63.61 |
| IFBench | Instruction loose | 59.52 | 65.65 |
| GPQA Diamond | Accuracy | 83.33 | 83.84 |
| MMLU-Pro | Mean accuracy | 84.72 | 85.14 |

## 测试配置

### 公共环境

| 配置项 | 设置 |
|---|---|
| 评测工具 | EvalScope 1.6.1 Native |
| 推理服务 | vLLM 0.24.0，OpenAI-compatible API |
| Qwen 服务 | `qwen3.6-27b-fp8`，GPU 0,1，TP=2，端口 8000 |
| ThinkingCap 服务 | `thinkingcap-qwen3.6-27b-fp8`，GPU 2,3，TP=2，端口 8001 |
| 并发 | 每模型 4；同一任务两模型并行，总请求并发 8 |
| 调度方式 | 七个 benchmark 串行，每个 benchmark 内两模型并行 |
| Seed | 42 |
| Thinking | `enable_thinking=true` |
| Stream / timeout | `true` / 900 秒 |

### 各任务生成参数

| Benchmark | Few-shot / split | temperature | top_p | top_k | max_tokens |
|---|---|---:|---:|---:|---:|
| IFEval | 0-shot / train | 0 | 1.0 | -1 | 8,192 |
| IFBench | 0-shot / train | 0 | 1.0 | -1 | 8,192 |
| GPQA Diamond | 0-shot CoT / train | 0.6 | 0.95 | 20 | 16,384 |
| MMLU-Pro | 5-shot CoT from validation / test | 0.6 | 0.95 | 20 | 8,192 |
| MMLU-Redux | 0-shot CoT / test | 0.6 | 0.95 | 20 | 8,192 |
| C-Eval | 5-shot from dev / val | 0.6 | 0.95 | 20 | 8,192 |
| LongBench v2 | 0-shot / train | 0.6 | 0.95 | 20 | 16,384 |

LongBench v2 原始数据共 503 题。按真实 Qwen chat template 计数并为输出预留 16,384 tokens 后，本地保留 400 题，另外 103 题因超过 262,144-token 上下文上限而明确跳过，不做静默截断。

## IFEval 是怎么跑的

IFEval 使用 `opencompass/ifeval` 数据集的 541 题，0-shot，两个模型使用完全相同的 prompt、生成参数和输出预算。Runner 同时启动两个 EvalScope 进程，分别调用 8000 和 8001 端点；生成完成后自动执行规则检查并汇总四项分数。

本次七项正式运行命令：

```bash
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode full \
  --run-id modelcard-core-full-20260714 \
  --eval-batch-size 4
```

只复现 IFEval：

```bash
bash benchmarks/modelcard/run_core_evaluations.sh \
  --mode full \
  --benchmark ifeval \
  --run-id <new-ifeval-run-id> \
  --eval-batch-size 4
```

运行尚未全部完成的五项任务：

```bash
# 先做 smoke
bash benchmarks/modelcard/run_remaining_evaluations.sh \
  --mode smoke \
  --run-id modelcard-remaining-smoke

# smoke 通过后跑全量
bash benchmarks/modelcard/run_remaining_evaluations.sh \
  --mode full \
  --run-id modelcard-remaining-full \
  --eval-batch-size 4
```

只跑一个任务时增加 `--benchmark`，例如 `--benchmark mmlu_pro`。脚本始终使用 `--resume`，重新执行同一个 run id 会跳过已成功的模型/任务。

IFEval 的四项指标：

- `Prompt strict`：一条 prompt 中所有指令都严格满足才通过，是本记录的主分数。
- `Instruction strict`：按 prompt 内每条指令分别计分。
- `Prompt loose`、`Instruction loose`：允许清理首尾行或 Markdown 星号等轻微格式差异后重新检查。

## 配置与结果位置

| 内容 | 路径 |
|---|---|
| 七项任务清单与参数 | `benchmarks/modelcard/core_tasks.tsv` |
| Runner | `benchmarks/modelcard/run_core_evaluations.sh` |
| 待完成五项入口 | `benchmarks/modelcard/run_remaining_evaluations.sh` |
| 套件状态 | `logs/eval/modelcard/modelcard-core-full-20260714/suite_status.tsv` |
| 每项冻结配置 | `logs/eval/modelcard/modelcard-core-full-20260714/<benchmark>/<model>/frozen_config/` |
| IFEval 原始结果 | `logs/eval/modelcard/modelcard-core-full-20260714/ifeval/` |
| IFBench 原始结果 | `logs/eval/modelcard/modelcard-core-full-20260714/ifbench/` |
| GPQA Diamond 原始结果 | `logs/eval/modelcard/modelcard-core-full-20260714/gpqa_diamond/` |

本表在后续任务完成后继续更新；只有两个模型都完成且样本数核对无误，才把该行标记为“完成”。
