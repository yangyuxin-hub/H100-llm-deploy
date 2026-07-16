# 04 评测协议

本页讲"用什么参数":P0/P1 冻结参数表、公共设置、可比性等级、配置一致性要求。参数的机器可读来源是 `benchmarks/modelcard/core_tasks.tsv` 和 `p1_tasks.tsv`,本页是可读说明。

## 可比性等级

Qwen3.5-122B-A10B 模型卡给出了 Language 和 Vision Language 测试项目及参考分数,本仓库整理为 82 项:37 项 Language、45 项 Vision Language。

官方没有完整公开每一项的 prompt、数据 revision、seed、harness、judge 和输出预算。因此:

- 122B 官方成绩是参考线,不自动构成严格 A/B。
- 两个本地模型(都部署在 node1)必须使用同题、同配置和同评分协议。
- 每个结果都要标记可比性等级。

| 等级 | 含义 |
|---|---|
| `Strict` | 数据、prompt、harness、参数和评分协议均一致 |
| `Aligned` | 主要协议一致,少量官方细节无法确认 |
| `Approximate` | 同名 benchmark,但数据范围或协议有明确差异 |
| `Reference only` | 只引用官方分数,不做直接胜负判断 |
| `Not reproducible` | 数据或协议未公开,当前无法复现 |

## P0 冻结参数

来源:`benchmarks/modelcard/core_tasks.tsv`。

| Benchmark | few-shot / split | temperature | top_p / top_k | max_tokens | seed | 主要指标 |
|---|---|---:|---:|---:|---:|---|
| IFEval | 0-shot / train | 0 | 1.0 / -1 | 8,192 | 42 | Prompt/Instruction strict、loose |
| IFBench | 0-shot / train | 0 | 1.0 / -1 | 8,192 | 42 | Prompt/Instruction strict、loose |
| GPQA Diamond | 0-shot CoT / train | 0.6 | 0.95 / 20 | 16,384 | 42 | accuracy |
| MMLU-Pro | validation 5-shot CoT / test | 0.6 | 0.95 / 20 | 8,192 | 42 | 14 subsets mean accuracy |
| MMLU-Redux | 0-shot CoT / test | 0.6 | 0.95 / 20 | 8,192 | 42 | inclusion-aware accuracy |
| C-Eval | dev 5-shot / val | 0.6 | 0.95 / 20 | 8,192 | 42 | 52 subsets mean accuracy |
| LongBench v2 | 0-shot / train | 0.6 | 0.95 / 20 | 16,384 | 42 | accuracy(180 short + 215 medium + 108 long) |

## P1 冻结参数

来源:`benchmarks/modelcard/p1_tasks.tsv`。BFCL/TAU2 两模型参数不同的依据见[附录 A](appendix-a-bfcl-tau2-rationale.md)。

| 顺序 | Benchmark | 模型 | temperature | top_p / top_k | presence_penalty | max_tokens | thinking | seed |
|---:|---|---|---:|---|---:|---:|---|---:|
| 1 | BFCL-V4 | 两模型相同 | 0.7 | 0.8 / 20 | 1.5 | 81,920 | 关闭 | 42 |
| 2 | TAU2-Bench | Qwen | 1.0 | 0.95 / 20 | 0.5 | 81,920 | 开启 | 42 |
| 2 | TAU2-Bench | ThinkingCap | 0.7 | 0.8 / 20 | 0.0 | 81,920 | 关闭 | 42 |
| 3 | LiveCodeBench v6 | 两模型相同 | 0.6 | 0.95 / 20 | 0.0 | 81,920 | 开启 | 42 |
| 4 | HMMT Nov 25 | 两模型相同 | 1.0 | 0.95 / 20 | 0.0 | 81,920 | 开启 | 42 |
| 5 | HMMT Feb 25 | 两模型相同 | 1.0 | 0.95 / 20 | 0.0 | 81,920 | 开启 | 42 |
| 6 | PolyMATH | 两模型相同 | 1.0 | 0.95 / 20 | 0.0 | 81,920 | 开启 | 42 |
| 7 | SuperGPQA | 两模型相同 | 1.0 | 0.95 / 20 | 0.0 | 81,920 | 开启 | 42 |
| 8 | MMMLU | 两模型相同 | 1.0 | 0.95 / 20 | 0.0 | 81,920 | 开启 | 42 |
| 9 | HLE w/ CoT | 两模型相同 | 1.0 | 0.95 / 20 | 0.0 | 81,920 | 开启 | 42 |
| 10 | AA-LCR | 两模型相同 | 1.0 | 0.95 / 20 | 0.0 | 81,920 | 开启 | 42 |

### TAU2 用户模拟器固定参数

TAU2 的用户模拟器固定复用稳定 Qwen 端点,使用 `temperature=0`、`enable_thinking=false`,且不传 `max_tokens`。用户模拟器只生成简短对话,关闭 thinking 可避免只产生 reasoning 而用户正文为空;不传输出上限则避免在共享交互环境里再注入 4K 人为预算。不能为了某一被测模型单独调整 user simulator。

## 公共设置

- 确定性指令任务(IFEval/IFBench):`top_p=1.0`、`top_k=-1`。
- 采样型任务:`top_p=0.95`、`top_k=20`、`min_p=0.0`、`repetition_penalty=1.0`。
- `enable_thinking=true`(BFCL 和 ThinkingCap-TAU2 例外,见上表)。
- 单请求 timeout 900 秒,流式输出。
- 正式 P0 默认每个模型并发 4。
- 同一 benchmark 的两个模型并行,benchmark 之间串行。
- `max_tokens=81920` 是所有 P1 被测模型任务的完整输出预算,不要求模型填满;自动化另检查 `finish_reason`,不能把达到上限的未完成输出当作正常答案。
- 高温单 seed 只能复现一次 pass@1;正式稳定性结论应补多个 seed 并报告均值与波动,不能挑最好 seed。

## frozen_config 目录结构

每次运行时 runner 会把实际配置写入:

```text
<run-dir>/<benchmark>/<model>/frozen_config/
├── generation_config.json
├── dataset_args.json
├── expected_total.txt
├── protocol_note.txt
└── sample_counts.json        # LongBench v2
```

修改任何冻结项后应创建新的 run id。不能把不同参数的结果写入同一个 run 目录。

## 配置一致性要求

两模型做 A/B 时应具有相同的:

- dataset args、subset、split、few-shot;
- generation config、seed、timeout 和输出预算(BFCL/TAU2 的 MODEL_SPECIFIC 差异除外,依据见附录 A);
- prompt/chat template、答案抽取和评分器;
- judge、sandbox、用户模拟器和外部 worker 版本。

配置不一致时 runner 拒绝复用 cache;应创建新 run id,不强制混用。
