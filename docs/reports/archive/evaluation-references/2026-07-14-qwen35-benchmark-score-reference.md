# Qwen3.5 模型卡评测分数参考表

## 记录说明

- 记录日期：2026-07-14。
- 数据来源：用户在项目会话中提供的模型评测表；官方模型卡为 [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)。IFEval 93.4 已对照官方模型卡，其他条目尚未逐项复核。
- 覆盖范围：Language 37 项、Vision Language 45 项，共 82 个 benchmark 行。
- `--` 表示成绩尚不可用或不适用；带 `/` 的双分数按原表保留。
- 不对不同 benchmark 的分数求平均，也不把本表直接视为本地部署实测结果。不同任务的指标、样本版本、prompt、judge、工具环境和输出预算可能不同。

## Language

| 类别 | Benchmark | GPT-5-mini 2025-08-07 | GPT-OSS-120B | Qwen3-235B-A22B | Qwen3.5-122B-A10B | Qwen3.5-27B | Qwen3.5-35B-A3B |
|---|---|---:|---:|---:|---:|---:|---:|
| Knowledge | MMLU-Pro | 83.7 | 80.8 | 84.4 | 86.7 | 86.1 | 85.3 |
| Knowledge | MMLU-Redux | 93.7 | 91.0 | 93.8 | 94.0 | 93.2 | 93.3 |
| Knowledge | C-Eval | 82.2 | 76.2 | 92.1 | 91.9 | 90.5 | 90.2 |
| Knowledge | SuperGPQA | 58.6 | 54.6 | 64.9 | 67.1 | 65.6 | 63.4 |
| Instruction Following | IFEval | 93.9 | 88.9 | 87.8 | 93.4 | 95.0 | 91.9 |
| Instruction Following | IFBench | 75.4 | 69.0 | 51.7 | 76.1 | 76.5 | 70.2 |
| Instruction Following | MultiChallenge | 59.0 | 45.3 | 50.2 | 61.5 | 60.8 | 60.0 |
| Long Context | AA-LCR | 68.0 | 50.7 | 60.0 | 66.9 | 66.1 | 58.5 |
| Long Context | LongBench v2 | 56.8 | 48.2 | 54.8 | 60.2 | 60.6 | 59.0 |
| STEM & Reasoning | HLE w/ CoT | 19.4 | 14.9 | 18.2 | 25.3 | 24.3 | 22.4 |
| STEM & Reasoning | GPQA Diamond | 82.8 | 80.1 | 81.1 | 86.6 | 85.5 | 84.2 |
| STEM & Reasoning | HMMT Feb 25 | 89.2 | 90.0 | 85.1 | 91.4 | 92.0 | 89.0 |
| STEM & Reasoning | HMMT Nov 25 | 84.2 | 90.0 | 89.5 | 90.3 | 89.8 | 89.2 |
| Coding | SWE-bench Verified | 72.0 | 62.0 | -- | 72.0 | 72.4 | 69.2 |
| Coding | Terminal Bench 2 | 31.9 | 18.7 | -- | 49.4 | 41.6 | 40.5 |
| Coding | LiveCodeBench v6 | 80.5 | 82.7 | 75.1 | 78.9 | 80.7 | 74.6 |
| Coding | CodeForces | 2160 | 2157 | 2146 | 2100 | 1899 | 2028 |
| Coding | OJBench | 40.4 | 41.5 | 32.7 | 39.5 | 40.1 | 36.0 |
| Coding | FullStackBench en | 30.6 | 58.9 | 61.1 | 62.6 | 60.1 | 58.1 |
| Coding | FullStackBench zh | 35.2 | 60.4 | 63.1 | 58.7 | 57.4 | 55.0 |
| General Agent | BFCL-V4 | 55.5 | -- | 54.8 | 72.2 | 68.5 | 67.3 |
| General Agent | TAU2-Bench | 69.8 | -- | 58.5 | 79.5 | 79.0 | 81.2 |
| General Agent | VITA-Bench | 13.9 | -- | 31.6 | 33.6 | 41.9 | 31.9 |
| General Agent | DeepPlanning | 17.9 | -- | 17.1 | 24.1 | 22.6 | 22.8 |
| Search Agent | HLE w/ tool | 35.8 | 19.0 | -- | 47.5 | 48.5 | 47.4 |
| Search Agent | Browsecomp | 48.1 | 41.1 | -- | 63.8 | 61.0 | 61.0 |
| Search Agent | Browsecomp-zh | 49.5 | 42.9 | -- | 69.9 | 62.1 | 69.5 |
| Search Agent | WideSearch | 47.2 | 40.4 | -- | 60.5 | 61.1 | 57.1 |
| Search Agent | Seal-0 | 34.2 | 45.1 | -- | 44.1 | 47.2 | 41.4 |
| Multilingualism | MMMLU | 86.2 | 78.2 | 83.4 | 86.7 | 85.9 | 85.2 |
| Multilingualism | MMLU-ProX | 78.5 | 74.5 | 77.9 | 82.2 | 82.2 | 81.0 |
| Multilingualism | NOVA-63 | 51.9 | 51.1 | 55.4 | 58.6 | 58.1 | 57.1 |
| Multilingualism | INCLUDE | 81.8 | 74.0 | 81.0 | 82.8 | 81.6 | 79.7 |
| Multilingualism | Global PIQA | 88.5 | 84.1 | 85.7 | 88.4 | 87.5 | 86.6 |
| Multilingualism | PolyMATH | 67.3 | 54.0 | 60.1 | 68.9 | 71.2 | 64.4 |
| Multilingualism | WMT24++ | 80.7 | 74.4 | 75.8 | 78.3 | 77.6 | 76.3 |
| Multilingualism | MAXIFE | 85.3 | 83.7 | 83.2 | 87.9 | 88.0 | 86.6 |

### Language 表脚注

- **CodeForces**：使用原表作者自有 query set 评测。
- **TAU2-Bench**：除 airline domain 外遵循官方设置；airline domain 对所有模型应用 Claude Opus 4.5 system card 提出的修复。
- **Search Agent**：多数基于相关模型构建的搜索 Agent 使用简单的 256K context-folding 策略；累计 Tool Response 长度达到阈值后，裁剪较早的 Tool Response，使上下文保持在限制内。
- **WideSearch**：使用 256K context window，不做 context management。
- **MMLU-ProX**：报告 29 种语言的平均准确率。
- **WMT24++**：对 WMT24 进行难度标注和再平衡得到的更难子集；使用 XCOMET-XXL，报告 55 种语言的平均分。
- **MAXIFE**：报告 English 与 multilingual original prompts 的准确率，共 23 种设置。
- 空单元格 `--` 表示分数尚不可用或不适用。

## Vision Language

| 类别 | Benchmark | GPT-5-mini 2025-08-07 | Claude-Sonnet-4.5 | Qwen3-VL-235B-A22B | Qwen3.5-122B-A10B | Qwen3.5-27B | Qwen3.5-35B-A3B |
|---|---|---:|---:|---:|---:|---:|---:|
| STEM and Puzzle | MMMU | 79.0 | 79.6 | 80.6 | 83.9 | 82.3 | 81.4 |
| STEM and Puzzle | MMMU-Pro | 67.3 | 68.4 | 69.3 | 76.9 | 75.0 | 75.1 |
| STEM and Puzzle | MathVision | 71.9 | 71.1 | 74.6 | 86.2 | 86.0 | 83.9 |
| STEM and Puzzle | Mathvista(mini) | 79.1 | 79.8 | 85.8 | 87.4 | 87.8 | 86.2 |
| STEM and Puzzle | DynaMath | 81.4 | 78.8 | 82.8 | 85.9 | 87.7 | 85.0 |
| STEM and Puzzle | ZEROBench | 3 | 4 | 4 | 9 | 10 | 8 |
| STEM and Puzzle | ZEROBench_sub | 27.3 | 26.3 | 28.4 | 36.2 | 36.2 | 34.1 |
| STEM and Puzzle | VlmsAreBlind | 75.8 | 85.5 | 79.5 | 96.7 | 96.9 | 97.0 |
| STEM and Puzzle | BabyVision | 20.9 | 18.6 | 22.2 | 40.2 / 34.5 | 44.6 / 34.8 | 38.4 / 29.6 |
| General VQA | RealWorldQA | 79.0 | 70.3 | 81.3 | 85.1 | 83.7 | 84.1 |
| General VQA | MMStar | 74.1 | 73.8 | 78.7 | 82.9 | 81.0 | 81.9 |
| General VQA | MMBenchEN-DEV-v1.1 | 86.8 | 88.3 | 89.7 | 92.8 | 92.6 | 91.5 |
| General VQA | SimpleVQA | 56.8 | 57.6 | 61.3 | 61.7 | 56.0 | 58.3 |
| General VQA | HallusionBench | 63.2 | 59.9 | 66.7 | 67.6 | 70.0 | 67.9 |
| Text Recognition and Document Understanding | OmniDocBench1.5 | 77.0 | 85.8 | 84.5 | 89.8 | 88.9 | 89.3 |
| Text Recognition and Document Understanding | CharXiv(RQ) | 68.6 | 67.2 | 66.1 | 77.2 | 79.5 | 77.5 |
| Text Recognition and Document Understanding | MMLongBench-Doc | 50.3 | -- | 56.2 | 59.0 | 60.2 | 59.5 |
| Text Recognition and Document Understanding | CC-OCR | 70.8 | 68.1 | 81.5 | 81.8 | 81.0 | 80.7 |
| Text Recognition and Document Understanding | AI2D_TEST | 88.2 | 87.0 | 89.2 | 93.3 | 92.9 | 92.6 |
| Text Recognition and Document Understanding | OCRBench | 82.1 | 76.6 | 87.5 | 92.1 | 89.4 | 91.0 |
| Spatial Intelligence | ERQA | 54.0 | 45.0 | 52.5 | 62.0 | 60.5 | 64.8 |
| Spatial Intelligence | CountBench | 91.0 | 90.0 | 93.7 | 97.0 | 97.8 | 97.8 |
| Spatial Intelligence | RefCOCO(avg) | -- | -- | 91.1 | 91.3 | 90.9 | 89.2 |
| Spatial Intelligence | ODInW13 | -- | -- | 43.2 | 44.5 | 41.1 | 42.6 |
| Spatial Intelligence | EmbSpatialBench | 80.7 | 71.8 | 84.3 | 83.9 | 84.5 | 83.1 |
| Spatial Intelligence | RefSpatialBench | 9.0 | 2.2 | 69.9 | 69.3 | 67.7 | 63.5 |
| Spatial Intelligence | LingoQA | 62.4 | 12.8 | 66.8 | 80.8 | 82.0 | 79.2 |
| Spatial Intelligence | Hypersim | -- | -- | 11.0 | 12.7 | 13.0 | 13.1 |
| Spatial Intelligence | SUNRGBD | -- | -- | 34.9 | 36.2 | 35.4 | 33.4 |
| Spatial Intelligence | Nuscene | -- | -- | 13.9 | 15.4 | 15.2 | 14.6 |
| Video Understanding | VideoMME(w sub.) | 83.5 | 81.1 | 83.8 | 87.3 | 87.0 | 86.6 |
| Video Understanding | VideoMME(w/o sub.) | 78.9 | 75.3 | 79.0 | 83.9 | 82.8 | 82.5 |
| Video Understanding | VideoMMMU | 82.5 | 77.6 | 80.0 | 82.0 | 82.3 | 80.4 |
| Video Understanding | MLVU | 83.3 | 72.8 | 83.8 | 87.3 | 85.9 | 85.6 |
| Video Understanding | MVBench | -- | -- | 75.2 | 76.6 | 74.6 | 74.8 |
| Video Understanding | LVBench | -- | -- | 63.6 | 74.4 | 73.6 | 71.4 |
| Video Understanding | MMVU | 69.8 | 70.6 | 71.1 | 74.7 | 73.3 | 72.3 |
| Visual Agent | ScreenSpot Pro | -- | 36.2 | 62.0 | 70.4 | 70.3 | 68.6 |
| Visual Agent | OSWorld-Verified | -- | 61.4 | 38.1 | 58.0 | 56.2 | 54.5 |
| Visual Agent | AndroidWorld | -- | -- | 63.7 | 66.4 | 64.2 | 71.1 |
| Tool Calling | TIR-Bench | 24.6 | 27.6 | 29.8 | 53.2 / 42.5 | 59.8 / 42.3 | 55.5 / 38.0 |
| Tool Calling | V* | 71.7 | 58.6 | 85.9 | 93.2 / 90.1 | 93.7 / 89.0 | 92.7 / 89.5 |
| Medical VQA | SLAKE | 70.5 | 73.6 | 54.7 | 81.6 | 80.0 | 78.7 |
| Medical VQA | PMC-VQA | 36.3 | 55.9 | 41.2 | 63.3 | 62.4 | 62.0 |
| Medical VQA | MedXpertQA-MM | 34.4 | 54.0 | 47.6 | 67.3 | 62.4 | 61.4 |

## 使用提醒

1. 本表适合用于选择评测集、查找公开基线和设计本地对照，不适合直接把不同 benchmark 的数值相加或平均。
2. 本地复现时应单独记录 dataset revision、split、prompt/chat template、sampling、max tokens、judge、harness、工具权限和 context management。
3. 双分数条目（BabyVision、TIR-Bench、V*）的两个指标含义需回到原始模型卡确认后再用于横向比较。
4. `HMMT Feb 25`、`HMMT Nov 25` 等简称按用户提供的原表保留；正式运行配置中应写明完整年份和版本。
