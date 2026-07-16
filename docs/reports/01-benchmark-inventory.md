# 01 测试集总表

本页一页看全"测什么":有哪些 benchmark、各自测什么能力、多少题、什么指标、什么前置条件、当前状态。后续所有文档引用本表的层级和命名。

## 选取依据

本项目的 benchmark 清单参照 **Qwen3.6-122B 官方模型卡**公布的评测项目选取。目标是在 27B-FP8 和 ThinkingCap-27B-FP8 上复现 122B 模型卡中的核心能力维度,做同协议 A/B 对比。

- **P0** 对应 122B 模型卡的基础能力项(指令遵循、知识、中文、推理、长上下文),优先跑出第一版可信模型卡。
- **P1** 补齐 122B 模型卡中的高价值能力(数学、代码、工具调用、Agent、多语言、高难推理)。
- **P2–P3** 覆盖 122B 模型卡的视觉、软件工程、终端、搜索、视频、GUI、医疗等扩展能力;当前大多缺外部 harness 或视觉环境,作为后续扩展储备。

122B 官方分数仅作为参考,不做同协议复现目标。可比性等级见 [06](06-monitoring-and-verification.md)。

## P0–P3 分层

P0–P3 表示模型卡业务优先级,不等于测试难度,也不等于当前环境已经可运行。环境是否就绪另用状态表示(`CONFIG_FROZEN`、`BLOCKED_JUDGE`、`BLOCKED_ENV` 等)。

| 层级 | 数量 | 目的 | 主要能力 | 启动条件 |
|---|---:|---|---|---|
| P0 | 7 | 形成第一版可信模型卡 | 指令、知识、中文、推理、长上下文 | 基础 EvalScope 和两个端点可用 |
| P1 | 10 | 补齐高价值能力 | 数学、代码、工具、Agent、多语言 | sandbox、TAU2、judge 等按项就绪 |
| P2 | 47 | 验证视觉和中等复杂环境 | 视觉、软件工程、终端、多语言、空间 | 视觉、sandbox、judge gate 通过 |
| P3 | 18 | 扩展搜索、视频、GUI、医疗 | 搜索 Agent、视频理解、GUI 操作、医疗 VQA | 单独建设外部 harness 和资源 |

总计 82 项,与 `benchmarks/evalscope/reference_benchmarks.tsv` 一致。

## 全量总表

| 层级 | Benchmark | 能力 | 题量 | 主要指标 | 前置条件 | 状态 |
|---|---|---|---:|---|---|---|
| P0 | IFEval | 指令遵循 | 541 | Prompt/Instruction strict、loose | 基础 EvalScope | 完成 |
| P0 | IFBench | 复杂指令遵循 | 300 生成/294 计分 | Prompt/Instruction strict、loose | ifbench 依赖 | 完成 |
| P0 | GPQA Diamond | 高难科学推理 | 198 | accuracy | 基础 EvalScope | 完成 |
| P0 | MMLU-Pro | 综合知识与推理 | 12,032 | 14 subsets mean accuracy | 基础 EvalScope | 完成 |
| P0 | MMLU-Redux | 综合知识复核 | 5,700 | 57 subsets inclusion-aware accuracy | 基础 EvalScope | 完成 |
| P0 | C-Eval | 中文知识与推理 | 1,346 | 52 subsets mean accuracy | 基础 EvalScope | 完成 |
| P0 | LongBench v2 | 长上下文 | 503(可运行 400) | accuracy | 本地长度安全子集 | 完成(Approximate) |
| P1 | BFCL-V4 | Function Calling | verify_preflight | pass@1 / accuracy | BFCL scorer | `BENCHMARK_SPECIFIC_PARTIAL` |
| P1 | TAU2-Bench | 多轮 Agent | 269 | reward / mean accuracy | TAU2 数据 + 用户模拟器 | `MODEL_SPECIFIC` |
| P1 | LiveCodeBench v6 | 代码生成 | verify_preflight | pass@1 | Docker sandbox | `CONFIG_FROZEN` |
| P1 | HMMT Nov 25 | 竞赛数学 | 30 | numeric exact match | 固定 revision 数据 | `CONFIG_FROZEN` |
| P1 | HMMT Feb 25 | 竞赛数学 | 30 | numeric exact match | 固定 30 题 | `CONFIG_FROZEN` |
| P1 | PolyMATH | 多语言数学 | 9,000 | accuracy、DW-ACC | 基础 EvalScope | `CONFIG_FROZEN` |
| P1 | SuperGPQA | 专业知识推理 | verify_preflight | 72 subsets macro mean accuracy | 基础 EvalScope | `CONFIG_FROZEN` |
| P1 | MMMLU | 多语言知识推理 | verify_preflight | 14 语言宏平均 | 基础 EvalScope | `CONFIG_FROZEN` |
| P1 | HLE w/ CoT | 高难通用推理 | 2,158 | judge score | 独立 judge | `BLOCKED_JUDGE` |
| P1 | AA-LCR | 长上下文推理 | 100 | judge score | 独立 judge | `BLOCKED_JUDGE` |
| P2 | 47 项 | 视觉/代码/多语言/空间 | 见 P2 详表 | 见 P2 详表 | 视觉环境、sandbox、外部 harness | 多为 `BLOCKED_EXTERNAL` |
| P3 | 18 项 | 搜索/视频/GUI/医疗 | 见 P3 详表 | 见 P3 详表 | 外部 harness、视频资源、VM | 全部 `BLOCKED_EXTERNAL` |

P2–P3 的逐项清单见下方详表,也可查 `benchmarks/evalscope/reference_benchmarks.tsv`。

### 状态字段含义

| 状态 | 含义 |
|---|---|
| `CONFIG_FROZEN` | 配置已冻结,环境就绪即可正式运行 |
| `MODEL_SPECIFIC` | 两模型使用各自冻结的不同参数(均有依据,见附录 A) |
| `BENCHMARK_SPECIFIC_PARTIAL` | 只覆盖部分 subset(如 BFCL 排除网页搜索类) |
| `BLOCKED_JUDGE` | 缺独立 judge;smoke 可自评,正式分数不能进模型卡 |
| `BLOCKED_ENV` / `BLOCKED_DATA` / `BLOCKED_PROTOCOL` / `BLOCKED_EXTERNAL` | 缺依赖、数据、协议或外部 worker |
| `REVIEW` | adapter 存在但协议未冻结,需核对 prompt、subset 或评分方式 |
| `verify_preflight` | 题量在 preflight 阶段由 runner 确认 |

## P0 七项详表

| Benchmark | 能力 | 计划题量 | 主要指标 |
|---|---|---:|---|
| IFEval | 指令遵循 | 541 | Prompt/Instruction strict、loose |
| IFBench | 复杂指令遵循 | 300 | Prompt/Instruction strict、loose |
| GPQA Diamond | 高难科学推理 | 198 | accuracy |
| MMLU-Pro | 综合知识与推理 | 12,032 | 14 subsets mean accuracy |
| MMLU-Redux | 综合知识复核 | 5,700 | inclusion-aware accuracy(57 subsets) |
| C-Eval | 中文知识与推理 | 1,346 | 52 subsets mean accuracy |
| LongBench v2 | 长上下文 | 原始 503(可运行 400) | accuracy |

P0 不覆盖代码执行、Agent、多语言和视觉;这些能力从 P1 开始补充。

## P1 十项详表

该表同时是默认执行顺序:先测最能区分 Agent/工具调用能力的 BFCL 与 TAU2,再测代码、高难数学、知识/多语言;缺独立 judge 的两项放在最后,避免阻塞已就绪任务。

| 顺序 | Benchmark | 能力 | 题量 | 环境要求 | 状态 |
|---:|---|---|---:|---|---|
| 1 | BFCL-V4 | Function Calling | verify_preflight | BFCL scorer | `BENCHMARK_SPECIFIC_PARTIAL`:无凭据 20 类,排除两类网页搜索 |
| 2 | TAU2-Bench | 多轮 Agent | 269 | TAU2 v0.2.0 数据 + 用户模拟器 | `MODEL_SPECIFIC`:两模型参数不同 |
| 3 | LiveCodeBench v6 | 代码生成 | verify_preflight | Docker sandbox | `CONFIG_FROZEN`:release_v6,禁用 release_latest |
| 4 | HMMT Nov 25 | 竞赛数学 | 30 | 固定 revision + SHA256 | `CONFIG_FROZEN` |
| 5 | HMMT Feb 25 | 竞赛数学 | 30 | 固定 30 题 | `CONFIG_FROZEN` |
| 6 | PolyMATH | 多语言数学 | 9,000 | 基础 EvalScope | `CONFIG_FROZEN`:18 语言 × 4 难度 |
| 7 | SuperGPQA | 专业知识推理 | verify_preflight | 基础 EvalScope | `CONFIG_FROZEN`:72 subsets |
| 8 | MMMLU | 多语言知识推理 | verify_preflight | 基础 EvalScope | `CONFIG_FROZEN`:14 语言宏平均 |
| 9 | HLE w/ CoT | 高难通用推理 | 2,158 | 独立 judge | `BLOCKED_JUDGE`:仅文本行,smoke 自评只验证链路 |
| 10 | AA-LCR | 长上下文推理 | 100 | 独立 judge | `BLOCKED_JUDGE`:不允许静默截断 |

各任务的具体生成参数见 [04 评测协议](04-evaluation-protocol.md);BFCL/TAU2 的参数设计依据见[附录 A](appendix-a-bfcl-tau2-rationale.md)。

## P2 详表(47 项)

P2 对应 122B 模型卡的视觉、软件工程、终端、多语言和空间能力。当前大多缺视觉环境、外部 harness 或需要协议核对,不作为本阶段正式评测目标。

### 语言 — 指令(1 项)

| Benchmark | 能力 | 环境要求 | 状态 |
|---|---|---|---|
| MultiChallenge | 多轮指令遵循 | 官方 harness 和协议 | `BLOCKED_EXTERNAL` |

### 语言 — 代码(6 项)

| Benchmark | 能力 | 环境要求 | 状态 |
|---|---|---|---|
| SWE-bench Verified | 软件工程 | Docker + oracle 协议 | `REVIEW`:full Verified,需 Docker 资源 |
| Terminal Bench 2 | 终端操作 | Harbor + Docker,固定 task 仓库 revision | `REVIEW` |
| CodeForces | 竞赛代码 | 官方 harness 和 sandbox | `BLOCKED_EXTERNAL` |
| OJBench | 代码 | 官方 harness 和 sandbox | `BLOCKED_EXTERNAL` |
| FullStackBench en | 全栈代码(英) | browser/service sandbox | `BLOCKED_EXTERNAL` |
| FullStackBench zh | 全栈代码(中) | browser/service sandbox | `BLOCKED_EXTERNAL` |

### 语言 — Agent(2 项)

| Benchmark | 能力 | 环境要求 | 状态 |
|---|---|---|---|
| VITA-Bench | 通用 Agent | 工具 fixture、用户模拟器、judge | `BLOCKED_EXTERNAL` |
| DeepPlanning | 规划 Agent | 官方 agent harness 和 judge | `BLOCKED_EXTERNAL` |

### 语言 — 多语言(6 项)

| Benchmark | 能力 | 环境要求 | 状态 |
|---|---|---|---|
| MMLU-ProX | 多语言知识(29 语言) | 29 语言 adapter 和聚合 | `BLOCKED_EXTERNAL` |
| NOVA-63 | 多语言知识(63 语言) | 63 语言 adapter 和聚合 | `BLOCKED_EXTERNAL` |
| INCLUDE | 多语言覆盖 | 多语言 adapter 和聚合 | `BLOCKED_EXTERNAL` |
| Global PIQA | 多语言常识 | 注意 local PIQA ≠ Global PIQA | `BLOCKED_EXTERNAL` |
| WMT24++ | 机器翻译 | COMET/XCOMET-XXL,55 语言 | `REVIEW`:需独立翻译环境 |
| MAXIFE | 多语言指令(23 设置) | 23 设置多语言指令 adapter | `BLOCKED_EXTERNAL` |

### 视觉 — STEM/解谜(9 项)

| Benchmark | 能力 | 环境要求 | 状态 |
|---|---|---|---|
| MMMU | 多学科视觉推理 | 30 subsets,图像传输 | `REVIEW` |
| MMMU-Pro | 多学科视觉推理(增强) | 30 subsets,图像传输 | `REVIEW` |
| MathVision | 数学视觉推理 | 5 subsets,对齐 boxed-answer | `REVIEW` |
| Mathvista(mini) | 数学视觉推理 | adapter 使用 testmini split | `CONFIG_FROZEN` |
| DynaMath | 动态数学视觉 | 官方视觉推理 adapter | `BLOCKED_EXTERNAL` |
| ZEROBench | 零样本视觉推理 | zerobench split + 独立 judge | `REVIEW` |
| ZEROBench_sub | 零样本视觉推理(子集) | adapter train split 存在但 CLI 未暴露模型卡子分 | `BLOCKED_PROTOCOL` |
| VlmsAreBlind | 视觉盲区测试 | 官方视觉推理 adapter | `BLOCKED_EXTERNAL` |
| BabyVision | 婴儿级视觉 | 官方 adapter 和双分协议 | `BLOCKED_EXTERNAL` |

### 视觉 — 通用 VQA(5 项)

| Benchmark | 能力 | 环境要求 | 状态 |
|---|---|---|---|
| RealWorldQA | 真实世界 VQA | 图像传输 smoke | `REVIEW` |
| MMStar | 综合 VQA | 6 subset 聚合 | `REVIEW` |
| MMBenchEN-DEV-v1.1 | 英文 VQA | adapter en dev 未固定到 v1.1 | `REVIEW` |
| SimpleVQA | 简单 VQA | 独立 judge | `BLOCKED_JUDGE` |
| HallusionBench | 视觉幻觉 | 对齐三项报告指标 | `REVIEW` |

### 视觉 — 文档(6 项)

| Benchmark | 能力 | 环境要求 | 状态 |
|---|---|---|---|
| OmniDocBench1.5 | 文档理解 | 本地 adapter 实现官方 v1.5 评分 | `CONFIG_FROZEN` |
| CharXiv(RQ) | 论文图表理解 | PDF 渲染 + RQ scorer | `BLOCKED_EXTERNAL` |
| MMLongBench-Doc | 长文档理解 | 多页渲染 + 长文档 scorer | `BLOCKED_EXTERNAL` |
| CC-OCR | OCR | 官方 OCR adapter 和 scorer | `BLOCKED_EXTERNAL` |
| AI2D_TEST | 科学图表 | adapter 使用 test split | `CONFIG_FROZEN` |
| OCRBench | OCR | 使用 OCRBench(非 v2) | `CONFIG_FROZEN` |

### 视觉 — 空间(10 项)

| Benchmark | 能力 | 环境要求 | 状态 |
|---|---|---|---|
| ERQA | 空间推理 | grounding 资产和 scorer | `BLOCKED_EXTERNAL` |
| CountBench | 计数 | 计数 adapter 和 scorer | `BLOCKED_EXTERNAL` |
| RefCOCO(avg) | 指代表达 | 4 subsets avg 计算 | `REVIEW` |
| ODInW13 | 目标检测 | 检测资产和 scorer | `BLOCKED_EXTERNAL` |
| EmbSpatialBench | 嵌入式空间 | 空间资产和 scorer | `BLOCKED_EXTERNAL` |
| RefSpatialBench | 指代空间 | grounding 输出协议 | `BLOCKED_EXTERNAL` |
| LingoQA | 驾驶视频 QA | 驾驶视频资产和 scorer | `BLOCKED_EXTERNAL` |
| Hypersim | 3D 场景 | 3D 资产和 grounding scorer | `BLOCKED_EXTERNAL` |
| SUNRGBD | 3D 场景 | 3D 资产和 grounding scorer | `BLOCKED_EXTERNAL` |
| Nuscene | 自动驾驶场景 | nuScenes 资产许可和 scorer | `BLOCKED_EXTERNAL` |

### 视觉 — 工具调用(2 项)

| Benchmark | 能力 | 环境要求 | 状态 |
|---|---|---|---|
| TIR-Bench | 视觉工具调用 | 对齐 tool-use/no-tool 双分 | `REVIEW` |
| V* | 视觉搜索 | 对齐 grounding 和双分协议 | `REVIEW` |

## P3 详表(18 项)

P3 对应 122B 模型卡的搜索 Agent、视频理解、GUI 操作和医疗 VQA。全部需要单独建设外部 harness、视频资源或 VM 环境,当前均为 `BLOCKED_EXTERNAL`。

### 语言 — 搜索 Agent(5 项)

| Benchmark | 能力 | 环境要求 |
|---|---|---|
| HLE w/ tool | 搜索增强推理 | 搜索后端凭据、folding、judge |
| Browsecomp | 浏览搜索 | 搜索后端凭据和 judge |
| Browsecomp-zh | 中文浏览搜索 | 中文搜索后端凭据和 judge |
| WideSearch | 宽域搜索 | 256K 上下文 + 实时搜索后端 |
| Seal-0 | 搜索验证 | 搜索后端凭据和 judge |

### 视觉 — 视频(7 项)

| Benchmark | 能力 | 环境要求 |
|---|---|---|
| VideoMME(w sub.) | 视频理解(带字幕) | 视频文件、字幕策略、帧采样 |
| VideoMME(w/o sub.) | 视频理解(无字幕) | 视频文件、无字幕策略、帧采样 |
| VideoMMMU | 视频多学科 | 视频 harness 和帧采样 |
| MLVU | 长视频理解 | 视频 harness 和帧采样 |
| MVBench | 视频理解 | 视频 harness 和帧采样 |
| LVBench | 长视频理解 | 长视频 harness 和帧采样 |
| MMVU | 视频理解 | 视频 harness 和 judge |

### 视觉 — 视觉 Agent(3 项)

| Benchmark | 能力 | 环境要求 |
|---|---|---|
| ScreenSpot Pro | 屏幕定位 | 桌面截图、grounding、action scorer |
| OSWorld-Verified | 桌面操作 | VM 快照、桌面 executor、task server |
| AndroidWorld | Android 操作 | Android 模拟器快照和 task executor |

### 视觉 — 医疗(3 项)

| Benchmark | 能力 | 环境要求 |
|---|---|---|
| SLAKE | 医疗 VQA | 数据集许可、视觉 adapter、scorer |
| PMC-VQA | 医疗 VQA | 数据集许可、视觉 adapter、scorer |
| MedXpertQA-MM | 医疗专家 QA | 数据集许可、视觉 adapter、scorer |
