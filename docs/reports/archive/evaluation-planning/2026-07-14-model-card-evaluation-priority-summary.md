# 模型卡 82 项评测优先级汇总

日期：2026-07-14

用途：面向 leader 汇报 `Qwen3.6-27B-FP8` 与 `ThinkingCap-Qwen3.6-27B-FP8` 的模型卡评测范围、优先级、执行条件和当前进度。

详细协议和执行规则见 [三模型模型卡对比评测最终方案](2026-07-14-model-card-evaluation-plan.md)，环境就绪情况见 [剩余 75 项 benchmark 环境配置状态](2026-07-14-remaining-benchmark-environment-status.md)。

## 1. 汇报结论

82 项评测不同时铺开，而是按汇报价值、能力增量、可复现性和执行成本分为 P0–P4：

1. P0 先形成可信的第一版模型卡；
2. P1 补齐数学、代码、工具、Agent 和多语言能力；
3. P2 验证视觉最小套件及中等复杂度执行环境；
4. P3 在视觉链路稳定后扩大视觉覆盖；
5. P4 作为需要专项基础设施和预算的候选池。

这里必须区分两个维度：

| 维度 | 回答的问题 | 示例 |
|---|---|---|
| 业务优先级 P0–P4 | 这个结果对模型卡和决策有多大价值、应该先做什么 | HMMT Nov 25 因三张模型卡共同出现而属于 P1 |
| 环境可执行性 | 当前是否已有 runner、数据、judge 和 worker | 高优先级任务也可能因缺 judge 暂时阻塞 |

因此，`BLOCKED` 表示基础设施或协议缺口，不表示模型能力失败，也不会自动降低任务的业务价值。

## 2. 总体优先级

| 优先级 | 数量 | 定位 | 主要能力 | 执行策略 |
|---|---:|---|---|---|
| P0 | 7 | 第一版模型卡必须完成 | 指令、知识、中文、高难推理、长上下文 | 当前主线，优先占用评测资源 |
| P1 | 10 | 高价值能力扩展 | 数学、代码、工具、Agent、多语言 | P0 后第一批扩展 |
| P2 | 13 | 视觉最小套件与中等基础设施 | 视觉、软件工程、终端、多语言 | 通过视觉、sandbox、judge gate 后运行 |
| P3 | 12 | 视觉能力扩展 | 通用视觉、幻觉、文档、grounding | P2 视觉链路稳定后推进 |
| P4 | 40 | 专项基础设施任务池 | 搜索、视频、3D、GUI、Android、医疗 | 不纳入近期承诺，按需要单独立项 |
| 合计 | **82** | 完整候选集 | — | — |

优先级采用以下判断标准：

| 判断因素 | 说明 |
|---|---|
| 汇报价值 | 能否直接支撑三模型对比和 leader 决策 |
| 能力增量 | 是否补充新的能力维度，而不是重复已有结论 |
| 可复现性 | 数据、prompt、版本、harness 和评分器能否冻结 |
| 执行成本 | 是否需要 sandbox、judge、搜索、VM 或大型数据 |
| 协议风险 | 失败是否会导致整批成绩不可比较或不可解释 |

## 3. P0：第一版模型卡核心任务

| 顺序 | Benchmark | 能力维度 | 主要价值 | 当前动作 |
|---:|---|---|---|---|
| 1 | IFEval | 指令遵循 | 严格验证格式、关键词和多条件约束 | 完成 541 题双模型正式运行并验收 |
| 2 | GPQA Diamond | 高难科学推理 | 检查研究生级科学知识和多步推理 | 冻结 revision 后运行 198 题 |
| 3 | IFBench | 复杂指令遵循 | 复核组合约束，补充 IFEval | 运行 300 题双模型正式评测 |
| 4 | MMLU-Pro | 综合知识与推理 | 第一版模型卡的主要综合能力指标 | 运行 12,032 题同题 A/B |
| 5 | LongBench v2 | 长上下文 | 检查长文档理解、检索和推理 | 503 题中运行预算内 400 题，显式记录 103 题超限 |
| 6 | C-Eval | 中文知识与推理 | 补充中文考试和专业学科能力 | 固定生成式多选口径后全量 |
| 7 | MMLU-Redux | 综合知识复核 | 用修订题目复核 MMLU 类结论 | 检查 57 个 config 和 revision 后全量 |

P0 能力覆盖：

| 能力 | 测试集 |
|---|---|
| 指令遵循 | IFEval、IFBench |
| 综合知识 | MMLU-Pro、MMLU-Redux |
| 中文能力 | C-Eval |
| 高难推理 | GPQA Diamond |
| 长上下文 | LongBench v2 |

P0 暂不覆盖代码执行、工具调用、Agent、多语言和视觉，这些能力分别在 P1 和 P2 中补齐。

## 4. P1：高价值能力扩展

| 顺序 | Benchmark | 能力维度 | 排入 P1 的原因 |
|---:|---|---|---|
| 1 | HMMT Nov 25 | 高难数学 | 三张候选模型卡共同出现，直接比较价值最高 |
| 2 | LiveCodeBench v6 | 代码生成与执行 | 补齐 P0 未覆盖的真实编程能力 |
| 3 | BFCL-V4 | Function Calling | 测试函数选择、参数生成和多工具调用 |
| 4 | TAU2-Bench | 多轮 Agent | 测试用户模拟、工具使用和多轮任务完成 |
| 5 | HLE w/ CoT | 高难综合推理 | 补充更困难、更开放的知识推理 |
| 6 | SuperGPQA | 专业知识 | 通过 72 个 subsets 扩大专业知识覆盖 |
| 7 | HMMT Feb 25 | 高难数学 | 复用 HMMT 链路，验证数学结果稳定性 |
| 8 | MMMLU | 多语言知识 | 检查不同语言下知识能力是否稳定 |
| 9 | PolyMATH | 多语言数学 | 检查数学推理是否受语言变化影响 |
| 10 | AA-LCR | 长上下文复核 | 使用不同协议复核 LongBench v2 结论 |

P1 能力汇总：

| 能力 | 数量 | 测试集 |
|---|---:|---|
| 高难数学 | 2 | HMMT Nov 25、HMMT Feb 25 |
| 代码 | 1 | LiveCodeBench v6 |
| 工具调用 | 1 | BFCL-V4 |
| Agent | 1 | TAU2-Bench |
| 高难推理与知识 | 2 | HLE w/ CoT、SuperGPQA |
| 多语言 | 2 | MMMLU、PolyMATH |
| 长上下文 | 1 | AA-LCR |

P1 的阶段验收标准：前 4 项至少形成双模型完整结果；10 项全部达到 `CONFIG_FROZEN`，或明确记录阻塞原因。HMMT Feb 25 已完成 Qwen 单题端到端 smoke，证明数据、请求、答案抽取和聚合链路可用，但 1/1 结果不作为质量结论。

## 5. P2：视觉最小套件与中等基础设施

P2 分为视觉和文本执行环境两条轨道。

### 5.1 P2-A：视觉最小套件（5 项）

| 顺序 | Benchmark | 能力维度 | 作用 |
|---:|---|---|---|
| 1 | MMMU-Pro | 综合视觉推理 | 综合判断学科、图片理解和推理能力 |
| 2 | MathVision | 视觉数学 | 测试图形、几何和公式推理 |
| 3 | RealWorldQA | 真实图像理解 | 测试真实场景中的通用视觉问答 |
| 4 | OCRBench | OCR 与文档 | 测试图片文字、表格和文档识别 |
| 5 | TIR-Bench | 视觉工具调用 | 测试视觉信息与工具调用的结合 |

视觉正式运行前必须通过以下 gate：

| Gate | 验收要求 |
|---|---|
| 输入支持 | 两个端点能够接收图片消息 |
| 请求一致 | 使用相同的图片格式、尺寸和 chat template |
| 输出可评分 | adapter 和评分器能够正常处理结果 |
| 模态处理 | 不支持视觉时记为 `N/A / unsupported modality`，不能记为 0 分 |
| 扩展条件 | 至少 3 项得到双模型有效结果后再进入 P3 |

### 5.2 P2-B：中等基础设施文本任务（8 项）

| 顺序 | Benchmark | 能力维度 | 主要前置条件 |
|---:|---|---|---|
| 1 | SWE-bench Verified | 软件工程 Agent | 完整 Verified 数据、仓库容器和测试环境 |
| 2 | Terminal Bench 2 | 终端任务执行 | 固定镜像、CPU、RAM、网络和 timeout |
| 3 | WMT24++ | 多语言翻译 | 固定 XCOMET 模型 revision 和 55 语言聚合 |
| 4 | Global PIQA | 多语言常识 | 确认不是用普通 PIQA 近似替代 |
| 5 | MultiChallenge | 复杂指令遵循 | 补接 adapter 并固定评分协议 |
| 6 | MMLU-ProX | 多语言知识 | 固定语言集合和宏平均 |
| 7 | INCLUDE | 多语言与区域知识 | 准备 adapter 与专用指标 |
| 8 | MAXIFE | 多语言指令遵循 | 固定语言范围和聚合方式 |

| P2 轨道 | 数量 | 定位 |
|---|---:|---|
| P2-A 视觉最小套件 | 5 | 判断是否值得扩大视觉评测 |
| P2-B 中等基础设施 | 8 | 软件工程、终端、多语言和复杂指令 |
| 合计 | **13** | — |

## 6. P3：视觉能力扩展

| 分类 | 数量 | Benchmark | 主要能力 |
|---|---:|---|---|
| 通用视觉与推理 | 8 | MMMU、Mathvista(mini)、ZEROBench、ZEROBench_sub、MMStar、MMBenchEN-DEV-v1.1、SimpleVQA、HallusionBench | 综合视觉、数学、常识与幻觉 |
| 文档与图表 | 2 | OmniDocBench1.5、AI2D_TEST | 文档结构、示意图和图表理解 |
| Grounding 与工具 | 2 | RefCOCO(avg)、V* | 目标定位、区域理解和视觉搜索 |
| 合计 | **12** | — | — |

版本对齐重点：

| Benchmark | 必须确认的问题 |
|---|---|
| Mathvista(mini) | 是否为模型卡使用的 mini split |
| ZEROBench_sub | 是否存在可复现的独立 subset 和子分数 |
| MMBenchEN-DEV-v1.1 | 是否严格对应英文 DEV v1.1 |
| OmniDocBench1.5 | adapter 是否对应 1.5 版本 |
| RefCOCO(avg) | 4 个 subsets 和平均分计算方式 |
| V* | grounding 设置和双指标记录方式 |

P3 排在 P2 之后，因为 P2 的 5 项已能覆盖综合视觉、数学、真实场景、OCR 和工具调用。P3 的作用是验证结论稳定性，并细分幻觉、文档图表和 grounding 能力。

## 7. P4：专项基础设施任务池

| 分类 | 数量 | Benchmark | 主要依赖 |
|---|---:|---|---|
| 自建文本、代码与 Agent | 7 | CodeForces、OJBench、FullStackBench en、FullStackBench zh、VITA-Bench、DeepPlanning、NOVA-63 | 新 adapter、代码 sandbox、Agent harness |
| 搜索 Agent | 5 | HLE w/ tool、Browsecomp、Browsecomp-zh、WideSearch、Seal-0 | 搜索后端、凭据、网页工具、judge |
| 未接入视觉推理与文档 | 6 | DynaMath、VlmsAreBlind、BabyVision、CharXiv(RQ)、MMLongBench-Doc、CC-OCR | 视觉 adapter、PDF 渲染、文档评分器 |
| 空间与 3D | 9 | ERQA、CountBench、ODInW13、EmbSpatialBench、RefSpatialBench、LingoQA、Hypersim、SUNRGBD、Nuscene | 3D 数据、坐标输出、grounding 评分 |
| 视频理解 | 7 | VideoMME(w sub.)、VideoMME(w/o sub.)、VideoMMMU、MLVU、MVBench、LVBench、MMVU | 视频数据、抽帧、字幕、视频输入 |
| GUI 与 Android Agent | 3 | ScreenSpot Pro、OSWorld-Verified、AndroidWorld | VM、模拟器、动作执行器、环境恢复 |
| 医疗 VQA | 3 | SLAKE、PMC-VQA、MedXpertQA-MM | 数据许可、医学评分和专业 judge |
| 合计 | **40** | — | — |

P4 不是永久不做。单项同时满足以下条件后，可以升级：

| 条件 | 要求 |
|---|---|
| 业务价值 | leader 明确需要该能力维度 |
| Harness | 存在官方或可审计实现 |
| 数据 | 许可允许使用，版本可以冻结 |
| 成本 | GPU、存储、judge 和工期有明确上限 |
| 安全 | VM、代码和搜索工具与模型宿主机隔离 |
| 可复现性 | 结果可以复查并与官方口径比较 |

## 8. 当前环境可执行性

截至 2026-07-14，本地已为剩余 75 项生成任务目录、协议骨架和双模型请求配置。当前环境状态为：

| 环境状态 | 数量 | 说明 |
|---|---:|---|
| 本地 runner 已生成，可进入 preflight | 23 | 依赖和 runner 基本就绪，部分仍需协议 review |
| 缺独立 judge | 5 | AA-LCR、HLE w/ CoT、ZEROBench、SimpleVQA、TIR-Bench |
| 外部 harness、数据或 worker 阻塞 | 46 | 搜索、Agent、代码、视频、3D、GUI、Android、医疗或非公开协议 |
| 协议阻塞 | 1 | ZEROBench_sub 当前 adapter 不暴露模型卡所需子分数 |
| 合计 | **75** | 不含正在运行的 7 项 P0 |

严格按任务状态统计：

| 状态 | 数量 |
|---|---:|
| `SMOKE_QWEN_SUCCESS` | 1 |
| `READY_PREFLIGHT` | 5 |
| `BLOCKED_PREFLIGHT` | 22 |
| `BLOCKED_EXTERNAL` | 46 |
| `BLOCKED_PROTOCOL` | 1 |
| 合计 | **75** |

P4 有 40 项，但环境统计中有 46 项外部阻塞，原因是少量 P1–P3 高价值任务当前同样缺 judge、数据或外部 worker。优先级与环境状态不能相互替代。

## 9. 当前执行进展快照

截至 2026-07-14 20:12：

| 项目 | Qwen | ThinkingCap | 说明 |
|---|---:|---:|---|
| IFEval | 541/541，完成 | 541/541，完成 | 等待结果完整性和指标验收 |
| IFBench | 144/300，运行中 | 182/300，运行中 | 两模型并行，同一配置运行 |
| LongBench v2 协议准备 | 400 题可运行 | 400 题可运行 | 原始 503 题中 103 题明确记录为超预算，不静默截断 |
| 剩余 75 项骨架 | 75/75 已生成 | 75/75 已生成 | 每项均有独立任务目录和状态索引 |

说明：上表是时间点快照，正式汇报时应以 `logs/eval/modelcard/modelcard-core-full-20260714/` 和任务状态索引的最新记录为准。

## 10. 执行路线与阶段产出

| 阶段 | 目标 | 主要产出 | 粗略时间 |
|---|---|---|---|
| P0 | 完成 7 项核心评测 | 第一版三模型对比模型卡 | 约 1–3 天 |
| P1 | 补齐数学、代码、工具、Agent 和多语言 | 文本与 Agent 能力扩展结论 | 配置 2–4 天，运行约 3–7 天 |
| P2-A | 验证视觉最小套件 | 视觉方向 go/no-go 结论 | 约 2–4 天 |
| P2-B | 验证软件工程、终端和多语言 | 真实任务执行能力报告 | 约 3–7 天 |
| P3 | 扩大视觉覆盖 | 视觉能力细分报告 | 约 1–2 周 |
| P4 | 按业务需求单独立项 | 搜索、视频、3D、GUI 或医疗专项报告 | 不纳入当前承诺工期 |

## 11. 调度和止损规则

| 规则 | 内容 |
|---|---|
| GPU 优先级 | P0 优先，其次 P1、P2、P3；P4 不抢占 P0/P1 资源 |
| 双模型对比 | 同一 benchmark 的两个模型并行，使用同题、同配置和同评分协议 |
| 并行边界 | 正式运行按 benchmark 串行；协议研究、数据 preflight 和 smoke 可并行 |
| 协议止损 | 半个工作日内仍不能冻结协议则标记 `BLOCKED`，继续下一项 |
| 成本止损 | 超过 1 GPU-day、需要收费 API、大型许可数据或 VM 时先估算再执行 |
| 视觉止损 | P2-A 至少 3 项形成双模型有效结果后才扩大到 P3 |
| 安全边界 | 不重启 H100 宿主机，不改写模型权重，外部执行环境与模型服务隔离 |

## 12. Leader 需要确认的决策

| 决策 | 建议 |
|---|---|
| 第一版模型卡范围 | 以 P0 的 7 项作为必须交付，先形成可审计结果 |
| 第一扩展批次 | 优先批准 P1 前四项：HMMT Nov 25、LiveCodeBench v6、BFCL-V4、TAU2-Bench |
| 视觉投入 | 先批准 P2-A 的 5 项最小套件，根据 go/no-go 结果决定是否进入 P3 |
| 外部环境预算 | P4 不整体承诺；按搜索、视频、GUI、3D、医疗分别立项 |
| Judge/API 成本 | 对需要独立 judge 或收费 API 的任务设单独预算上限 |

汇报时可概括为：

> 82 项评测不会同时铺开。P0 先形成可信的基础模型卡，P1 补齐数学、代码、工具、Agent 和多语言，P2 验证视觉及复杂执行环境，P3 扩大视觉覆盖，P4 作为需要专项基础设施和预算的候选池。优先级代表业务价值，环境状态代表当前可执行性；高价值任务如果协议或环境未就绪，会明确记录阻塞，而不会产出不可复现的近似分数。
