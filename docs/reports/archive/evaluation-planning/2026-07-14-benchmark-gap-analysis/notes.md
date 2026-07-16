# 82 项模型评测覆盖差距：分析说明

## 问题与范围

- 问题：对照用户提供的 Qwen3.5 Language / Vision Language 82 项评测表，判断当前项目做模型测评还缺什么。
- 目标模型：当前部署的 Qwen3.6-27B-FP8 与 ThinkingCap-Qwen3.6-27B-FP8。
- 截止时间：2026-07-14。
- 比较单位：benchmark 行；同一名称的不同版本、subset 或协议不自动视为严格等价。

## 控制来源

1. `docs/reports/archive/evaluation-references/2026-07-14-qwen35-benchmark-score-reference.md`：用户提供的 82 项候选 benchmark。
2. `benchmarks/evalscope/capability_suites.tsv`：项目当前自动化主套件，共 30 项。
3. 本地 EvalScope 1.6.1 `evalscope benchmark-info --list`：当前安装版本注册的 158 个 benchmark adapter。
4. `benchmarks/thinkingcap/datasets.yaml` 与 `benchmarks/thinkingcap/PREPARATION_STATUS.md`：ThinkingCap 模型卡任务的数据来源、缓存和专用 harness 状态。
5. `logs/eval/`：本地已经生成的实际评测输出与 dry-run/preflight 状态。

## 状态定义

- **主套件已接入**：同名任务已经写入 `capability_suites.tsv`；只代表脚本可调度，不代表数据已完整下载或已跑出成绩。
- **部分接入**：存在主套件任务，但目标版本、subset 或规模不一致，例如 LiveCodeBench 未固定 v6、SWE-bench 当前只配置 Verified mini。
- **EvalScope 已注册，待接入/对齐**：本地 EvalScope 1.6.1 有同名或近似 adapter，但主套件未配置，或仍需核对版本、subset、prompt、judge。
- **未接入**：主套件和本地 EvalScope 注册表均未找到可直接对应的任务，需要外部/官方 harness 或自定义适配器。
- **已有结果**：必须存在真实模型请求和评分输出；`DRY_RUN`、依赖检查和模型卡公开分数都不算本地结果。

## 汇总结果

| 覆盖状态 | Language | Vision Language | 合计 |
|---|---:|---:|---:|
| 主套件已接入 | 10 | 0 | 10 |
| 部分接入 | 2 | 0 | 2 |
| EvalScope 已注册，待接入/对齐 | 9 | 17 | 26 |
| 未接入 | 16 | 28 | 44 |
| 合计 | 37 | 45 | 82 |

## 已有结果与证据风险

- 82 项中只有 MMLU-Pro 存在较成体系的本地 pilot：ThinkingCap 280 题 exact-match 84.29%；Qwen 对照未完成，因此仍不能形成有效 A/B。
- Qwen 的 HumanEval-Plus 与 MBPP-Plus 各有 2 题 sandbox smoke 成功，但这两项不在本次 82 项参考表中，只能证明代码沙箱链路可用。
- LiveCodeBench 已进入实际运行但停在约 1.25GB 数据下载阶段，没有结果。
- 其余主套件任务目前主要是 `DRY_RUN` / preflight，不能当作能力成绩。
- 用户提供的 82 项公开分数尚未逐项对照原始模型卡核验，不能作为严格官方基线引用。

## 关键协议缺口

- **版本与 subset**：LiveCodeBench v6、HMMT Feb/Nov 2025、OmniDocBench 1.5、ZEROBench_sub、Global PIQA 等不能只靠近似 adapter 名称认定等价。
- **Judge**：AA-LCR、Search Agent、部分 VQA/OCR/医疗任务需要固定 judge 模型、prompt、温度、盲评与重试策略。
- **代码与 Agent 环境**：SWE-bench、Terminal Bench、TAU2、OSWorld、AndroidWorld 需要固定容器镜像、CPU/RAM、timeout、网络、工具权限和 max turns。
- **长上下文管理**：Search Agent 多数使用 256K context folding；WideSearch 明确不做 context management。两种协议必须分开。
- **多模态输入**：需要固定图片/视频解码、分辨率、帧采样、文档渲染、OCR 预处理和 max pixels，当前主套件没有这一层统一配置。

## 建议执行分层

1. **P0：先跑已有文本核心，建立可信 A/B**：MMLU-Pro、MMLU-Redux、C-Eval、GPQA Diamond、IFEval、IFBench、LongBench v2；统一 run-id、题目、seed、temperature、max tokens 和解析器。
2. **P1：补齐编码与通用 Agent**：固定 LiveCodeBench v6，增加完整 SWE-bench Verified 与 Terminal Bench 2；再跑 BFCL-V4、TAU2-Bench。
3. **P2：建立多模态最小套件**：从 EvalScope 已注册任务中先接 MMMU-Pro、MathVision、RealWorldQA、OCRBench、TIR-Bench，验证图像、OCR、空间和视觉工具调用链路。
4. **P3：最后做专用 Agent/视频/医疗**：Browsecomp、WideSearch、VideoMME、ScreenSpot Pro、OSWorld、AndroidWorld、SLAKE、PMC-VQA、MedXpertQA-MM。它们需要单独 harness 和环境治理，不能靠通用 QA runner 伪装完成。

## 可视化合同

- 问题：82 项候选 benchmark 在当前项目中处于什么接入状态？
- 形式：单系列横向柱状图，四个互斥状态，按数量降序排列。
- 数据：82 个 benchmark 行按上述状态定义计数；图表用于显示工程覆盖差距，不表示模型能力。
- 配色：单一蓝色根色；不使用颜色承担额外语义；精确值由标签和下方明细表提供。
