# 三模型模型卡对比评测最终方案

日期：2026-07-14

## 1. 目标

形成一份可向 leader 汇报、过程可追溯的模型卡，对比：

- `Qwen3.5-122B-A10B`：引用官方模型卡分数；
- `Qwen3.6-27B-FP8`：远程 node1 `8000` 端点实测；
- `ThinkingCap-Qwen3.6-27B-FP8`：远程 node1 `8001` 端点实测。

候选范围为原始模型卡中的 82 个 benchmark：37 个 Language、45 个 Vision Language。最终不仅记录分数，还要记录题库规模、评测标准、请求参数、数据版本、执行过程、失败与修复，保证结果可以复查。

本地跑分统一称为“官方/经验推荐配置跑分”，不宣称经过无边界搜索得到全局最优配置。

## 2. 最终模型卡需要展示的内容

每个 benchmark 至少包含：

| 字段 | 说明 |
|---|---|
| Benchmark | 官方名称和本地任务 ID |
| Category / Modality | 能力分类和语言/视觉模态 |
| Official score | Qwen3.5-122B-A10B 官方模型卡分数 |
| Qwen score | Qwen3.6-27B-FP8 本地实测分数 |
| ThinkingCap score | ThinkingCap-Qwen3.6-27B-FP8 本地实测分数 |
| Question count | 官方题量、加载题量、实际成功评分题量 |
| Evaluation standard | 指标、答案抽取、聚合、judge、few-shot 等 |
| Request config | temperature、top_p、top_k、max_tokens、seed 等 |
| Protocol grade | `Exact`、`Close`、`Approximate` 或 `Not reproducible` |
| Execution status | `SUCCESS`、`FAILED`、`BLOCKED` 或 `N/A` |
| Evidence | 配置、命令、日志、原始预测和报告路径 |

不同单位的指标不得混合处理。例如 CodeForces 使用 rating/Elo，SWE-bench 使用 resolved rate，普通选择题使用 accuracy；BabyVision、TIR-Bench、V* 等双分数任务必须分别保存两个指标。

## 3. 配置选择原则

请求配置按以下优先级确定：

1. Benchmark 官方 harness 明确给出的配置；
2. Qwen3.5-122B-A10B 模型卡披露的配置；
3. 被测模型官方推荐配置；
4. 任务类型的经验配置。

不根据测试集分数反复调参。只有以下情况允许修改配置：

- API 不接受某个请求字段；
- 输出格式无法由官方评分器解析；
- dataset、split、subset、prompt 或 few-shot 配置错误；
- judge、代码 sandbox 或工具环境失败；
- 上下文超过模型限制；
- 后续核对发现与官方协议不一致。

每次修改都必须记录旧配置、错误、修改原因、新配置和重试结果。

### 3.1 默认经验配置

官方协议没有明确说明时使用下表：

| 任务类型 | temperature | top_p | top_k | max_tokens |
|---|---:|---:|---:|---:|
| 格式严格、IFEval、分类 | 0 | 1.0 | -1 | 4096–8192 |
| 知识 MCQ、推理、数学 | 0.6 | 0.95 | 20 | 8192–16384 |
| 长上下文 | 0.6 | 0.95 | 20 | 16384 |
| Coding pass@1 | 0.2 | 0.95 | 20 | 16384–32768 |
| Agent / Tool Calling | 0.6 | 0.95 | 20 | 16384 |
| VLM 选择题、OCR | 0 | 1.0 | -1 | 4096–8192 |
| 开放视觉推理 | 0.6 | 0.95 | 20 | 8192–16384 |
| Judge | 0 | 1.0 | -1 | 1024–4096 |

官方要求多 seed 时按官方执行；否则使用 `seed=42`。thinking、reasoning parser、tool parser 和 chat template 必须显式记录，不能只依赖服务端隐式默认值。

## 4. 第一阶段：7 个核心任务

第一阶段优先形成可信的文本模型卡主体。

### 4.1 MMLU-Pro

- Adapter：`mmlu_pro`
- 14 个 subset；当前本地加载证据显示 test 为 12,032 题
- EvalScope 默认 5-shot CoT
- 主指标：mean accuracy
- 建议请求：`temperature=0.6`、`top_p=0.95`、`top_k=20`、`max_tokens=8192`、`seed=42`

### 4.2 MMLU-Redux

- Adapter：`mmlu_redux`
- 57 个学科
- 正式 preflight 时固定 dataset revision、split 和准确题量
- 主指标：accuracy
- 使用知识 MCQ profile

### 4.3 C-Eval

- Adapter：`ceval`
- 52 个学科
- 使用 EvalScope 可评分 split
- 正式 preflight 时固定准确题量
- 主指标：accuracy
- 使用知识 MCQ profile

### 4.4 GPQA Diamond

- Adapter：`gpqa_diamond`
- 198 题
- 0-shot CoT
- 主指标：accuracy
- 建议 `max_tokens=16384`

### 4.5 IFEval

- Adapter：`ifeval`
- 官方约 541 条指令，最终以冻结 revision 的实际加载量为准
- 指标：prompt-level 和 instruction-level strict/loose accuracy
- 格式敏感，使用 `temperature=0`、`top_p=1.0`、`top_k=-1`、`max_tokens=8192`

### 4.6 IFBench

- Adapter：`ifbench`
- 正式 preflight 时记录准确题量
- 记录所有官方指令遵循指标
- 使用与 IFEval 相同的确定性配置

### 4.7 LongBench v2

- Adapter：`longbench_v2`
- short、medium、long 三组
- 正式模型卡应按官方口径选择完整 subset，不能把 short/medium pilot 写成全量成绩
- 主指标：accuracy
- 建议 `temperature=0.6`、`top_p=0.95`、`top_k=20`、`max_tokens=16384`
- 必须记录每题输入长度和超过 262K 上下文限制时的处理方式，禁止静默截断

## 5. 7 项剩余执行优先级

任务之间按顺序运行；同一个 benchmark 的两个模型并行运行。这样既利用两个独立服务，又能将数据、prompt 或评分器问题定位到单一任务。

优先级不只按题量从小到大排列，而按以下顺序判断：

1. 能否直接支撑第一版模型卡和 leader 汇报；
2. 是否补充新的能力维度，而不是重复已有知识类结论；
3. 协议、数据和 adapter 是否已就绪；
4. 预计 GPU 时间、judge 或 sandbox 成本；
5. 是否存在会使整批结果失效的协议风险。

### 5.1 2026-07-14 19:34 状态快照

- 两个远端服务均健康，正式评测期间 GPU 0–3 处于高负载；
- `IFEval`、`IFBench`、`GPQA Diamond`、`MMLU-Pro`、`MMLU-Redux` 和 `C-Eval` 的双模型 smoke 已完成；
- smoke 的 `--limit 1` 是每 subset 限制，因此 `MMLU-Pro`、`MMLU-Redux`、`C-Eval` 实际分别处理 14、57、52 条，不可写成各 1 题；
- `LongBench v2` 的双模型 smoke 均在 long 样本失败：输入至少 245,761 tokens，再预留 16,384 输出后超过 262,144 上下文上限；
- `IFEval` 541 题双模型正式运行已经启动。已启动任务优先自然跑完，不为切换队列而中断。

smoke 只证明数据、请求和评分链路基本可用，不代表 benchmark 已完成。

### 5.2 P0：第一版模型卡必须完成

| 顺序 | Benchmark | 当前动作 | 排序理由 |
|---:|---|---|---|
| 1 | IFEval | 完成已经启动的 541 题双模型正式运行并验收 | 已在运行；格式和指令遵循是独立能力维度 |
| 2 | GPQA Diamond | 冻结 revision 后运行 198 题全量 | 题量小、推理信号强，也是多张模型卡的直接对比项 |
| 3 | IFBench | 核对完整题量与指标后全量运行 | 与 IFEval 互相验证，但协议不同，成本相对可控 |
| 4 | MMLU-Pro | 运行 12,032 题全量同题 A/B | 已有 ThinkingCap 280 题 pilot，补齐 Qwen 对照后才能形成可信结论 |
| 5 | LongBench v2 | 先修正输出预算策略并重跑 short/medium/long smoke，通过后再全量 | 长上下文提供不可替代的信息，但当前协议未通过，不允许带着 400 错误开全量 |
| 6 | C-Eval | 固定生成式多选口径后全量 | 在 MMLU-Pro 之后补充中文知识覆盖，边际信息高于再跑一个英文 MMLU 变体 |
| 7 | MMLU-Redux | 检查 57 个 config 和 revision 后全量 | 与 MMLU-Pro 重叠较大，作为知识结果复核放在核心队列末尾 |

`LongBench v2` 不通过以下 gate 时保持 `BLOCKED`，不阻塞其他任务：

- 在请求前计算实际 prompt tokens；
- 满足 `input_tokens + max_tokens <= 262144`，并保留安全余量；
- 输出预算调整必须按公开协议或统一规则进行，不能只为避错随意缩短；
- short、medium、long 各至少 1 题双模型成功；
- 结果明确区分完整协议、动态输出预算和不可复现样本，禁止静默截断。

每项执行流程：

```text
协议核对
  → 数据与题量预检
  → 两模型各 1–2 题 smoke
  → 核对请求参数是否被实际接受
  → 冻结配置
  → 两模型并行全量运行
  → 检查题量、漏题和错误
  → 使用原配置补跑失败项
  → 生成单项结果
  → 合入模型卡
```

初始 `eval_batch_size=4`。连续完成至少 50 个请求且无服务错误、解析错误或明显排队异常后，才可提高到 8。并发调整只改变调度，不改变生成参数和评分口径。

## 6. 题库数量记录标准

题量必须同时保存以下字段：

```text
official_total
train_count
validation_count
test_count
subset_count
subset_sizes
loaded_total
expected_to_evaluate
attempted
successful
request_failed
parse_failed
judge_failed
skipped
retried
```

必须区分“每 subset limit”和“整个 benchmark limit”。例如 MMLU-Pro 使用 `--limit 1` 时，14 个 subset 可能产生 14 个实际请求，不能在报告中写成只测试 1 题。

题量来源也必须记录：官方文档、adapter 元数据、数据加载日志或最终结果文件。官方题量和实际加载题量不一致时，先定位原因，再决定是否继续正式运行。

## 7. 评测标准记录

每个任务至少记录：

- task type；
- primary metric 和辅助指标；
- 分数单位、方向和计算公式；
- micro、macro 或加权聚合方式；
- subset 汇总方式；
- prompt、system prompt 和 chat template；
- few-shot 数量与示例来源；
- answer extraction 和格式要求；
- judge 模型、judge prompt、温度和失败处理；
- seed、repeats、pass@1 或 pass@k；
- sandbox、timeout、CPU、RAM、网络限制；
- Agent 工具 schema、max turns 和成功判定；
- 长上下文裁剪或 folding；
- 图片分辨率、max pixels、视频抽帧和字幕策略。

## 8. 剩余 75 项优先级与配置队列

剩余 75 项不以“全部跑完”作为第一目标。先在知识、推理、代码、工具调用、Agent、多语言和视觉各取得至少一个可信结果，再决定是否对同类 benchmark 加密。这样能更快形成覆盖均衡的模型卡，也能避免大量 GPU 时间消耗在高度相关的题集上。

| 执行级别 | 数量 | 定位 | 进入正式运行的条件 |
|---|---:|---|---|
| P1 | 10 | 文本高价值扩展 | 本地 adapter/harness 基本可用，协议能够冻结 |
| P2 | 13 | 中等基础设施文本任务 + 视觉最小套件 | 依赖、judge、sandbox 或视觉 API gate 已通过 |
| P3 | 12 | 已有 adapter 的视觉扩展 | P2 视觉最小套件至少 3 项形成双模型有效结果 |
| P4 | 40 | 专用 harness、外部系统或高成本环境 | 单独立项，先评估可复现性和资源预算 |

上述数量合计 75，不含第 5 节的 7 个 P0 核心任务。

### 8.1 P1：核心完成后的第一扩展批次（10 项）

按下列顺序配置和运行；前四项优先保证模型卡尽快覆盖数学、代码、工具和 Agent，而不是继续堆叠知识选择题：

1. `HMMT Nov 25`：三张候选模型卡共同出现，直接可比价值最高；先确认 `hmmt25` adapter 对应月份；
2. `LiveCodeBench v6`：固定 v6 数据版本，完成数据后使用代码 sandbox 跑 pass@1；
3. `BFCL-V4`：先跑不需要外部搜索凭据的 function-calling 子集；
4. `TAU2-Bench`：本地包和 602MB 数据已就绪，固定用户模拟模型、domain 和 airline 修复口径；
5. `HLE w/ CoT`：确认 text-only/CoT subset、关闭工具并冻结输出预算；
6. `SuperGPQA`：固定 72 subsets、聚合方式和答案抽取；
7. `HMMT Feb 25`：在 Nov 版本协议确认后复用数学评测链路；
8. `MMMLU`：固定语言集合及 macro average，补充多语言知识；
9. `PolyMATH`：固定 18 个语言/子集的汇总方式，补充多语言数学；
10. `AA-LCR`：固定 judge、judge prompt 和长上下文协议后运行。

P1 完成标准不是“命令能启动”，而是至少前 4 项形成双模型完整结果，且 10 项全部达到 `CONFIG_FROZEN` 或明确记录 `BLOCKED`。

### 8.2 P2：两条并行轨道（13 项）

#### P2-A：视觉最小套件（5 项）

先用 1–2 个公开样本验证两个端点都能接收相同的图片消息格式、图片尺寸和 chat template。这个 gate 通过后依次运行：

1. `MMMU-Pro`：综合视觉与学科推理；
2. `MathVision`：视觉数学；
3. `RealWorldQA`：真实图像通用问答；
4. `OCRBench`：文本识别与文档理解；
5. `TIR-Bench`：视觉工具调用，分别保存两项官方指标。

如果某个模型端点不支持视觉输入，应将对应任务记为 `N/A / unsupported modality`，不能记为 0 分或模型能力失败。

#### P2-B：中等基础设施文本任务（8 项）

按依赖成熟度排序：

1. `SWE-bench Verified`；
2. `Terminal Bench 2`；
3. `WMT24++`；
4. `Global PIQA`；
5. `MultiChallenge`；
6. `MMLU-ProX`；
7. `INCLUDE`；
8. `MAXIFE`。

其中 SWE-bench 必须从当前 50 题 mini 明确升级为完整 Verified；Terminal Bench 固定镜像、CPU、RAM、timeout 和网络；WMT24++ 只有在 XCOMET-XXL 与 55 语言平均口径可复现时才进入正式运行；Global PIQA 未确认版本等价前不能用普通 `piqa` 结果替代。

### 8.3 P3：已有 adapter 的视觉扩展（12 项）

P2-A 证明视觉链路稳定后，再按“通用视觉 → 文档/OCR → grounding/工具”扩展：

1. 通用与推理：`MMMU`、`Mathvista(mini)`、`ZEROBench`、`ZEROBench_sub`、`MMStar`、`MMBenchEN-DEV-v1.1`、`SimpleVQA`、`HallusionBench`；
2. 文档与图表：`OmniDocBench1.5`、`AI2D_TEST`；
3. grounding 与工具：`RefCOCO(avg)`、`V*`。

版本名带有 `mini`、`sub`、`1.5`、`DEV-v1.1` 的任务必须先证明本地 adapter 对应同一数据版本；无法证明时标记 `Approximate`，不能用近似版本填入主对比表。

### 8.4 P4：最后处理或单独立项（40 项）

这些任务不是没有价值，而是当前新增结论相对于接入成本较低，或依赖外部搜索、专用执行环境、大型媒体数据、许可数据和非标准评分器。当前只做协议研究、数据许可与阻塞盘点，不承诺立即正式跑分。

- 自建/未接入文本与代码：`CodeForces`、`OJBench`、`FullStackBench en`、`FullStackBench zh`、`VITA-Bench`、`DeepPlanning`、`NOVA-63`；
- 搜索 Agent：`HLE w/ tool`、`Browsecomp`、`Browsecomp-zh`、`WideSearch`、`Seal-0`；
- 未接入视觉推理与文档：`DynaMath`、`VlmsAreBlind`、`BabyVision`、`CharXiv(RQ)`、`MMLongBench-Doc`、`CC-OCR`；
- 空间与 3D：`ERQA`、`CountBench`、`ODInW13`、`EmbSpatialBench`、`RefSpatialBench`、`LingoQA`、`Hypersim`、`SUNRGBD`、`Nuscene`；
- 视频：`VideoMME(w sub.)`、`VideoMME(w/o sub.)`、`VideoMMMU`、`MLVU`、`MVBench`、`LVBench`、`MMVU`；
- GUI / Android Agent：`ScreenSpot Pro`、`OSWorld-Verified`、`AndroidWorld`；
- 医疗 VQA：`SLAKE`、`PMC-VQA`、`MedXpertQA-MM`。

P4 中的单项只有同时满足以下条件才升级：leader 明确需要该能力维度；官方或可审计 harness 可获得；数据许可允许；资源和运行成本有上限；失败不会影响生产服务或宿主机安全。

### 8.5 调度与止损规则

- 正式 GPU 运行始终优先 P0，再按 P1、P2、P3 推进；P4 不抢占 P0/P1 资源；
- 同一时间只正式运行一个 benchmark，但该 benchmark 的两个模型并行；配置研究、数据预检和小型 smoke 可以与正式运行并行；
- 每完成一个能力维度，先生成双模型差异和异常摘要，再决定是否继续同类题集；
- 单项若在半个工作日内仍无法冻结协议，降为 `BLOCKED` 并继续下一项，不能拖住整条队列；
- 预计成本超过 1 GPU-day、需要收费 judge/API、需要大型受许可数据或专用 VM 的任务，先单独给出成本估算再执行；
- 外部任务在没有明确授权时不自动申请凭据、不下载大型受许可数据、不启动 VM/Android 环境，也不把 unsupported 输入记成模型能力失败。

如果后续明确采用多 agent 配置，可拆成 Language、Vision Language、外部 harness 三个独立队列；各执行者只写自己的任务目录，共享 manifest 仍由单一负责人审核合并。

## 9. 单项任务目录

每个 benchmark 使用独立目录：

```text
benchmarks/modelcard/tasks/<order>-<benchmark>/
├── protocol.yaml
├── qwen.yaml
├── thinkingcap.yaml
├── README.md
└── status.json
```

其中 `protocol.yaml` 记录官方口径，两个模型配置文件记录实际请求差异，`status.json` 记录任务当前阶段、owner、阻塞和证据路径。

建议状态机：

```text
RESEARCH
  → CONFIGURED
  → PREFLIGHT
  → SMOKE_QWEN
  → SMOKE_THINKINGCAP
  → CONFIG_FROZEN
  → FULL_RUN
  → VERIFIED
  → REPORTED
```

## 10. 运行输出和过程证据

每次正式运行使用唯一 `run_id`：

```text
logs/eval/modelcard/<run-id>/<benchmark>/
├── frozen_config/
├── commands.log
├── environment.json
├── sample_counts.json
├── issue_log.tsv
├── qwen/
│   ├── predictions/
│   ├── reports/
│   └── runner.log
└── thinkingcap/
    ├── predictions/
    ├── reports/
    └── runner.log
```

`issue_log.tsv` 至少记录：

- benchmark；
- 模型；
- 发生阶段；
- 原始错误；
- 初步原因；
- 修复动作；
- 重试结果；
- 是否影响成绩可比性；
- owner；
- 证据路径。

## 11. 完成验收标准

一个 benchmark 只有同时满足以下条件才可标为完成：

- 官方协议来源已记录；
- dataset revision、split 和 subset 已冻结；
- 官方题量和本地实际题量明确；
- 两个模型的请求配置已冻结；
- 请求参数被实际端点接受；
- 两模型完成预期样本；
- 请求、解析、judge 和 sandbox 失败数量明确；
- 没有静默漏题；
- 分数能够从原始预测重新计算；
- 与 122B 官方成绩的可比性等级明确；
- 配置、命令、日志、预测和结果均可追溯。

仅完成 dry-run、数据下载或 API 健康检查不算跑通。

## 12. Leader 汇报结构

完成第一阶段后生成第一版 HTML 模型卡报告，包含：

1. Executive Summary；
2. 模型、服务和评测环境版本；
3. 82 项总体覆盖状态；
4. 7 项核心结果与 122B 官方分数对比；
5. 各项题库数量和评测标准；
6. 实际请求配置和配置来源；
7. 执行成功率、漏题、重试和异常；
8. 协议一致性和可比性限制；
9. 剩余 75 项配置进度；
10. 结论和下一步建议。

主结果表至少包含：

| Benchmark | 官方题量 | 实际评分题量 | 主指标 | 122B 官方 | Qwen | ThinkingCap | 协议等级 | 异常数 |
|---|---:|---:|---|---:|---:|---:|---|---:|

## 13. 时间预估

- P0 的 7 项正式运行：约 1–3 天；MMLU 系列和 LongBench v2 可能需要过夜，LongBench 协议修复时间不计入纯运行时间；
- P1 的 10 项：配置冻结约 2–4 天，完整运行约 3–7 天；代码和 Agent sandbox 的失败重试可能扩大工期；
- P2 的 13 项：视觉 gate 与最小套件约 2–4 天，中等基础设施文本任务约 3–7 天；
- P3 的 12 项：复用稳定视觉链路后约 1–2 周；
- P4 的 40 项不纳入当前承诺工期，搜索、视频、3D、GUI、Android 和医疗环境需分别立项和估算。

以上是队列级粗估，不把所有任务简单相加成截止日期。每完成 P0、P1 或 P2 的一个阶段，应根据实际平均输出长度、失败率和 judge/sandbox 成本重新估算下一阶段。

CodeForces 自有 query set、动态搜索结果和部分非公开测试答案可能无法与 122B 官方成绩严格同题复现。这类项目必须标为 `Approximate` 或 `Not reproducible`，不能用近似结果冒充严格对比。

## 14. 安全边界

- 不重启 H100 宿主机；
- 不删除、移动或重写 `models/` 下的权重；
- 不在配置、日志或报告中保存真实 token、SSH key 或搜索凭据；
- 两个模型服务只作为被测端点，评测脚本不得擅自停止或重启容器；
- GUI、Android、搜索、代码执行和受许可数据应使用独立 worker，不直接赋予模型宿主机不受控权限；
- 始终区分本地观察、远端服务证据和官方公开成绩。
