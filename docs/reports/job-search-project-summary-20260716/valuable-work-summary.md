# 大模型部署与评测：有价值工作记录

更新时间：2026-07-16

## 一句话总结

在受限、离线的 4×H100 80GB 环境中，我把两个 27B FP8 模型从部署、推理优化、工具协议兼容、自动化评测做到可复现交付，并能够跨模型输出、reasoning/tool parser、评测 adapter 和 scorer 定位复杂故障。

这段经历的核心价值不是“跑了几个模型和榜单”，而是完成了一条完整的 LLM Systems 工作链：

> 识别约束 → 部署服务 → 验证协议 → 测量性能 → 冻结评测 → 审计结果 → 修复失败 → 恢复长任务 → 给出有边界的结论。

## 一、模型部署与推理服务

### 做成的事情

- 在同一台 4×H100 80GB 节点上并行运行两个 27B FP8 模型：
  - Qwen3.6-27B-FP8：GPU 0、1，TP2，稳定服务。
  - ThinkingCap-Qwen3.6-27B-FP8：GPU 2、3，TP2，实验服务。
- 使用 Docker 和 vLLM 0.24.0 固化运行环境，提供 OpenAI-compatible API。
- 配置并验证 FP8 KV cache、FLASH_ATTN、MTP3、262,144-token 最大上下文、GPU/NUMA 绑定、reasoning parser 和 tool parser。
- 在不影响稳定服务的情况下完成 ThinkingCap 容器切换、冷启动、健康检查、真实生成、工具调用和 opencode 接入验证。
- 记录容量边界：`max_num_batched_tokens=131072` 会 OOM，最终使用 65,536 作为安全配置。

### 量化结果

- 同配置并发 16 稳态复测：
  - Qwen：2,342.2 output tok/s。
  - ThinkingCap：2,354.3 output tok/s。
  - 差异 0.52%，可视为吞吐持平。
- 两个模型完成 8K–140K 长上下文扫描，未出现请求失败。
- 通过 MTP 与采样参数消融，区分了配置对齐收益和采样温度对 MTP 接受率的影响。

### 体现的能力

- GPU 资源规划与服务隔离。
- vLLM 推理参数、KV cache、MTP、并行与上下文容量理解。
- 以对照实验验证性能，而不是只凭单次 benchmark 判断。
- 在稳定服务不能中断、宿主机不能重启、节点无公网的约束下交付。

## 二、评测平台与实验方法

### 做成的事情

- 用 `uv` 建立隔离的 EvalScope 1.6.1 / Python 3.10.20 环境，处理节点无公网、依赖版本冲突和本地数据加载问题。
- 建成两套自动化评测主线：
  - 七项核心评测，覆盖知识、数学、推理、指令遵循、代码、长上下文和事实性。
  - 十项 P1 ranked 评测，覆盖 BFCL-V4、TAU2-Bench、LiveCodeBench v6、HMMT、PolyMATH、SuperGPQA、MMMLU 等。
- runner 支持 preflight、dry-run、smoke、full、后台运行、固定 run ID、断点续跑、单 benchmark、模型子集、全局锁、旧进程保护、append-only 状态和去重状态汇总。
- 为代码 benchmark 验证 Docker sandbox；为 WMT 单独隔离依赖环境；为无公网节点准备本地数据、模型和 embedding 依赖。
- 整理 122 条模型—benchmark 记录，识别同名分数因 prompt、seed、harness、judge、输出预算不同而不可直接横比。

### 量化结果

- 七项核心正式评测：14 个模型—任务状态全部成功。
- P1 双模型真实 smoke：20/20 SUCCESS。
- 七项核心结果中 ThinkingCap 五项领先、一项持平、一项落后：
  - MMLU-Redux：93.19% vs 93.19%。
  - C-Eval：Qwen 91.09%，ThinkingCap 90.79%。
  - LongBench v2：在上下文预算内的 400/503 题上，Qwen 61.75%，ThinkingCap 63.25%。
- LongBench 的 103 条超长样本被明确排除，没有静默截断，也没有把 400 条结果包装成完整 503 条结果。

### 体现的能力

- 把一次性脚本变成可重复、可恢复、可审计的评测系统。
- 理解 benchmark 协议、分母、生成参数、judge 和数据版本对可比性的影响。
- 能在大量环境问题中区分“模型失败”“评测框架失败”和“依赖/数据失败”。

## 三、结果复核，而不只是报告分数

### IFEval 案例

- 正式 Prompt strict：ThinkingCap 87.80%，Qwen 84.84%，表面差距为 +2.96 个百分点。
- 进一步检查发现，至少一方达到 8,192-token 上限的样本显著影响差距。
- 剔除这 39 条配对样本后，差距缩小为 +1.20 个百分点，paired bootstrap 95% 区间跨过 0。
- 因此最终结论不是简单的“ThinkingCap 指令遵循能力高 2.96pp”，而是：ThinkingCap 正式成绩和输出长度控制更好，但全量差距不能全部归因于非截断样本上的基础能力。

### 体现的能力

- 主动检查截断、遗漏样本和统计不确定性。
- 能识别看似漂亮但可能有混杂因素的结果。
- 用审慎结论替代过度宣传，保证评测可信度。

## 四、TAU2-Bench 故障定位与修复

### 失败现象

TAU2 不是因为网络断开而失败，而是被测模型返回了无法构成合法 assistant message 的结果：

- Qwen 有时生成了 reasoning token，但经过 reasoning parser 后，`content` 和 `tool_calls` 都为空。
- ThinkingCap 有时耗尽 81,920 output tokens，最终仍没有合法内容或工具调用。
- EvalScope/TAU2 随后触发 `AssistantMessage must have either content or tool calls`，导致整套任务 fail-fast。

### 修复思路

- 增加 raw response audit，保留原始 finish reason、token 使用量和解析后的输出，便于区分 parser 空响应与 max-token 空响应。
- 增加专门的 `EmptyAgentResponseError`。
- 被测 agent 返回无效空响应时，将该样本记 0 分并继续，而不是：
  - 伪造一条 assistant 文本；
  - 自动重试直到成功；
  - 从分母中删除失败样本。
- 这样保留 pass@1 的真实失败和完整分母。
- 将 TAU2 单轮输出预算从 81,920 调整为 8,192，避免单条异常轨迹长期占用服务。

### 实验验证

- 对 Qwen 关闭 thinking 后，50 次 smoke 出现文本形式的伪工具调用，得分为 0，因此没有把“能跑完”误判为配置改善，最终恢复 thinking。
- ThinkingCap 非 thinking 配置产生原生 tool call，smoke 得分 1.0。
- Qwen 恢复 thinking 后定向运行 2 题，得分 0.5，并产生 15 次原生工具调用。
- TAU2 正式任务按 airline、retail、telecom 实际共 269 题启动。

### 长任务运维

- 因 P1 full 全量运行耗时很长，使用进程组 `SIGSTOP` 安全暂停原任务，没有 kill，也没有破坏断点状态。
- 单独启动 `tau2-priority-full-20260716`。
- 设置 watcher，在 TAU2 完成后自动对原进程组发送 `SIGCONT`。
- 截至本文档更新时间，TAU2 正式任务仍在运行，因此不把过程分数写成最终成绩。

### 体现的能力

- 跨模型行为、推理解析、OpenAI 消息协议、评测 adapter 和 scorer 进行根因分析。
- 在“让任务继续”和“保持评测语义正确”之间做正确取舍。
- 对长任务进行安全暂停、优先级调整、恢复和审计。

## 五、这段经历对求职的帮助

### 最匹配的岗位

1. **LLM 推理/部署工程师：强匹配**
   - 证据：H100、vLLM、TP、FP8、KV cache、MTP、NUMA、吞吐、长上下文和 OOM 边界。
   - 需要补强：线上监控、SLO、容量规划和成本模型。

2. **模型评测/Benchmark 工程师：强匹配**
   - 证据：统一协议、runner、断点恢复、raw audit、分母与截断审计、统计复核。
   - 需要补强：独立 judge、统计功效和更多公开可复现实验。

3. **ML Systems / AI Infra：较强匹配**
   - 证据：容器、GPU 隔离、离线依赖、自动化、锁、进程保护和长任务恢复。
   - 需要补强：Kubernetes/调度、Prometheus/Grafana、告警和多租户服务。

4. **Agent / Tool-use 工程师：较强匹配**
   - 证据：BFCL、TAU2、tool parser、chat template、轨迹与 raw response 调试。
   - 需要补强：真实 Agent 产品、记忆/状态管理和线上任务成功率。

### 暂时不占优势的岗位

- 纯模型训练/算法研究：当前缺少预训练、微调、数据配方、训练稳定性和论文型消融。
- 通用后端开发：有系统自动化能力，但还需要用 API 设计、数据库、消息队列和服务治理证明通用后端深度。

## 六、简历可用版本

项目名称建议：**双 27B 大模型推理部署与自动化评测平台**

- 在离线 4×H100 80GB 节点上并行部署 Qwen3.6-27B-FP8 与 ThinkingCap-Qwen3.6-27B-FP8，基于 vLLM、Docker、TP2、FP8 KV cache、FLASH_ATTN 与 MTP3 完成资源隔离、262K 上下文配置和 OpenAI-compatible API 验证。
- 构建双模型自动化评测流水线，覆盖七项核心任务与十项 P1 ranked 任务，支持 preflight/smoke/full、断点续跑、全局锁、原始响应审计及独立状态汇总；完成核心评测 14 个模型—任务状态和 P1 真实 smoke 20/20。
- 完成并发和长上下文性能验证：并发 16 时两模型输出吞吐约 2.34K tok/s、差异 0.52%，8K–140K 上下文扫描无失败；通过 MTP/采样消融定位接受率变化来源。
- 建立评测结果复核方法，显式处理上下文超限与输出截断；在 IFEval 中将表面 +2.96pp 优势拆解为截断敏感性问题，避免给出过度结论。
- 定位 TAU2-Bench reasoning parser/输出耗尽导致的空 assistant 崩溃，增加 raw audit 与逐样本失败隔离，在不重写响应、不重试的前提下保留 pass@1 分母，并完成优先 full-run 编排。

实际简历保留其中 3–4 条：推理岗保留部署、性能、TAU2；评测岗保留自动化、结果复核、TAU2。

## 七、面试最值得讲的三个故事

### 1. 双模型隔离部署

约束是只有 4 张 H100、稳定服务不能受影响、宿主机不能重启且节点无公网。重点讲如何按 GPU/NUMA 切分、如何固定容器和依赖、如何验证 parser/MTP/长上下文，以及如何用实验确认安全容量。

### 2. 为什么评测不能只看排行榜

用 LongBench 的 400/503 分母和 IFEval 的 max-token 敏感性说明：同名分数不等于同协议，完整性和混杂因素比一张排行榜更重要。

### 3. TAU2 为什么失败、为什么这样修

重点不是贴 traceback，而是说明如何从 suite 级崩溃追到两类空响应；为什么重试、伪造响应或删除样本会破坏 pass@1；最后如何逐样本隔离并安全调整长任务优先级。

## 八、下一步最有求职价值的工作

1. 等 TAU2 与 P1 full 完成，补正式成绩、失败分类和修复前后对比。
2. 做一页脱敏项目 README：架构图、约束、关键数字、三次关键故障、复现命令。
3. 录制 3–5 分钟演示：双服务状态、真实请求、断点续跑和 raw audit。
4. 补 Prometheus/Grafana 小闭环，记录 TTFT、TPOT、吞吐、GPU、错误率与 SLO。
5. 分别准备推理部署版和模型评测版简历，避免用一套 bullet 投所有岗位。

## 九、结论边界

- 目前可描述为“面向生产约束”或“production-minded”，不应直接声称已经建设多租户生产平台。
- HLE、AA-LCR 缺独立 judge；TAU2 full 尚未结束；ThinkingCap 的 MMLU-Pro 84.29% pilot 缺完整 Qwen 对照，这些都不能包装成完整 A/B 结论。
- 对外材料必须脱敏内网 IP、服务器路径和不可公开模型资产。

## 证据入口

- 项目总记录：`PROJECT_LOG.md`
- 部署与性能：`docs/project-log/2026-07-13.md`
- 核心评测：`docs/project-log/2026-07-14.md`
- P1 与 Agent 评测：`docs/project-log/2026-07-15.md`
- 推理配置：`config/serving.env`
- P1 runner：`benchmarks/modelcard/README.md`
- TAU2 修复：`benchmarks/modelcard/tau2_runtime_patch.py`
