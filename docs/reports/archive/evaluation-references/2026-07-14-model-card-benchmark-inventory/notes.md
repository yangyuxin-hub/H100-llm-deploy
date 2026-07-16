# 三张模型卡评测集清单：来源与整理说明

## 报告任务

- 问题：整理 GLM-5.2、ThinkingCap-Qwen3.6-27B、Qwen3.5-122B-A10B 模型卡公开的评测集。
- 受众：需要设计本地模型能力评测的技术人员。
- 截止时间：2026-07-14。
- 主来源：三张 Hugging Face 模型卡；不从第三方排行榜补全模型卡未披露的协议。

## 来源

1. https://huggingface.co/zai-org/GLM-5.2
2. https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B
3. https://huggingface.co/Qwen/Qwen3.5-122B-A10B

## 整理口径

- “评测条目数”按模型卡表格中的结果行计数，不等于唯一数据集数量。
- 同一数据集使用不同 harness、版本或模式时保留为不同条目，例如 Terminal-Bench 2.1 的两种 harness、HLE 与 HLE w/ Tools。
- ThinkingCap 的 `in-domain` 指训练混合中所含数据集训练集的留出测试集；不能与真正域外泛化结果混为一类。
- Qwen3.5 模型卡同时包含语言与视觉语言两张大表；双分数按模型卡原样保存。
- 分数列只用于定位模型卡结果，不做跨任务排序，因为各行指标、样本、裁判和 harness 不同。

## 可视化说明

- 报告只使用一张“模型卡评测条目数”柱状图说明覆盖广度。
- 不绘制模型能力排名图：三张卡只有少量同名测试集，且公开协议不完全一致，直接比较分数会误导。
- 详细内容使用表格，因为该任务的主要需求是精确查找 benchmark 名称、类别和协议。

## 技术报告结构映射

- Technical summary：报告开头的“技术摘要”。
- Key findings with visual evidence：覆盖广度柱状图与交集表。
- Scope, data, and metric definitions：三张模型卡详细清单及“口径与方法”。
- Methodology：模型卡逐行抄录、规范化命名、同名交集匹配。
- Limitations, uncertainty, and robustness checks：“可比性限制”。
- Recommended next steps：“建议的本地评测分层”。
- Further questions：报告末尾列出仍需固定的数据 revision、prompt、seed、judge 与 harness。
