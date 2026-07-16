# 附录 A:BFCL/TAU2 生成参数设计依据

本附录说明 P1 中 BFCL-V4 和 TAU2-Bench 的生成参数为什么这样设。参数本身见 [04 评测协议](04-evaluation-protocol.md),本页只讲原理与取舍,不是操作步骤。

## 总原则

生成参数按 benchmark 和模型分别冻结。BFCL 两模型都使用 non-thinking Function Calling 配置;TAU2 的 Qwen 使用 thinking,ThinkingCap 使用较低随机性的 Function Calling 采样并关闭 thinking。两套 harness 都不传 `preserve_thinking`,不强制 `tool_choice`,也不做正文转工具调用的后处理。

## Function Calling 不设 preserve_thinking

Function Calling 不设 `preserve_thinking=true`:BFCL 上游对 FC 模型省略历史 reasoning,实测强制回灌后 ThinkingCap 会把 JSON、`<core_memory_replace>` 或残缺 `<tool_call>` 写进普通 `content`,反而损害原生 OpenAI tool protocol。

## max_tokens=81920 的依据

评测端不再注入过小的 4,096-token 上限,而是显式使用 Qwen 官方 Chat Completions 示例的 `max_tokens=81920`;它保留模型完整输出预算,同时避免循环响应一直扩展到整个 262,144-token 上下文窗口。

81,920 是模型卡预算,不是 BFCL 官方硬规则。如果样本达到该上限,必须在审计中标记为 `finish_reason=length`,不能当成正常完成。不能使用"完全不设预算"代替合理高预算:vLLM 0.24 的服务端 `override_generation_config` 只识别 `max_new_tokens` 作为缺省上限,现有 `max_tokens` 字段并未成为缺省值;请求不传上限时,循环响应会持续到剩余上下文长度或 HTTP timeout。

## TAU2 用户模拟器固定参数

TAU2 的用户模拟器固定复用稳定 Qwen 端点,使用 `temperature=0, enable_thinking=false`,且不传 `max_tokens`。用户模拟器只生成简短对话,关闭 thinking 可避免只产生 reasoning 而用户正文为空;不传输出上限则避免在共享交互环境里再注入 4K 人为预算。被测 Qwen 保持 thinking;ThinkingCap 因相同的 reasoning-only 空响应证据关闭 thinking,但两者都保留 81K 输出预算。不能为了某一被测模型单独调整 user simulator。

## ThinkingCap 关闭 thinking 的依据(TAU2)

审计既多次捕获到 thinking-only 空响应,也捕获到原生 `temperature=1.0` 单轮持续生成;因此 ThinkingCap 在 TAU2 上使用 `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=0` 并关闭 thinking。`presence_penalty=0` 避免惩罚从上下文复制工具必填 ID。

## BFCL 原始响应审计

BFCL 通过 `benchmarks/modelcard/run_evalscope_with_bfcl_patch.py` 安装兼容补丁,原始审计位于每个模型任务目录下的 `audit/openai_responses.jsonl`。每条至少保存 `finish_reason`、`content`、`reasoning_content`、`tool_calls`、请求参数和用量;新正式轮次应记录 `temperature=0.7`、`max_tokens_source=request` 且 `max_tokens=81920`。补丁不强制 `tool_choice=required`,也不把普通正文改写为工具调用。

`Failed to decode the model response. Proceed to next turn.` 是 BFCL 上游将当前响应判定为非函数调用时的通用日志;在 Agentic 任务中,最后一条自然语言回答本来就是评分对象。因此不能用该日志行数代替原始响应审计,也不能等同于"ThinkingCap 没有调工具"。

## ThinkingCap 服务端模板修复

ThinkingCap 的服务端 `thinkingcap_agent.jinja` 必须在 system 末尾重申原生 Qwen tool protocol。除了禁止 `<invoke>` 和 JSON code block,还必须禁止实测出现的 `<tool_code>`、Python 函数写法和其他伪 XML,并明确工具调用不能与用户正文混在同一轮。该修复仅约束"已决定调工具"时的输出格式,不强制每轮调用。

修改模板或重启容器后必须先重跑 TAU2 smoke,不能直接沿用修复前的失败/成功状态。
