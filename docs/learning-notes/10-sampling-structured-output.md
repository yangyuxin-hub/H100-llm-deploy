# 采样策略与结构化输出

## 一句话理解

采样是从模型输出的概率分布里选下一个 token 的过程，top-k/top-p/temperature 是控制采样随机性的参数；思考模型用高温度保证探索性，编码用低温度保证确定性；结构化输出（grammar/JSON/regex 约束）在采样阶段强制模型输出符合特定格式，是 Function Calling 和工具调用的底层基础。

---

## 一、采样的数学基础

### 1. 模型输出的是概率分布

模型最后输出 logits（每个 token 的原始分数），经 softmax 变成概率：

```text
logits = [2.1, 0.5, -1.3, 0.8, ...]   # vocab_size 维
probs = softmax(logits) = [0.45, 0.08, 0.01, 0.12, ...]
```

下一个 token 从这个分布里选。选法不同，生成行为不同。

### 2. 贪心解码（Greedy）

永远选概率最高的 token：

```text
next_token = argmax(probs)
```

- 优点：确定性，同样输入永远同样输出。
- 缺点：容易重复、无创造力、数学任务常用。

### 3. 随机采样

按概率随机选：

```text
next_token = sample(probs)
```

- 优点：有多样性。
- 缺点：可能选到概率很低的 token，输出胡言乱语。

所以实际用**带截断的采样**：top-k、top-p、min-p。

---

## 二、采样参数详解

### 1. temperature（温度）

用温度缩放 logits 再 softmax：

```text
probs = softmax(logits / temperature)
```

| temperature | 效果 |
|---|---|
| → 0 | 趋近贪心，永远选最高概率 |
| < 1 | 分布变尖锐，更确定 |
| = 1 | 原始分布 |
| > 1 | 分布变平坦，更随机 |

```text
temperature=0.1:  [0.95, 0.04, 0.01]  → 几乎必选第一个
temperature=1.0:  [0.45, 0.30, 0.25]  → 较均匀
temperature=2.0:  [0.40, 0.35, 0.25]  → 更均匀，选哪个都有可能
```

### 2. top-k

只保留概率最高的 k 个 token，其余设为 0 再归一化：

```text
原始 probs: [0.45, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02]
top-k=3:    [0.45, 0.20, 0.15, 0,    0,     0,     0] → 归一化 → [0.56, 0.25, 0.19]
```

- k 越小越保守，k 越大越多样。
- k=1 等价于贪心。
- 当前项目用 top_k=20。

### 3. top-p（nucleus sampling）

保留概率累加到 p 的最小 token 集合：

```text
原始 probs 排序: [0.45, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02]
top-p=0.9: 累加 0.45+0.20+0.15+0.10=0.90 → 保留前 4 个
```

- 自适应：分布尖锐时保留少，分布平坦时保留多。
- 比 top-k 更智能。
- 当前项目用 top_p=0.95。

### 4. min-p

保留概率 ≥ min_p × max_prob 的 token：

```text
max_prob = 0.45
min-p=0.05: 阈值 = 0.45 × 0.05 = 0.0225
保留 probs ≥ 0.0225 的 token
```

- 比 top-p 更简单，且对长尾更友好。
- 新方法，部分模型/框架支持。

### 5. presence_penalty / frequency_penalty

调整已出现 token 的概率：

```text
presence_penalty: token 出现过就惩罚（鼓励新话题）
frequency_penalty: 按 token 出现次数惩罚（减少重复）
```

当前项目 Agents-A1 用 `presence_penalty=1.1`，减少重复输出。

---

## 三、思考模型的采样参数选择

### 1. 为什么思考模型用高温度

思考模型（thinking model）的 reasoning 过程是**探索性推理**：

- 需要尝试不同思路，走错路再回退。
- 低温度会让模型"一条路走到黑"，容易卡在错误推理上。
- 高温度（1.0）保证探索性，允许模型试错。

当前项目 Qwen3.6 官方推荐：

| 模式 | temperature | top_p | top_k | 适用 |
|---|---|---|---|---|
| 思考（通用） | 1.0 | 0.95 | 20 | 日常对话、推理 |
| 思考（编码） | 0.6 | 0.95 | 20 | 代码生成 |
| 非思考 | 0.7 | 0.8 | 20 | 快速回答 |

### 2. 编码为什么用 0.6 而不是 1.0

编码任务的特点：

- 语法必须正确（确定性高）。
- 逻辑要精确（少试错）。
- 但仍需一定多样性（不同实现方式）。

0.6 是折中：比贪心（0）有探索性，比 1.0 更确定。

### 3. 非思考模式为什么用 0.7 + presence_penalty

非思考模式直接输出答案，不经过 reasoning：

- 0.7：适度随机，避免太死板。
- presence_penalty=1.5：防止直接回答时重复（无 reasoning 缓冲，更容易重复）。

---

## 四、思考模型的输出结构

### 1. reasoning 与 content 分离

Qwen3.6 思考模式的输出：

```json
{
  "reasoning": "让我想想...1+1就是两个东西合在一起...",
  "content": "1+1=2"
}
```

- `reasoning`：思考过程（可能几百~几千 token）。
- `content`：最终答案。

### 2. max_tokens 要给够

思考过程消耗大量 token，如果 max_tokens 太小：

```text
max_tokens=300:
  reasoning 用掉 280 token
  content 只剩 20 token，答案被截断
```

所以思考模型 max_tokens 建议：

- 简单任务：≥1000
- 复杂任务：81920（当前项目配置）

### 3. 关闭思考的方法

Qwen3.6 不支持 `/no_think`（那是 Qwen3 的语法）。正确方式：

```python
resp = client.chat.completions.create(
    model="qwen3.6-27b-fp8",
    messages=[...],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)
```

这是通过 chat template 的参数控制，不是 API 层面的参数。

---

## 五、结构化输出（Constrained Decoding）

### 1. 为什么需要结构化输出

很多时候要求模型输出特定格式：

- Function Calling：输出 JSON 格式的工具调用。
- 数据提取：输出符合 JSON Schema 的结构化数据。
- 代码生成：输出特定语法的代码。

普通采样不保证格式正确，模型可能输出：

```text
"我来调用工具：{"name": "search", "args": {...}}"  ← 夹杂了说明文字
```

结构化输出在采样阶段**强制**模型只能输出符合格式的 token。

### 2. 约束解码的原理

在每个 token 采样时，根据当前已生成的内容和目标格式，计算"哪些 token 是合法的下一个 token"，把不合法的 token 概率设为 -inf：

```text
已生成: {"name": "search", "args":
合法下一个 token: " (引号开始)
非法下一个 token: 数字、字母、大部分符号

probs["\""] = 0.9 (保留)
probs["a"] = -inf (禁止)
probs["1"] = -inf (禁止)
```

### 3. 约束类型

| 约束 | 支持的格式 | 适用 |
|---|---|---|
| JSON Schema | 符合 JSON Schema 的 JSON | Function Calling、数据提取 |
| Regex | 正则表达式 | 特定格式（电话、邮箱） |
| Grammar | CFG 语法 | 编程语言、复杂结构 |
| Choice | 枚举值 | 分类任务 |

### 4. vLLM 的 guided_decoding

vLLM 支持结构化输出，通过 `guided_decoding` 后端：

```python
resp = client.chat.completions.create(
    model="qwen3.6-27b-fp8",
    messages=[...],
    extra_body={
        "guided_json": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "args": {"type": "object"}
            }
        }
    }
)
```

后端选项：

- `outlines`：基于 outlines 库，支持 JSON/regex/grammar。
- `lm-format-enforcer`：另一个库，类似功能。
- `xgrammar`：性能更好的 grammar 后端。

---

## 六、Function Calling 的底层机制

### 1. Function Calling 是什么

让模型调用外部工具：

```text
用户: "查一下北京天气"
模型: 调用 search_weather(city="北京")
工具返回: {"temp": 25, "weather": "晴"}
模型: "北京今天 25 度，晴天"
```

模型不是真的"执行"工具，而是输出一个结构化的工具调用请求，由外部程序执行。

### 2. Function Calling 的流程

```text
1. 客户端发送请求，附带可用工具列表
2. 模型生成 tool_calls（结构化 JSON）
3. vLLM 解析 tool_calls（用 tool-call-parser）
4. 客户端执行工具，拿到结果
5. 客户端把结果作为新消息发回模型
6. 模型基于结果生成最终回答
```

### 3. tool-call-parser 的作用

模型输出的 tool_calls 是文本，不同模型格式不同：

```text
Qwen3.6:    <tool_call>{"name": "search", "args": {...}}</tool_call>
Hermes:     <tool_call>...</tool_call>
Llama:      [TOOL_CALLS] ...
```

parser 把这些文本解析成结构化的 `tool_calls` 字段。选错 parser 会导致：

- 模型输出了工具调用文本，但 parser 不认识，当成普通文本返回。
- 客户端收不到 `tool_calls` 字段，无法执行工具。

当前项目踩坑：

| Parser | Agents-A1 | 结果 |
|---|---|---|
| hermes | ❌ | 不解析，输出文本 |
| qwen3_xml | ⚠️ | 能工作但非官方 |
| qwen3_coder | ✅ | 官方推荐 |

### 4. reasoning-parser 的作用

思考模型输出有 reasoning 和 content 两部分，reasoning-parser 把它们分开：

```text
模型输出: <think>思考过程...</think>正式回答
reasoning-parser qwen3: 
  reasoning = "思考过程"
  content = "正式回答"
```

选错 parser 会导致 reasoning 和 content 混在一起，客户端无法区分。

当前项目两个模型都用 `--reasoning-parser qwen3`。

---

## 七、采样参数对性能的影响

### 1. 采样开销

采样本身有计算开销：

- softmax：O(vocab_size)，vocab 通常 10万+。
- top-k 排序：O(vocab_size × log k)。
- top-p 累加：O(k)。

对 decode 每步都要做，大 vocab 下不可忽略。

### 2. 结构化输出的开销

约束解码要维护状态机（跟踪当前格式位置），每步：

1. 更新状态机。
2. 计算合法 token 集合。
3. 屏蔽非法 token。

复杂 grammar 的状态机可能很大，开销显著。vLLM 用 xgrammar 等优化库降低开销。

### 3. MTP 与采样的交互

MTP 一次预测多个 token，每个都要采样 + 验证：

- draft 模型生成 K 个候选 token（各自采样）。
- 主模型验证 K 个 token 是否符合分布。
- 验证时要做 K 次"接受/拒绝"判断。

采样参数影响 MTP 接受率：

- 低温度：draft 和主模型容易一致，接受率高。
- 高温度：draft 可能选不同 token，接受率低。

当前项目 benchmark 显示 MTP 接受率 79-87%，温度 0.7 下属正常范围。

---

## 八、放到当前项目里看

### 1. 采样参数配置

当前项目两个模型的采样参数（官方推荐）：

| 模型 | 场景 | temperature | top_p | top_k | presence_penalty |
|---|---|---|---|---|---|
| Qwen3.6（思考） | 通用 | 1.0 | 0.95 | 20 | 0.0 |
| Qwen3.6（思考） | 编码 | 0.6 | 0.95 | 20 | 0.0 |
| Qwen3.6（非思考） | 快速 | 0.7 | 0.8 | 20 | 1.5 |
| Agents-A1 | 通用 | 0.85 | 0.95 | 20 | 1.1 |

### 2. opencode 场景的参数选择

opencode 是编码工具，主要场景：

- 代码生成：用 Qwen3.6 思考模式编码参数（temp=0.6）。
- 快速补全：用 Agents-A1（temp=0.85，探索性强）。
- 工具调用：Function Calling，用 qwen3_coder parser。

### 3. Function Calling 配置

两个模型都启用 Function Calling：

```bash
--enable-auto-tool-choice          # 开启工具调用
--tool-call-parser qwen3_coder     # 解析格式
--reasoning-parser qwen3           # 思考链解析
```

opencode 通过这些接口实现文件读写、命令执行等工具调用。

### 4. 结构化输出的潜在应用

当前项目没显式用 `guided_decoding`，但 Function Calling 底层就是结构化输出。未来可扩展：

- 强制 JSON 输出（数据提取场景）。
- 强制特定代码语法（DSL 生成）。
- 枚举分类（情感分析等）。

---

## 总结

| 概念 | 作用 | 项目里的体现 |
|---|---|---|
| temperature | 控制采样随机性 | 思考用 1.0，编码用 0.6 |
| top-k | 截断到 k 个候选 | 20 |
| top-p | 截断到概率累加 p | 0.95 |
| presence_penalty | 减少重复 | Agents-A1 用 1.1 |
| 思考模型采样 | 高温度探索 | Qwen3.6 reasoning |
| 结构化输出 | 强制格式 | Function Calling 底层 |
| tool-call-parser | 解析工具调用文本 | qwen3_coder |
| reasoning-parser | 分离 reasoning/content | qwen3 |
| guided_decoding | JSON/regex/grammar 约束 | 未显式用，可扩展 |
