# 大模型推理基础

## 一句话理解

大语言模型推理就是：把输入文本切成 token，然后模型一次预测下一个 token，预测出来后再继续预测下一个，直到生成完整回答。

YOLO 这类视觉模型通常是一次前向推理输出检测框；大语言模型是循环生成 token。

## 推理流程

### 1. 文本变成 token

用户输入：

```text
介绍一下 H100
```

模型不能直接处理文字，需要 tokenizer 把文本切成 token，再转成 token id。

可以简单理解为：

```text
文本 -> token -> token id
```

真实切分结果由模型自己的 tokenizer 决定，不一定等于人类看到的词语。

### 2. token 变成向量

token id 会经过 embedding 层，变成模型能计算的高维向量。

可以理解为：

```text
token id -> embedding 向量
```

这些向量保存了词语、上下文和语义之间的关系。

### 3. 向量经过多层模型计算

以当前项目里的 Qwen3.6-27B-FP8 为例，语言模型部分有 64 层。

推理时，向量会一层一层往前传：

```text
embedding
-> layer 0
-> layer 1
-> ...
-> layer 63
-> lm_head
```

每一层都在做两类事情：

1. 从上下文里找重要信息。
2. 加工当前 token 的表示。

传统 Transformer 里主要是 attention 和 MLP。Qwen3.6 的结构更复杂，还包含 Gated DeltaNet 等模块。初学时可以先理解成“多层上下文理解和信息加工”。

### 4. 输出下一个 token 的概率

模型最后不会直接输出一句话，而是输出“下一个 token 是什么”的概率分布。

例如：

```text
下一个 token 是 “是” 的概率：0.32
下一个 token 是 “属于” 的概率：0.18
下一个 token 是 “NVIDIA” 的概率：0.15
```

服务端会根据采样参数选择一个 token。

### 5. 循环生成完整回答

生成一个 token 后，这个 token 会被拼回上下文，继续预测下一个 token。

流程是：

```text
输入 prompt
-> 预测第 1 个 token
-> 预测第 2 个 token
-> 预测第 3 个 token
-> ...
-> 遇到停止条件
-> 返回完整回答
```

所以大模型输出越长，推理循环次数越多。

## Prefill 和 Decode

大模型推理可以分成两个阶段。

### Prefill 阶段

Prefill 是“读题”的阶段。

模型把用户输入的 prompt 一次性处理完，建立上下文状态。

特点：

1. 输入越长，prefill 越慢。
2. 主要影响首 token 延迟。
3. 适合并行计算。

### Decode 阶段

Decode 是“写答案”的阶段。

模型一个 token 一个 token 地生成回答。

特点：

1. 输出越长，decode 越久。
2. 主要影响 tokens/s。
3. 每一步都依赖上一步结果。

## KV Cache

如果每生成一个 token 都重新计算所有历史 token，会非常慢。

所以模型会缓存历史 token 的 key/value 状态，这个缓存叫 KV Cache。

可以理解为：

```text
KV Cache = 模型生成时保存的上下文计算结果
```

KV Cache 的作用是减少重复计算。

但 KV Cache 会占显存，所以：

```text
上下文越长 -> KV Cache 越大 -> 显存占用越高
并发越高 -> KV Cache 越多 -> 显存占用越高
```

这就是为什么大模型部署时，不能只看模型权重大小，还要看上下文长度和并发数。

## 放到当前项目里看

当前项目里 Qwen 的请求路径大致是：

```text
curl 请求
-> vLLM API Server
-> tokenizer
-> Qwen3.6-27B-FP8
-> 4 张 H100 使用 TP=4 协同推理
-> 生成 token
-> detokenizer
-> 返回 OpenAI 兼容 API 结果
```

需要重点观察的指标：

1. 首 token 延迟。
2. 输出 tokens/s。
3. GPU 显存占用。
4. 长上下文是否稳定。
5. 并发请求是否稳定。
