# 部署相关核心概念

## vLLM

vLLM 是大模型推理服务框架。

它在当前项目里负责：

1. 加载模型权重。
2. 把模型切到多张 GPU 上。
3. 管理 KV Cache。
4. 接收 OpenAI 兼容 API 请求。
5. 调度多个请求。
6. 执行 prefill 和 decode。
7. 把生成结果返回给客户端。

当前项目使用的接口是：

```text
/v1/models
/v1/chat/completions
```

## TP=4 和 DP=1

`TP` 是 tensor parallelism，中文是张量并行。

`TP=4` 的意思是：

```text
一个模型服务同时使用 4 张 GPU
4 张 GPU 共同跑一个模型
```

它不是：

```text
4 张 GPU 各跑一个完整模型
```

在 vLLM 里，对应参数是：

```text
--tensor-parallel-size 4
```

当前项目配置在：

```text
config/serving.env
TENSOR_PARALLEL_SIZE=4
DATA_PARALLEL_SIZE=1
```

`DP` 是 data parallelism，中文是数据并行。

`DP=1` 的意思是只启动 1 份模型实例。4 张卡全部给这个实例做 TP。

4 卡环境下当前推荐：

```text
TP=4, DP=1
```

原因是：Qwen3.6-27B-FP8 要支持 262K 长上下文和 MTP，先把 4 张 H100 都给同一个模型实例，显存和稳定性更稳。

后续如果要提高并发吞吐，可以再实验：

```text
TP=2, DP=2
```

这表示启动 2 份模型实例，每份模型使用 2 张 GPU。它更适合多用户并发，但单个请求可用的显存和长上下文空间会减少。

## FP8

FP8 是一种 8-bit 浮点格式。

Qwen3.6-27B-FP8 的权重使用 FP8，可以降低显存占用，提高 H100 上的推理效率。

简单对比：

```text
BF16 / FP16：更常见，显存占用更高
FP8：更省显存，H100 支持更好
```

需要注意：

1. FP8 需要推理框架支持。
2. FP8 需要 GPU 硬件支持。
3. H100 很适合跑 FP8。

## 上下文长度

上下文长度是模型一次能看到的 token 数量。

当前 Qwen 配置：

```text
QWEN_MAX_MODEL_LEN=262144
```

也就是 262K tokens。

上下文越长，模型能看更多历史信息，但 KV Cache 显存占用也会变大。

初次部署策略：

1. 先用原生 262K。
2. 如果显存压力大，降到 128K。
3. 不要一开始直接扩到 1M。

## OpenAI 兼容 API

OpenAI 兼容 API 的意思是：虽然后端跑的是本地 Qwen 或 DeepSeek，但接口格式模仿 OpenAI。

好处是：

1. 客户端容易接入。
2. 可以用 `/v1/chat/completions` 发对话请求。
3. 很多已有工具可以直接改 base_url 使用。

当前 Qwen 服务名是：

```text
qwen3.6-27b-fp8
```

请求时需要写：

```json
{
  "model": "qwen3.6-27b-fp8"
}
```

## MTP / Speculative Decoding

MTP 是 multi-token prediction。

普通生成通常是一个 token 一个 token 地预测。MTP 的目标是一次尝试预测多个 token，从而提升生成速度。

在 Qwen3.6-27B-FP8 里，可以后续实验：

```text
speculative_config={"method":"qwen3_next_mtp","num_speculative_tokens":2}
```

当前项目要求 Qwen 必须支持 MTP，所以 Qwen 默认开启 MTP。

推荐顺序：

1. 先使用 `TP=4, DP=1` 跑通 Qwen + MTP。
2. 再测试 text-only。
3. 最后再实验 `TP=2, DP=2` 的并发吞吐。

## Text-only 模式

Qwen3.6-27B-FP8 是带视觉 encoder 的模型。

如果当前任务只需要文本、代码和 Agent，可以测试 text-only 模式：

```text
--language-model-only
```

可能收益：

1. 减少不需要的多模态部分开销。
2. 给文本推理和 KV Cache 留更多空间。
3. 让纯文本服务更稳定。

是否作为默认配置，需要等 H100 上实际测试后决定。
