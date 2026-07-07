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

## 双模型并行部署(TP=2 × 2)

当前项目在 4 卡 H100 上同时运行两个模型:

```text
Qwen3.6-27B-FP8   → GPU 0,1 (TP=2) → 端口 8000
Agents-A1-FP8     → GPU 2,3 (TP=2) → 端口 8001
```

### TP=2 的含义

`TP` 是 tensor parallelism,中文是张量并行。`TP=2` 表示:

```text
一个模型服务同时使用 2 张 GPU
2 张 GPU 共同跑一个模型
```

在 vLLM 里,对应参数是:

```text
--tensor-parallel-size 2
```

当前项目配置在 `config/serving.env`:

```text
QWEN_FP8_TENSOR_PARALLEL_SIZE=2
AGENTS_TENSOR_PARALLEL_SIZE=2
```

### GPU 隔离:CUDA_VISIBLE_DEVICES

两个模型同时运行时,需要用 `CUDA_VISIBLE_DEVICES` 限制每个容器可见的 GPU:

```text
QWEN_FP8_CUDA_VISIBLE_DEVICES=0,1    # Qwen 容器只用 GPU 0,1
AGENTS_CUDA_VISIBLE_DEVICES=2,3      # Agents 容器只用 GPU 2,3
```

容器内 vLLM 看到的 GPU 编号会重新映射为 0,1(即使宿主机是 2,3),所以 `--tensor-parallel-size 2` 对应容器内的 GPU 数。

### 为什么不用 TP=4

- TP=4 是 4 卡全部给一个模型,单模型吞吐最高,但同一时刻只能服务一个模型。
- 双模型并行(TP=2 × 2)牺牲单模型峰值吞吐,换取同时提供两个不同模型的能力。
- 适合 opencode 这类工具:小模型做快速补全,大模型做复杂任务。

## FP8

FP8 是一种 8-bit 浮点格式。

Qwen3.6-27B-FP8 和 Agents-A1-FP8 的权重都使用 FP8,可以降低显存占用,提高 H100 上的推理效率。

简单对比：

```text
BF16 / FP16：更常见,显存占用更高
FP8：更省显存,H100 支持更好
```

需要注意：

1. FP8 需要推理框架支持。
2. FP8 需要 GPU 硬件支持。
3. H100 很适合跑 FP8。

## 上下文长度

上下文长度是模型一次能看到的 token 数量。

当前两个模型都配置为原生最大值:

```text
QWEN_FP8_MAX_MODEL_LEN=262144   # 256K
AGENTS_MAX_MODEL_LEN=262144     # 256K
```

上下文越长,模型能看更多历史信息,但 KV Cache 显存占用也会变大。

KV cache 量化(`--kv-cache-dtype fp8`)可以把 KV cache 显存占用减半,是当前项目在 2 卡 TP=2 下还能放下 256K 上下文的关键。

## OpenAI 兼容 API

OpenAI 兼容 API 的意思是:虽然后端跑的是本地 Qwen 或 Agents,但接口格式模仿 OpenAI。

好处是:

1. 客户端容易接入。
2. 可以用 `/v1/chat/completions` 发对话请求。
3. 很多已有工具可以直接改 base_url 使用。

当前两个模型的 served-model-name 是:

```text
qwen3.6-27b-fp8    (端口 8000)
agents-a1-fp8      (端口 8001)
```

请求时需要写:

```json
{"model": "qwen3.6-27b-fp8"}
```

或

```json
{"model": "agents-a1-fp8"}
```

## MTP / Speculative Decoding

MTP 是 multi-token prediction。

普通生成通常是一个 token 一个 token 地预测。MTP 的目标是一次尝试预测多个 token,从而提升生成速度。

两个模型都启用 MTP:

```text
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

- MTP 只加速 decode 阶段,prefill 不加速。
- MoE 模型(Agents-A1)decode 计算密度低,MTP 加速效果更明显。
- 启用后启动时间会增加(需要编译额外图)。

## Function Calling(工具调用)

opencode 等 Agent 工具需要 Function Calling(让模型调用工具),vLLM 默认关闭。

vLLM 启动加两个参数:

```text
--enable-auto-tool-choice          # 开启工具调用
--tool-call-parser qwen3_coder     # 指定解析格式(两个模型都用这个)
```

不同模型用不同的 parser,选错了模型会输出工具调用文本但不被解析。两个模型的官方推荐都是 `qwen3_coder`。

## One API 网关

两个 vLLM 端点(8000、8001)通过 One API 网关统一对外:

```text
应用 → One API (10.30.75.58:18082) → vLLM (10.16.11.24:8000 或 8001)
```

One API 的作用:
- 统一入口:多个模型通过一个端口访问
- 计费统计:记录每次调用的 token 数
- Token 管理:给不同应用发不同 token
- 渠道路由:根据 model 名自动转发到对应后端

每个模型配一个「渠道」,渠道类型=OpenAI,Base URL 指向 vLLM 端点。

## Text-only 模式

Qwen3.6-27B-FP8 是带视觉 encoder 的模型。

当前项目只用文本、代码和 Agent 场景,所以加 `--language-model-only` 跳过 vision encoder:

- 减少不需要的多模态部分开销。
- 给文本推理和 KV Cache 留更多空间。
- 让纯文本服务更稳定。

Agents-A1-FP8 是纯文本模型,不需要这个参数。
