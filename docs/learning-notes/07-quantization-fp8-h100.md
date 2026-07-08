# 量化、FP8 与 H100 Tensor Core

## 一句话理解

量化是把权重或激活值从高精度（FP16/BF16）压到低精度（FP8/INT8/FP4），用精度换显存和速度；H100 有原生 FP8 Tensor Core，所以 FP8 几乎免费提速；但 NVFP4 在 H100 上没有原生计算单元，要靠 Marlin 反量化，反而吃掉理论收益。

---

## 一、浮点数格式基础

### 1. 浮点数的结构

浮点数 = 符号位（sign） + 指数位（exponent） + 尾数位（mantissa）

```text
value = (-1)^sign × 1.mantissa × 2^(exponent - bias)
```

- 指数位决定**动态范围**（能表示多大、多小）。
- 尾数位决定**精度**（能区分多接近的两个数）。

### 2. 常见格式对比

| 格式 | 符号 | 指数 | 尾数 | 总位数 | 字节数 | 动态范围 | 精度 |
|---|---|---|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | 32 | 4 | 极大 | 极高 |
| FP16 | 1 | 5 | 10 | 16 | 2 | 小（易溢出） | 中 |
| BF16 | 1 | 8 | 7 | 16 | 2 | 大（同 FP32） | 低 |
| FP8 E4M3 | 1 | 4 | 3 | 8 | 1 | 中 | 低 |
| FP8 E5M2 | 1 | 5 | 2 | 8 | 1 | 大 | 极低 |
| INT8 | - | - | - | 8 | 1 | 固定 | 低 |
| FP4 E2M1 | 1 | 2 | 1 | 4 | 0.5 | 极小 | 极低 |

### 3. BF16 为什么是大模型主流

FP16 指数位只有 5 位，动态范围小，大模型训练时梯度容易溢出。BF16 用 FP32 的指数位（8 位），动态范围一样大，精度低一点但训练稳定。所以大模型权重和激活默认用 BF16。

### 4. FP8 的两种格式

| 格式 | 指数 | 尾数 | 适用场景 |
|---|---|---|---|
| E4M3 | 4 | 3 | 前向推理、权重存储（精度优先） |
| E5M2 | 5 | 2 | 反向传播梯度（动态范围优先） |

推理只用 E4M3。当前项目的 FP8 权重和 FP8 KV Cache 都是 E4M3。

---

## 二、量化的分类

### 1. 按量化对象分

| 类型 | 量化什么 | 典型格式 | 特点 |
|---|---|---|---|
| Weight-only | 只量化权重，激活保持高精度 | W4A16、W8A16 | 简单，权重读取快，激活计算还是高精度 |
| Weight+Activation | 权重和激活都量化 | W8A8、FP8 | 计算也用低精度，Tensor Core 加速 |

当前项目：

- Qwen3.6-27B-FP8：权重和激活都是 FP8（W8A8 等价），H100 原生加速。
- Qwen3.6-27B-NVFP4：权重是 FP4（W4A16），激活还是 BF16，要反量化。

### 2. 按量化粒度分

量化的核心难点是**缩放因子（scale）**怎么选。一个 scale 管多大范围，决定精度损失。

| 粒度 | scale 管辖范围 | 精度 | 复杂度 |
|---|---|---|---|
| per-tensor | 整个张量一个 scale | 最低 | 最简单 |
| per-channel | 每个输出通道一个 scale | 中 | 中等 |
| per-token | 每个 token 的激活一个 scale | 高 | 较复杂 |
| per-group | 每 N 个连续元素一个 scale | 最高 | 最复杂 |

大模型推理常用 per-channel（权重）+ per-token（激活），或 per-group（如 NVFP4 用 16 元素一组）。

### 3. 对称 vs 非对称

- 对称量化：`quantized = round(value / scale)`，零点就是 0，简单但负数范围浪费。
- 非对称量化：`quantized = round((value - zero_point) / scale)`，有零点偏移，精度更好但计算复杂。

FP8 通常用对称量化，INT8 可以用非对称。

---

## 三、H100 的 Tensor Core 与 FP8

### 1. Tensor Core 是什么

普通 CUDA Core 一次算一个乘加。Tensor Core 是矩阵乘法专用单元，一次算一个矩阵块（如 16×16×16）。

```text
普通 Core:   C[i,j] += A[i,k] × B[k,j]    （一次一个乘加）
Tensor Core: C[16,16] += A[16,16] × B[16,16]  （一次一整块矩阵）
```

H100 的 Tensor Core 算力远超普通 Core，是大模型推理速度的关键。

### 2. H100 Tensor Core 支持的格式

| 格式 | 算力（稠密，单卡） | 是否原生支持 |
|---|---|---|
| FP64 | 67 TFLOPS | ✓ |
| TF32 | 989 TFLOPS | ✓（FP32 的截断版本） |
| BF16 / FP16 | 1979 TFLOPS | ✓ |
| FP8 | 3958 TFLOPS | ✓（H100 新增） |
| INT8 | 3958 TOPS | ✓ |
| FP4 | 不原生支持 | ✗（Blackwell 才有） |

关键点：

- **FP8 算力是 BF16 的 2 倍**。所以 FP8 推理不仅省显存，计算也快一倍。
- **H100 没有原生 FP4 计算单元**。FP4 权重要先反量化成 BF16/FP8 再算，反量化本身有开销。

### 3. Transformer Engine

NVIDIA 提供的库，封装 FP8 Tensor Core 操作。vLLM 的 FP8 路径底层用 Transformer Engine 或 FlashInfer/Triton 实现。

作用：

1. 自动管理 FP8 缩放因子。
2. 把 BF16 矩阵乘法转成 FP8 Tensor Core 调用。
3. 处理 FP8 的精度问题（如 per-tensor scale、per-token scale）。

---

## 四、FP8 在当前项目的应用

### 1. Qwen3.6-27B-FP8 的量化方式

HuggingFace 上的 `Qwen/Qwen3.6-27B-FP8` 是**预量化模型**，权重已经存成 FP8 格式，vLLM 直接加载。

- 权重格式：FP8 E4M3，per-channel scale。
- 激活：推理时动态量化成 FP8（per-token scale）。
- 计算：直接走 H100 FP8 Tensor Core，算力 3958 TFLOPS。

显存对比：

```text
BF16 权重: 27B × 2 bytes = 54 GB
FP8 权重:  27B × 1 byte  = 27 GB  （省一半）
```

2 卡 TP=2：每卡 13.5 GB 权重，剩 66 GB 给 KV Cache 和激活，能放 256K 上下文。

### 2. FP8 KV Cache

默认 KV Cache 用 BF16（2 字节/token）。`--kv-cache-dtype fp8` 把它压成 FP8（1 字节/token）。

效果：

- KV Cache 显存减半。
- 能放更长的上下文或更多并发。

代价：

- KV Cache 精度下降，长上下文可能有精度损失（注意力分数偏移）。
- 读写 KV Cache 时可能要反量化，有微小开销。

当前项目 256K 上下文 + TP=2，不开 fp8 KV cache 根本放不下，所以这是必需优化。

### 3. FP8 的精度问题

FP8 只有 4 位指数 + 3 位尾数，能表示的值有限：

```text
E4M3 能表示的正数范围: ~0.00195 到 ~448
精度: 3 位尾数，相对精度约 1/8
```

所以 FP8 对**数值范围敏感**的操作（如 softmax 前的 logits、attention score）可能不准。常见做法：

- softmax 在 FP32 里算。
- attention score 用 FP16 累加。
- 只有线性层（QKV 投影、MLP）用 FP8。

vLLM 的 FP8 路径会自动处理这些，但极端情况（如数值溢出）仍可能出错。

---

## 五、NVFP4 与 Marlin 反量化

### 1. NVFP4 是什么

NVFP4 是 NVIDIA 定义的 4-bit 浮点格式（E2M1），用于 ModelOpt 量化工具。

```text
NVFP4: 1 符号 + 2 指数 + 1 尾数 = 4 bit = 0.5 byte
```

- 优势：权重再省一半（27B 模型只要 13.5 GB）。
- 劣势：精度极低（只有 1 位尾数），必须用 per-group scale（如 16 元素一组）补偿。

### 2. H100 没有原生 FP4 计算

H100 Tensor Core 支持 FP8，但不支持 FP4。所以 NVFP4 权重不能直接算，要**反量化**：

```text
存储: NVFP4 (0.5 byte) + per-group scale
计算: 反量化成 BF16 (2 bytes) → BF16 Tensor Core 计算
```

反量化发生在每次权重读取时。Marlin 是 NVIDIA 优化的反量化 + 矩阵乘法 kernel，能把反量化和矩阵乘融合在一起，减少开销。

### 3. 为什么 NVFP4 在 H100 上没比 FP8 快

理论分析（见 `03-qwen3nvfp4-optimization.md` roofline 章节）：

| 量化 | 每卡权重（TP=4） | 理论 decode 上界 |
|---|---|---|
| FP8 | 6.75 GB | 496 tok/s |
| NVFP4 | 3.375 GB | 992 tok/s（理论 2 倍） |

实测：

| 配置 | 实测单流 decode |
|---|---|
| FP8 + MTP2 | 147 tok/s |
| NVFP4 + MTP3 + decode-only CUDA graph | 162 tok/s |

NVFP4 理论上界是 FP8 的 2 倍，实测只快 10%。原因：

1. **Marlin 反量化开销**：每次权重读取都要反量化，吃掉一部分省下的访存时间。
2. **H100 无原生 FP4 单元**：算力没提升，只是访存减少。
3. **其他瓶颈**：attention、KV Cache、通信、MTP draft 模型开销不变。

### 4. 什么时候 NVFP4 才真正划算

- **Blackwell 架构（B100/B200）**：有原生 FP4 Tensor Core，反量化消失，理论 2 倍收益能兑现。
- **极小 batch + 长序列**：访存瓶颈占比更大，反量化开销占比下降。
- **显存极度紧张**：NVFP4 权重小，能腾更多显存给 KV Cache。

H100 是 FP4 的"过渡期硬件"，NVFP4 主要是为了省显存，不是为了提速。

---

## 六、ModelOpt 量化工具

### 1. ModelOpt 是什么

NVIDIA 的模型量化工具，能把 BF16 模型量化成 FP8 或 NVFP4。

```text
BF16 模型 → ModelOpt → FP8 或 NVFP4 模型
```

`nvidia/Qwen3.6-27B-NVFP4` 就是用 ModelOpt 量化的。

### 2. 混合量化（Mixed Precision）

ModelOpt 不是一刀切全量化，而是**按层分析，混合使用**：

- 对精度敏感的层：保留 FP8。
- 对精度不敏感的层：用 NVFP4。

`nvidia/Qwen3.6-27B-NVFP4` 的 `config.json` 里有 `quantization=modelopt_mixed`，说明是 FP8 + NVFP4 混合。

vLLM 加载时的识别日志：

```text
Detected ModelOpt fp8 checkpoint
Detected ModelOpt NVFP4 checkpoint
quantization=modelopt_mixed
```

### 3. 量化校准（Calibration）

量化不是直接截断，而是用一批样本数据跑一遍，统计每层的数值范围，选合适的 scale。这叫校准。

- 校准数据：通常用少量通用文本（如 WikiText）。
- 校准质量：直接影响量化后精度。
- 预量化模型（如 HF 上的 FP8 版本）已经校准好，直接用即可。

---

## 七、量化与 KV Cache 的关系

### 1. 权重量化 vs KV Cache 量化

| 对象 | 量化什么 | 影响 | 当前项目 |
|---|---|---|---|
| 权重量化 | 模型参数 | 显存 + 计算速度 | FP8 / NVFP4 |
| KV Cache 量化 | 推理时的 K/V 缓存 | 显存（上下文长度、并发数） | fp8 |

两者独立，可以单独开：

```bash
--quantization fp8              # 权重量化（模型已是 FP8 时可省）
--kv-cache-dtype fp8            # KV Cache 量化
```

### 2. KV Cache 量化的精度影响

KV Cache 存的是每层的 K、V 张量。FP8 化后：

- **短期影响小**：最近 token 的 K/V 精度足够，attention 正常。
- **长期影响累积**：长上下文时，靠前 token 的 K/V 经过多次读取，误差可能累积。
- **极端情况**：数值很小的 K/V 量化后可能变成 0，信息丢失。

实践上，fp8 KV cache 对大多数任务影响可接受，但对需要精确长程依赖的任务（如长文档检索、代码理解）可能有细微退化。

### 3. 当前项目的显存账

Qwen3.6-27B-FP8，TP=2，256K 上下文：

| 项目 | bf16 KV Cache | fp8 KV Cache |
|---|---|---|
| 权重（每卡） | 13.5 GB (FP8) | 13.5 GB (FP8) |
| 单请求 256K KV Cache | 64 GB | 32 GB |
| 2 卡总显存 | 160 GB | 160 GB |
| 可用给 KV 的空间（每卡） | ~60 GB | ~60 GB |
| 单卡可放 256K 请求数 | ~1 个 | ~2 个 |

fp8 KV cache 把可并发数翻倍，是 256K 上下文可行的关键。

---

## 八、放到当前项目里看

### 1. 量化路径总结

| 模型 | 权重量化 | KV Cache | 算力路径 | 显存/卡 |
|---|---|---|---|---|
| Qwen3.6-27B-FP8 | FP8 E4M3 | FP8 | H100 FP8 Tensor Core（原生） | 71.6 GB |
| Agents-A1-FP8 | FP8 (compressed-tensors) | FP8 | H100 FP8 Tensor Core（原生） | 75.3 GB |
| Qwen3.6-27B-NVFP4（历史） | NVFP4 + FP8 混合 | auto | Marlin 反量化 → BF16 计算 | 76.4 GB |

### 2. 为什么最终选 FP8 而不是 NVFP4

| 维度 | FP8 | NVFP4 |
|---|---|---|
| 实测速度 | 147 tok/s（MTP2） | 162 tok/s（MTP3 + CUDA graph） |
| 稳定性 | 无 NaN bug | torch.compile 触发 NaN |
| 配置复杂度 | 简单 | 需 workaround（禁 torch.compile + triton GDN） |
| 显存 | 27 GB 权重 | 13.5 GB 权重 |
| 长上下文 | 256K 可行 | 降到 32K（为 MTP 腾显存） |

FP8 虽然权重比 NVFP4 大一倍，但：

1. 原生计算，无需反量化。
2. 无 torch.compile NaN 问题。
3. 256K 上下文可行。
4. 速度差距小（147 vs 162）。

综合更优，所以当前双模型并行用 FP8。

### 3. DeepSeek-V4 的 FP4

`DeepSeek-V4-Flash-DSpark` 的 config 里有 `expert_dtype: fp4`。这是 DeepSeek 自己的 FP4 实现（DSSpark），不是 NVFP4。部署时要用 `DS_HF_OVERRIDES` 处理，属于待测试项。

---

## 总结

| 概念 | 作用 | 项目里的体现 |
|---|---|---|
| FP8 E4M3 | 推理用的 8-bit 浮点格式 | Qwen/Agents 权重都是 FP8 |
| H100 Tensor Core | 矩阵乘法专用单元，FP8 算力 2× BF16 | FP8 原生加速 |
| Weight-only 量化 | 只量化权重，激活高精度 | NVFP4 (W4A16) |
| W8A8 量化 | 权重和激活都量化 | FP8（W8A8 等价） |
| KV Cache 量化 | 压缩 KV Cache 显存 | `--kv-cache-dtype fp8` |
| Marlin 反量化 | NVFP4 权重转 BF16 计算 | NVFP4 在 H100 上的开销来源 |
| ModelOpt | NVIDIA 量化工具 | nvidia/Qwen3.6-27B-NVFP4 |
| Transformer Engine | NVIDIA FP8 计算库 | vLLM FP8 路径底层 |
