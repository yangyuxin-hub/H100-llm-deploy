# KV Cache、PagedAttention 与长上下文

## 一句话理解

KV Cache 是大模型推理快的原因，也是显存吃紧的原因；PagedAttention 是 vLLM 管好用这块显存的核心手段；长上下文之所以贵，是因为注意力计算和 KV Cache 都随序列长度非线性增长。

---

## 一、KV Cache 复盘

### 1. 为什么需要 KV Cache

Transformer 的 self-attention 每生成一个 token，都要"看到"之前所有 token。如果每步都重算所有历史 token 的 Key/Value 投影，序列越长越慢。

解决方法：把每一层算出的 K、V 缓存下来，下一步只算新 token 的 K、V 追加进去。

```text
第 t 步输入: x_t
每层都做:
  K_t = W_K · x_t        # 只算新 token
  V_t = W_V · x_t        # 只算新 token
  K_cache = [K_1, ..., K_t]   # 追加
  V_cache = [V_1, ..., V_t]   # 追加
  attention(Q_t, K_cache, V_cache)
```

这就是 KV Cache。它把 decode 阶段从 O(n²) 降到每步 O(n)。

### 2. KV Cache 的显存代价

KV Cache 占显存，而且随序列长度和并发数线性增长。精确公式：

```text
KV Cache 显存 = 2 × num_layers × seq_len × num_kv_heads × head_dim × dtype_size × batch_size
```

- `2`：K 和 V 两个张量
- `num_layers`：每层都有一份
- `seq_len`：当前序列长度（含 prompt + 已生成）
- `num_kv_heads`：KV 头数（GQA 下不等于 query 头数，见下节）
- `head_dim`：每个头的维度
- `dtype_size`：fp16/bf16=2 字节，fp8=1 字节
- `batch_size`：并发请求数

### 3. 放到当前项目算一遍

以 Qwen3.6-27B-FP8 为例（读 `config.json` 得到）：

| 参数 | 值 |
|---|---|
| num_hidden_layers | 64 |
| num_attention_heads | 40（query 头） |
| num_kv_heads | 8（GQA，5:1 压缩） |
| head_dim | 128 |
| max_model_len | 262144（256K） |
| kv-cache-dtype | fp8（1 字节） |

单请求满上下文的 KV Cache：

```text
2 × 64 × 262144 × 8 × 128 × 1 byte
= 2 × 64 × 262144 × 1024 byte
= 34,359,738,368 byte
≈ 32 GiB
```

也就是说，**一个 256K 请求的 KV Cache 就要 32 GiB 显存**。这就是为什么 TP=2 下每卡 80G 显存，扣掉权重和框架开销后，能同时塞下的 256K 请求并发数有限。

vLLM 启动日志会打印 `Maximum concurrency for 262,144 tokens per request: Xx`，就是用这个公式反推的。

---

## 二、GQA（Grouped-Query Attention）

### 1. 为什么 KV Cache 公式里要单独算 `num_kv_heads`

早期 Transformer 用 MHA（Multi-Head Attention）：query 头数 = KV 头数。例如 40 个 query 头，就有 40 个 KV 头，KV Cache 很大。

后来出现两种优化：

| 方案 | query 头 | KV 头 | KV Cache 大小 |
|---|---|---|---|
| MHA | H | H | 基准 |
| MQA（Multi-Query） | H | 1 | 缩小 H 倍 |
| GQA（Grouped-Query） | H | G（1 < G < H） | 缩小 H/G 倍 |

- MQA 最省显存，但精度损失大。
- GQA 是折中：把 query 头分组，每组共享一对 KV。
- 现在的 27B 级模型几乎都用 GQA。

### 2. Qwen3.6-27B 的 GQA 配置

```text
num_attention_heads = 40   # query 头
num_kv_heads = 8           # KV 头
分组比例 = 40 / 8 = 5      # 每 5 个 query 头共享 1 对 KV
```

相比 MHA，KV Cache 缩小到 1/5。如果还是 MHA，单请求 256K 的 KV Cache 会是 160 GiB，2 张 H100 根本放不下。

### 3. GQA 对 TP 的影响

GQA 的 KV 头数要能被 TP size 整除，否则切不均匀。

```text
num_kv_heads = 8
TP=2 → 每卡 4 个 KV 头 ✓
TP=4 → 每卡 2 个 KV 头 ✓
TP=8 → 每卡 1 个 KV 头（退化为 MQA，可行但失去 GQA 优势）
```

当前项目 TP=2，每卡 4 个 KV 头，没问题。

---

## 三、PagedAttention（vLLM 的核心创新）

### 1. 传统 KV Cache 管理的问题

最朴素的思路：每个请求预分配一整块连续显存，大小按 `max_model_len` 算。

问题：

1. **内部碎片**：请求实际只用了 2K token，但预分配了 256K 空间，浪费 254K。
2. **外部碎片**：请求来了又走，显存里全是大小不一的空洞，新请求塞不进去。
3. **无法共享**：两个请求有相同的 system prompt，KV Cache 各算一份，重复浪费。

### 2. PagedAttention 的思路：把 KV Cache 当虚拟内存管

借鉴操作系统的**分页内存管理**：

```text
逻辑地址（请求看到的连续 KV）  →  物理块（实际显存里的固定大小 block）
```

核心概念：

| 概念 | 对应 OS 术语 | 作用 |
|---|---|---|
| block（物理块） | page | KV Cache 的最小分配单元，固定 token 数（如 16） |
| block table（页表） | page table | 记录"请求的第 i 个逻辑块 → 物理块编号" |
| scheduler | MMU | 分配/回收物理块，维护页表 |

### 3. 工作流程

```text
1. 请求到来，scheduler 分配若干物理 block
2. prefill：算出 prompt 的 KV，按 block 大小切块，填入物理 block
3. block table 记录：请求 A 的逻辑块 0 → 物理 block #7，逻辑块 1 → 物理 block #23 ...
4. decode：新生成的 token 的 KV 追加到当前最后一个 block，满了就申请新 block
5. 请求结束：回收所有 block，放回空闲池
```

关键收益：

1. **无内部碎片**：只在最后一个 block 有少量浪费（平均浪费半个 block）。
2. **无外部碎片**：物理 block 大小固定，任意空闲 block 都能用。
3. **可共享**：两个请求前缀相同，block table 指向同一组物理 block（copy-on-write）。

### 4. block 大小的取舍

vLLM 默认 `block_size = 16`（每个物理块存 16 个 token 的 KV）。

- 太大：最后一个 block 浪费多，内部碎片上升。
- 太小：block table 变长，访存间接寻址次数多，kernel 效率下降。
- 16 是经验值，多数场景够用。

### 5. PagedAttention 对 attention kernel 的要求

普通 attention kernel 假设 K/V 在连续内存里。PagedAttention 要重写 kernel，让它能按 block table 跳跃读取。

vLLM 的 attention kernel（FlashAttention、FlashInfer、Triton）都有 PagedAttention 版本，这就是为什么 backend 选择和 PagedAttention 强相关。

---

## 四、Continuous Batching（连续批处理）

### 1. 静态批处理的问题

传统做法：凑齐一个 batch，一起 prefill，一起 decode，一起结束。

问题：同一 batch 里请求长度差异大，短请求要等长请求结束，GPU 空转。

```text
请求 A: |----|
请求 B: |--------|
请求 C: |--|
静态 batch: |--------|  （C 和 A 空等 B）
```

### 2. Continuous Batching 的做法

不等一个 batch 全部结束，**每一步都可以插入新请求、移出已完成请求**：

```text
时刻 1: [A, B, C]  都在 decode
时刻 2: [A, B, C, D]  D 完成 prefill 加入
时刻 3: [B, C, D]  A 结束移出
时刻 4: [B, C, D, E]  E 加入
```

这要求 prefill 和 decode 能混在一起做，也就是 **chunked prefill**（把长 prompt 切块，和 decode 一起调度）。

### 3. 配合 PagedAttention

Continuous batching 之所以在 vLLM 里能跑起来，是因为 PagedAttention 让每个请求的 KV Cache 独立管理：

- 新请求加入：申请新 block，不影响已有请求。
- 请求移出：回收 block，立即可被新请求用。
- 不同请求在同一 batch 里，block table 各自独立。

没有 PagedAttention，continuous batching 很难高效实现（要频繁搬运整块 KV）。

### 4. 关键调度参数

| 参数 | 含义 | 当前项目值 |
|---|---|---|
| `max-num-seqs` | 一个 batch 里最多多少个请求同时 decode | 16 |
| `max-num-batched-tokens` | 一次 forward 最多处理多少 token（prefill+decode 合计） | 默认值 |
| `gpu-memory-utilization` | 允许 vLLM 用多少显存（含权重+KV Cache） | 0.90 |

benchmark 里发现"并发 16 未饱和"，就是因为 `max-num-seqs=16` 卡住了调度上限——GPU 还有算力和显存，但调度器不让更多请求进 batch。调高这个值能进一步提升吞吐。

---

## 五、长上下文的代价

### 1. 注意力计算的 O(n²) 复杂度

标准 self-attention：

```text
attention(Q, K, V) = softmax(Q·K^T / √d) · V
```

- Q 是 `[seq_len, d]`
- K^T 是 `[d, seq_len]`
- Q·K^T 是 `[seq_len, seq_len]` —— 这就是 O(n²) 的来源

prefill 阶段一次处理整个 prompt，所以 prefill 计算量随 prompt 长度平方增长。

| seq_len | attention 矩阵大小 | 相对 4K |
|---|---|---|
| 4K | 16M | 1× |
| 32K | 1G | 64× |
| 256K | 64G | 4096× |

这就是为什么 benchmark 里长文档（8K 输入）TTFT 是 342ms，而对话（500 输入）只有 70ms——差近 5 倍，符合 prefill 计算量增长。

### 2. FlashAttention：用 tiling 把显存从 O(n²) 降到 O(n)

标准 attention 要把 `[seq_len, seq_len]` 的注意力矩阵写回 HBM，显存爆炸。

FlashAttention 的核心：**不把完整注意力矩阵落地到 HBM**。

做法（tiling）：

```text
1. 把 Q、K、V 切成小块（tile），放进 SRAM（片上高速缓存）
2. 在 SRAM 里算一小块注意力，累加结果
3. 只把最终输出写回 HBM，中间矩阵不落地
```

效果：

- 显存：O(n²) → O(n)（只存 KV Cache，不存注意力矩阵）
- 速度：减少 HBM 读写，反而更快（HBM 带宽是瓶颈）

### 3. FlashAttention 与 PagedAttention 的关系

- FlashAttention：解决"单次 attention 计算的显存和速度"。
- PagedAttention：解决"多个请求的 KV Cache 显存管理"。
- vLLM 的 FlashAttention backend：两者的结合，在 tile 计算时按 block table 跳跃读取。

### 4. decode 阶段不是 O(n²)

decode 每步只生成 1 个 token，Q 是 `[1, d]`，K/V 是 `[seq_len, d]`，attention 是 `[1, seq_len]`，计算量 O(n)。

但 decode 是**访存瓶颈**：每步都要把整个 KV Cache 读一遍。序列越长，每步越慢。

| seq_len | 单步 decode 需读 KV Cache（GQA，8头，fp8） |
|---|---|
| 4K | 2 × 64 × 4096 × 8 × 128 × 1 = 512 MB |
| 256K | 2 × 64 × 262144 × 8 × 128 × 1 = 32 GB |

256K 上下文每步 decode 要读 32GB KV Cache，按 H100 HBM 3.35TB/s 算，光读 KV 就要 9.5ms，再加上权重和其他开销，TPOT 会显著上升。

---

## 六、位置编码与长上下文外推

### 1. RoPE（Rotary Position Embedding）

主流大模型（Qwen、LLaMA、DeepSeek）用 RoPE 做位置编码。核心思想：用旋转矩阵把位置信息编码进 Q、K。

```text
q_m = R_m · q    # q 在位置 m 被旋转
k_n = R_n · k    # k 在位置 n 被旋转
q_m · k_n = q · R_{n-m} · k   # 内积只依赖相对距离 (n-m)
```

RoPE 的好处：

1. 相对位置，适合自然语言。
2. 可以外推（理论上任意长度都能算）。

### 2. 为什么训练长度有限，但能"扩展"上下文

模型 `config.json` 里的 `max_position_embeddings` 是训练时见过的最大长度。超过这个长度，RoPE 旋转矩阵的频率会让模型"没见过"的相对位置表现异常。

外推方法（让模型在更长上下文上还能工作）：

| 方法 | 原理 | 特点 |
|---|---|---|
| 直接外推 | 不做任何处理 | 超过训练长度后效果急剧下降 |
| NTK-aware | 调整 RoPE 基础频率 base，让低频更平滑 | 常用，无需微调 |
| YaRN | 分段插值 + NTK | 效果更好，DeepSeek/Qwen 长上下文常用 |
| Dynamic NTK | 根据当前序列长度动态调整 | 适合变长输入 |

Qwen3.6 的 `config.json` 通常会带 `rope_scaling` 字段，说明已经用 YaRN 或类似方法做过外推，所以 `max_position_embeddings=262144` 是"原生支持"而非硬撑。

### 3. 外推不是免费的

即使用了 YaRN，长上下文仍有代价：

1. **精度下降**：超出训练长度的部分，注意力分布可能失真，模型可能"忘记"中间内容（lost in the middle）。
2. **显存和速度**：KV Cache 和 attention 计算随长度增长，上面已算过。
3. **prefill 慢**：O(n²) 计算量。

所以"支持 256K"不等于"256K 和 4K 一样快一样准"。

---

## 七、Prefix Caching（前缀缓存）

### 1. 场景

多轮对话、RAG、Agent 工具调用，很多请求有相同的 system prompt 或前缀。每个请求都重算一遍 prefill，浪费。

### 2. PagedAttention 让前缀共享变得简单

因为 KV Cache 是按 block 管理的，如果两个请求前缀相同：

```text
请求 A: [system_prompt][user_msg_1]
请求 B: [system_prompt][user_msg_2]
```

system_prompt 部分的 KV Cache 可以复用：

1. 请求 A 来了，正常 prefill，system_prompt 的 KV 存在物理 block #10-#15。
2. 请求 B 来了，检测到 system_prompt 相同，block table 直接指向 #10-#15。
3. 只对 user_msg_2 做 prefill。

### 3. vLLM 开启方式

```bash
--enable-prefix-caching
```

效果：

- 首次请求正常速度。
- 后续相同前缀请求 prefill 阶段加速 50-90%。
- 当前项目测试用不同 prompt 看不到效果，需要多轮对话或固定 system prompt 场景才有效。

---

## 八、放到当前项目里看

### 1. 为什么 TP=2 下能放下 256K

关键配置组合：

```text
--max-model-len 262144          # 256K 上下文
--kv-cache-dtype fp8            # KV Cache 量化，显存减半
--gpu-memory-utilization 0.90   # 用满 90% 显存
--tensor-parallel-size 2        # 每卡只放一半权重
--language-model-only           # 跳过 vision encoder，省显存给 KV
```

fp8 KV cache 是关键：把单请求 256K 的 KV Cache 从 64 GiB（bf16）降到 32 GiB（fp8），2 卡才能同时放下多个并发请求。

### 2. benchmark 里的现象解释

| 现象 | 机制解释 |
|---|---|
| 长文档 TTFT 342ms vs 对话 70ms | prefill O(n²)，8K 输入比 500 输入计算量大几十倍 |
| 并发 16 未饱和 | `max-num-seqs=16` 限制了 batch 里的请求数，GPU 还有余量 |
| TPOT 随并发温和增长 | decode 是访存瓶颈，batch 增大后 KV Cache 读取量上升 |
| ITL P99 长尾增大 | 调度抖动 + KV Cache block 分配/回收延迟 |

### 3. 优化方向

- 调高 `max-num-seqs`（如 32、64），让更多请求进 batch，提升吞吐。
- 开 `--enable-prefix-caching`，多轮对话场景 prefill 加速。
- 开 `--enable-chunked-prefill`（vLLM 新版默认），长 prompt 切块和 decode 混合调度，降低 TTFT。

---

## 总结

| 概念 | 作用 | 项目里的体现 |
|---|---|---|
| KV Cache | 避免重算历史 token，加速 decode | 256K 上下文单请求 32GiB |
| GQA | 减少 KV Cache 大小 | num_kv_heads=8，缩小 5 倍 |
| PagedAttention | 高效管理 KV Cache 显存，无碎片 | vLLM 能动态分配/回收 block |
| Continuous Batching | 请求级别动态调度，提升吞吐 | max-num-seqs=16 是当前调度上限 |
| FlashAttention | attention 计算显存 O(n²)→O(n) | 长上下文 prefill 能跑起来 |
| RoPE + YaRN | 位置编码外推到长上下文 | 原生支持 256K |
| Prefix Caching | 复用相同前缀的 KV Cache | 多轮对话场景可加速 |
