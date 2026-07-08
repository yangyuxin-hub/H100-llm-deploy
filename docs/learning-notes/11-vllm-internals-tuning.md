# vLLM 内部架构与性能调优

## 一句话理解

vLLM 由 Scheduler（调度器）、Worker（GPU 执行器）、KV Cache Manager（显存管理器）三层组成，调度器决定每步算哪些请求、Worker 负责实际计算、Manager 负责分配/回收 KV Cache block；理解 `max-num-seqs`、`max-num-batched-tokens`、`gpu-memory-utilization` 的相互作用，是性能调优的关键。

---

## 一、vLLM 的三层架构

### 1. 架构总览

```text
API Server (OpenAI 兼容接口)
    ↓
Engine (vLLM 核心)
    ├── Scheduler (调度器，CPU 侧)
    │     ├── 请求队列管理
    │     ├── prefill/decode 调度
    │     └── batch 组装
    ├── KV Cache Manager (显存管理)
    │     ├── block 分配/回收
    │     ├── block table 维护
    │     └── prefix caching
    └── Workers (GPU 侧执行器)
          ├── TP 每张卡一个 worker
          ├── 执行 attention/MLP 计算
          └── NCCL 通信
```

### 2. Scheduler（调度器）

**职责**：决定每一步（step）执行哪些请求，组装 batch。

**输入**：

- 等待队列（waiting）：新请求，还没 prefill。
- 运行队列（running）：已 prefill，正在 decode。
- 显存状态：空闲 block 数。

**输出**：一个 batch，包含：

- 要 prefill 的请求（从 waiting 取）。
- 要 decode 的请求（从 running 取）。
- 每个 request 的 block table。

**调度策略**：

1. 优先 decode running 中的请求（不中断生成）。
2. 剩余显存允许时，从 waiting 取新请求 prefill。
3. 显存不足时，preempt（换出）部分 running 请求（把 KV Cache 写回 CPU，腾显存）。

### 3. KV Cache Manager

**职责**：管理物理 block 的分配、回收、共享。

**核心数据结构**：

```text
free_blocks: 空闲物理 block 池
block_table[request_id]: 每个请求的逻辑块 → 物理块映射
```

**操作**：

- `allocate(request, num_tokens)`: 给请求分配 block。
- `free(request)`: 请求结束，回收 block。
- `can_allocate(num_tokens)`: 检查是否有足够 block。

**与 Scheduler 的配合**：

Scheduler 决定能否接收新请求，要先问 Manager "还有多少 free block"。

### 4. Workers

**职责**：在 GPU 上执行实际计算。

**每个 TP rank 一个 Worker**：

- TP=2 时有 2 个 worker，各负责一半权重。
- Worker 间用 NCCL all-reduce 通信。

**Worker 的工作**：

1. 接收 Scheduler 组装的 batch。
2. 执行 forward（attention + MLP）。
3. 返回生成的 token。

---

## 二、关键调度参数详解

### 1. max-num-seqs

**含义**：一个 batch 里最多多少个请求同时 decode。

**当前值**：16

**影响**：

- 太小：GPU 算力和显存没用满，吞吐低。
- 太大：显存压力（每个请求的 KV Cache），调度开销增加。

**benchmark 观察**：

当前项目并发 16 时 output 吞吐 1798 tok/s，但"未饱和"——说明 GPU 还有算力，是 `max-num-seqs=16` 卡住了调度上限。调高到 32 或 64 可能进一步提升吞吐。

**调优建议**：

```bash
# 测试不同 max-num-seqs
--max-num-seqs 32
--max-num-seqs 64
```

观察：

- 吞吐是否提升。
- 显存是否够（KV Cache 是否 OOM）。
- TPOT 是否恶化（batch 太大每步变慢）。

### 2. max-num-batched-tokens

**含义**：一次 forward 最多处理多少 token（prefill + decode 合计）。

**作用**：限制 prefill 阶段的 batch 大小，防止单次 prefill 太大挤占 decode。

**示例**：

```text
max-num-batched-tokens = 8192

情况 1: 新请求 prompt 4000 token + 10 个 decode 请求各 1 token
  总 token = 4000 + 10 = 4010 < 8192 ✓ 一起算

情况 2: 新请求 prompt 10000 token
  10000 > 8192 → chunked prefill，切成 8192 + 1808 两次算
```

**影响**：

- 太小：长 prompt 被切太碎，prefill 慢。
- 太大：单次 prefill 占满 GPU，decode 请求等待久，TPOT 抖动。

**与 chunked prefill 的关系**：

`--enable-chunked-prefill` 开启后，长 prompt 按这个参数切块，和 decode 混合调度。vLLM 新版默认开启。

### 3. gpu-memory-utilization

**含义**：vLLM 允许使用的显存占总显存的比例。

**当前值**：0.90

**实际分配**：

```text
总显存 80GB × 0.90 = 72GB 给 vLLM
其中:
  - 权重: 固定（FP8 27B TP=2 → 每卡 13.5GB）
  - 框架开销: 固定（约 5-10GB）
  - CUDA graph pool: 固定（约 2.5GB）
  - KV Cache: 动态（剩余空间全给它）
```

**影响**：

- 太低：KV Cache 空间少，能放的请求少，吞吐低。
- 太高：留给系统/其他进程的显存少，可能 OOM 或影响容器稳定性。

**调优建议**：

- 0.85-0.92 是常见范围。
- 0.95 以上风险高（框架临时 buffer 可能 OOM）。
- 双模型并行时，两个容器各占 0.90，加起来 180% 但物理上是不同卡，互不影响。

### 4. 三个参数的相互作用

```text
gpu-memory-utilization 决定总显存
  → 扣掉权重和开销 = KV Cache 空间
  → KV Cache 空间 / 单请求 KV = 能放多少请求
  → 但 max-num-seqs 限制了实际 batch 大小
  → max-num-batched-tokens 限制了单次 forward 的 prefill 量
```

**当前项目的瓶颈**：

benchmark 显示并发 16 未饱和，说明：

1. KV Cache 空间还有（能放更多请求）。
2. 但 `max-num-seqs=16` 限制了 batch 里的请求数。
3. 调高 `max-num-seqs` 能提升吞吐。

---

## 三、Prefill vs Decode 的调度差异

### 1. Prefill 的特点

- 计算密集（O(n²) attention）。
- 单次 forward 处理整个 prompt（或 chunk）。
- 不适合大 batch（每个请求 prompt 都长，合计 token 数爆炸）。

### 2. Decode 的特点

- 访存密集（每步读全部 KV Cache）。
- 单次 forward 每请求只生成 1 token。
- 适合大 batch（多请求摊销权重读取）。

### 3. 混合调度（chunked prefill）

不开 chunked prefill 时：

```text
情况 A: 这一步只 prefill（新请求加入）
  batch = [新请求 prompt]
情况 B: 这一步只 decode
  batch = [所有 running 请求各 1 token]
```

prefill 和 decode 交替，decode 会被 prefill 打断，TPOT 抖动。

开 chunked prefill 后：

```text
batch = [新请求 prompt 切块] + [running 请求各 1 token]
一步里同时 prefill 和 decode
```

好处：

1. decode 不被 prefill 打断，TPOT 稳定。
2. prefill 切块，长 prompt 不阻塞短请求。
3. GPU 利用率更高（prefill 计算密集 + decode 访存密集，混合用满算力和带宽）。

---

## 四、Preemption（抢占）

### 1. 什么时候触发

当显存不足，Scheduler 无法给新请求分配 block 时，要 preempt（换出）部分 running 请求：

```text
显存不足 → 选择一些 running 请求
  → 把它们的 KV Cache 写回 CPU 内存
  → 释放显存给新请求
  → 以后再换回来继续 decode
```

### 2. 代价

- 写回 CPU：PCIe 传输，慢。
- 换回 GPU：又要传输 + 重建状态。
- 被抢占的请求延迟增加。

### 3. 避免抢占

- 调低 `max-num-seqs`（减少并发）。
- 调低 `max-model-len`（减少单请求 KV Cache）。
- 开 `--kv-cache-dtype fp8`（KV Cache 减半）。
- 调高 `gpu-memory-utilization`（更多显存给 KV Cache）。

当前项目用 fp8 KV cache + 256K max-model-len + 0.90 utilization，就是为了在 TP=2 下避免抢占。

### 4. 检测抢占

vLLM 日志会有 `Preemption` 相关信息：

```bash
docker logs <container> 2>&1 | grep -i preempt
```

如果频繁 preempt，说明显存配置太激进，要调参。

---

## 五、Prefix Caching 的实现

### 1. vLLM 的 prefix caching

基于 PagedAttention 的 block 结构，自然支持前缀共享：

```text
请求 A: [system_prompt][user_msg_1]
  block table: [block_1, block_2, block_3, block_4]

请求 B: [system_prompt][user_msg_2]
  检测到 system_prompt 相同
  block table: [block_1, block_2, block_5, block_6]  ← 前两个 block 复用
```

### 2. 开启方式

```bash
--enable-prefix-caching
```

### 3. 效果场景

| 场景 | 效果 |
|---|---|
| 多轮对话（system prompt 固定） | prefill 加速 50-90% |
| RAG（相同检索上下文） | 显著加速 |
| Agent（工具定义固定） | 显著加速 |
| 单次请求（无重复前缀） | 无效果 |

### 4. 与 PagedAttention 的关系

prefix caching 依赖 PagedAttention 的 block 结构：

- block 是固定大小，可比较哈希。
- block table 允许逻辑块指向同一物理块。
- copy-on-write：请求要修改共享 block 时才复制。

没有 PagedAttention，prefix caching 实现复杂（要搬运整块连续 KV）。

---

## 六、性能调优方法论

### 1. 瓶颈定位

| 现象 | 可能瓶颈 | 排查 |
|---|---|---|
| TPOT 高 | decode 访存 / TP 通信 | 看 GPU 利用率、NVLink 利用率 |
| TTFT 高 | prefill 计算 / 调度 | 看 prefill 时间、batch 组成 |
| 吞吐低 | max-num-seqs / KV Cache 不足 | 看并发是否达上限、显存使用 |
| 长尾延迟大 | 调度抖动 / preempt | 看 preempt 次数、调度日志 |

### 2. 调优顺序

1. **先保证正确性**：模型能正常输出，无 NaN。
2. **开 CUDA graph**：decode 提速 2-5 倍。
3. **调 max-num-seqs**：找吞吐最优点。
4. **开 prefix caching**：多轮对话场景。
5. **开 chunked prefill**：降 TTFT 抖动。
6. **调 KV cache 量化**：省显存换更多并发。
7. **调 attention backend**：H100 用 FLASH_ATTN。

### 3. 当前项目的调优空间

| 参数 | 当前值 | 可尝试 |
|---|---|---|
| max-num-seqs | 16 | 32、64（提升吞吐） |
| enable-prefix-caching | 未开 | 开（opencode 多轮对话） |
| enable-chunked-prefill | 默认 | 显式开 + 调 max-num-batched-tokens |
| gpu-memory-utilization | 0.90 | 0.92（谨慎） |

---

## 七、benchmark 工具与指标

### 1. vLLM bench serve

当前项目用的在线 benchmark 工具：

```bash
vllm bench serve \
  --backend vllm \
  --model qwen3.6-27b-fp8 \
  --endpoint /v1/completions \
  --dataset-name random \
  --input-len 500 \
  --output-len 512 \
  --request-rate inf \
  --max-concurrency 16
```

### 2. 关键指标解读

| 指标 | 含义 | 当前项目值（并发 16） |
|---|---|---|
| TTFT | Time To First Token，首 token 延迟 | 111ms（对话） |
| TPOT | Time Per Output Token，单 token 生成时间 | 8.28ms |
| ITL | Inter-Token Latency，相邻 token 间隔 | 78.8ms（P99） |
| output tok/s | 输出吞吐 | 1798 |
| total tok/s | 总吞吐（含 prefill） | 3553 |
| 投机接受率 | MTP draft 被接受的比例 | 79.4% |

### 3. 指标间关系

```text
TPOT ≈ 1 / (单流 decode 速度)
output tok/s ≈ 并发数 / TPOT
TTFT ≈ prefill 时间（受 input_len 影响）
ITL P99 ≈ TPOT + 调度抖动
```

### 4. request-rate=inf 的含义

所有请求在 t=0 同时发出，用 `--max-concurrency` 限制服务端并发处理上限。这测的是"饱和吞吐"，不是"真实用户负载"。

真实用户负载用 `--request-rate <数值>` 模拟泊松到达。

---

## 八、放到当前项目里看

### 1. 当前配置的瓶颈分析

从 benchmark 数据看：

| 并发 | output tok/s | 是否线性扩展 | 判断 |
|---|---|---|---|
| 1 | 147 | 基准 | - |
| 4 | 501 | 3.4× | 近线性 |
| 8 | 1013 | 6.9× | 近线性 |
| 16 | 1798 | 12.2× | 增长放缓 |

并发 16 时增长放缓，但 GPU 算力未饱和，说明：

- `max-num-seqs=16` 是调度上限。
- 调高后可能继续线性扩展到 32 或 64。
- 最终会撞到 KV Cache 显存上限或 TPOT 恶化。

### 2. 推荐调优实验

```bash
# 实验 1：调高 max-num-seqs
--max-num-seqs 32

# 实验 2：开 prefix caching（opencode 场景）
--enable-prefix-caching

# 实验 3：显式开 chunked prefill
--enable-chunked-prefill
--max-num-batched-tokens 8192

# 实验 4：组合
--max-num-seqs 32 \
--enable-prefix-caching \
--enable-chunked-prefill
```

### 3. 监控点

调优时要监控：

- 吞吐（output tok/s）是否提升。
- TPOT 是否恶化（batch 大了每步变慢）。
- 显存是否够（KV Cache 是否 OOM）。
- preempt 是否频繁（显存不足的信号）。
- 长尾延迟（ITL P99）是否可接受。

---

## 总结

| 概念 | 作用 | 项目里的体现 |
|---|---|---|
| Scheduler | 决定每步算哪些请求 | continuous batching |
| KV Cache Manager | 分配/回收 block | PagedAttention 基础 |
| Worker | GPU 执行计算 | TP=2 两个 worker |
| max-num-seqs | batch 最大请求数 | 16，是当前吞吐瓶颈 |
| max-num-batched-tokens | 单次 forward token 上限 | 影响 chunked prefill |
| gpu-memory-utilization | 显存使用比例 | 0.90 |
| chunked prefill | prefill 切块混合 decode | 降 TTFT 抖动 |
| preemption | 显存不足时换出请求 | 要避免 |
| prefix caching | 复用相同前缀 KV | opencode 场景可加速 |
| bench serve | 在线 benchmark 工具 | 已用于 256K 测试 |
