# 张量并行、通信与多卡部署

## 一句话理解

张量并行（TP）把一个模型的权重切到多张卡上，每张卡只算一部分，每层结束用 all-reduce 把结果合并；通信带宽是 TP 效率的关键瓶颈，所以 NVLink 比 PCIe 快很多，`CUDA_VISIBLE_DEVICES` 控制容器看到哪些卡。

---

## 一、为什么需要多卡并行

单个 27B 模型权重约 27GB（FP8），加上 KV Cache、激活值、框架开销，单张 80GB H100 也可能吃紧，更别说 70B、175B 甚至更大的模型。

多卡并行的目标：

1. **装得下**：把模型切到多张卡，每张卡只放一部分权重。
2. **跑得快**：多卡同时算，减少单卡计算时间。
3. **支持更大 batch**：更多卡有更多显存放 KV Cache。

并行策略主要有四种：TP、DP、PP、EP。当前项目只用 TP，但要理解它们各自适合什么场景。

---

## 二、四种并行策略对比

### 1. 数据并行（DP，Data Parallelism）

思路：每张卡放一份**完整模型**，各自处理不同的请求 batch，互不通信（除了梯度同步，推理时不需要）。

```text
GPU 0: 完整模型 + batch A
GPU 1: 完整模型 + batch B
GPU 2: 完整模型 + batch C
GPU 3: 完整模型 + batch D
```

- 优点：无通信开销，吞吐线性扩展。
- 缺点：每张卡都要装下完整模型，单卡显存要够。
- 适合：模型能放进单卡，想提升吞吐。

### 2. 张量并行（TP，Tensor Parallelism）

思路：把**每一层的权重**切到多张卡，每张卡算这一层的一部分，层与层之间用 all-reduce 合并。

```text
GPU 0: 每层权重的左半边
GPU 1: 每层权重的右半边
每层结束：all-reduce 合并两半结果
```

- 优点：能放下单卡装不下的模型；层内并行，延迟低。
- 缺点：每层都要通信，通信带宽敏感。
- 适合：单卡装不下，或想降低单请求延迟。

### 3. 流水线并行（PP，Pipeline Parallelism）

思路：把模型**按层切**，每张卡负责连续的几层，请求像流水线一样流过去。

```text
GPU 0: layer 0-15
GPU 1: layer 16-31
GPU 2: layer 32-47
GPU 3: layer 48-63
请求: GPU 0 → GPU 1 → GPU 2 → GPU 3
```

- 优点：通信量小（只传激活值，不传权重）。
- 缺点：有 bubble（流水线气泡），部分卡在等前一层结果。
- 适合：层数很多的超大模型，跨节点部署。

### 4. 专家并行（EP，Expert Parallelism）

思路：针对 MoE 模型，把**不同的专家**放到不同卡上，token 路由到对应专家所在的卡。

```text
GPU 0: expert 0-63
GPU 1: expert 64-127
GPU 2: expert 128-191
GPU 3: expert 192-255
```

- 优点：MoE 模型专家多，天然适合切分。
- 缺点：token 路由有 all-to-all 通信，负载均衡难。
- 适合：MoE 模型（DeepSeek、Agents-A1）。

### 5. 当前项目为什么选 TP

| 方案 | 是否可行 | 原因 |
|---|---|---|
| DP | ❌ | 单卡 80G 放不下 27B + 256K KV Cache |
| TP | ✅ | 权重切到 2 卡，每卡 40G 权重，剩 40G 给 KV Cache |
| PP | ⚠️ | 64 层切 2 段可行，但延迟比 TP 高，且 vLLM 对 PP 支持不如 TP 成熟 |
| EP | ❌ | Qwen3.6-27B 是稠密模型，无专家 |

双模型并行场景：4 卡分成两组，每组 2 卡跑一个模型，两个模型互不干扰。

---

## 三、TP 的切分细节

### 1. 列并行（Column Parallel）

把权重矩阵按列切：

```text
W = [W_1 | W_2]    # 按列切成两块

GPU 0 算: Y_1 = X · W_1
GPU 1 算: Y_2 = X · W_2
```

每个 GPU 拿到**完整的输入 X**，乘以**部分权重**，得到**部分输出**。

特点：

- 输入 X 在两张卡上是一样的（要广播或一开始就复制）。
- 输出 Y_1、Y_2 是按特征维度切的，可以直接拼成完整 Y，也可以分别往下算。

### 2. 行并行（Row Parallel）

把权重矩阵按行切：

```text
W = [W_1]    # 按行切成两块
    [W_2]

GPU 0 算: Y_1 = X_1 · W_1   # X 也按列切
GPU 1 算: Y_2 = X_2 · W_2
Y = Y_1 + Y_2               # all-reduce 求和
```

每个 GPU 拿到**部分输入 X**，乘以**对应部分权重**，得到**部分输出**，最后 all-reduce 求和。

特点：

- 输入 X 要按列切（对应 W 按行切）。
- 输出要 all-reduce 求和（通信）。

### 3. Transformer 层里的混合切分

一个 Transformer 层有 attention 和 MLP 两部分，TP 采用**列并行 + 行并行组合**，让每层只需一次 all-reduce：

```text
Attention:
  QKV 投影: 列并行（W_q, W_k, W_v 按列切，每卡算部分 head）
  Output 投影: 行并行（W_o 按行切，all-reduce 合并）
  
MLP:
  Gate/Up 投影: 列并行（W_gate, W_up 按列切）
  Down 投影: 行并行（W_down 按行切，all-reduce 合并）
```

一层里：

```text
输入 X (每卡都有完整副本)
  → QKV 列并行：每卡算自己的 head，无通信
  → attention 计算：每卡独立
  → O 投影行并行：all-reduce 合并  ← 第一次通信
  → MLP gate/up 列并行：每卡独立
  → MLP down 行并行：all-reduce 合并  ← 第二次通信
```

每层 2 次 all-reduce。64 层就是 128 次。这就是 TP 对通信带宽敏感的原因。

### 4. GQA 与 TP 切分

attention 的 head 要能被 TP size 整除：

```text
Qwen3.6-27B:
  num_attention_heads = 40
  num_kv_heads = 8

TP=2: query head 40/2=20, kv head 8/2=4 ✓
TP=4: query head 40/4=10, kv head 8/4=2 ✓
TP=8: query head 40/8=5, kv head 8/8=1 (退化为 MQA)
```

如果 head 数不能整除 TP size，vLLM 会报错或自动调整。

---

## 四、All-Reduce 通信

### 1. All-Reduce 是什么

```text
输入: GPU 0 有 A, GPU 1 有 B
输出: 所有 GPU 都有 A+B
```

TP 每层结束都要 all-reduce 把各卡的部分结果求和，得到完整输出。

### 2. All-Reduce 的实现方式

| 方法 | 原理 | 特点 |
|---|---|---|
| Ring All-Reduce | 卡排成环，每卡向邻居传数据，绕一圈 | 通信量与卡数无关，适合大规模 |
| Tree All-Reduce | 树形归约再广播 | 延迟低，适合小数据 |
| NCCL | NVIDIA 官方通信库，自动选最优算法 | vLLM 默认用 NCCL |

### 3. custom all-reduce 与 driver 兼容

vLLM 有一个 `--disable-custom-all-reduce` 选项。当前项目 NVFP4 配置里开了它，原因是：

- custom all-reduce 是 vLLM 自己实现的优化路径，绕过 NCCL，用 CUDA kernel 直接通信。
- 它依赖较新的 driver 功能。
- H100 服务器 driver 550.144.03 不支持，会报错或性能更差。
- 禁用后回退到标准 NCCL，稳定但略慢。

### 4. 通信量估算

每层 all-reduce 传输的数据量 = hidden_size × batch_size × seq_len × dtype_size。

以 Qwen3.6-27B 为例（hidden_size ≈ 5120，bf16）：

```text
单层单 token all-reduce 数据量 = 5120 × 2 = 10 KB
每层 2 次 all-reduce，64 层：
单 token 总通信量 = 10 KB × 2 × 64 = 1.28 MB
```

看起来不大，但 TP=2 时这些数据要在两张卡之间来回传，带宽不够就成为瓶颈。

---

## 五、通信带宽：NVLink vs PCIe

### 1. 带宽差异

| 互联方式 | 单向带宽 | 典型延迟 |
|---|---|---|
| PCIe 4.0 x16 | ~32 GB/s | ~5 μs |
| PCIe 5.0 x16 | ~64 GB/s | ~3 μs |
| NVLink 3（H100） | ~450 GB/s（单向），900 GB/s（双向） | ~1 μs |

H100 的 NVLink 带宽是 PCIe 的 10 倍以上。TP 对带宽极敏感，有 NVLink 和没 NVLink 性能差距很大。

### 2. 怎么查机器有没有 NVLink

```bash
# 查 NVLink 状态
nvidia-smi nvlink -s

# 查 NVLink 拓扑
nvidia-smi topo -m
```

`nvidia-smi topo -m` 输出示例：

```text
        GPU0  GPU1  GPU2  GPU3
GPU0     X    NV2    NV2    NV2
GPU1    NV2    X    NV2    NV2
GPU2    NV2   NV2     X    NV2
GPU3    NV2   NV2    NV2    X
```

- `NV2`：通过 NVLink 连接（最好）
- `PIX`：通过 PCIe 交换机连接（次之）
- `NODE`：同节点但不同 PCIe 交换机
- `SYS`：跨节点（最差，TP 一般不用）

如果输出是 `SYS` 或 `NODE`，TP 性能会打折，这时 DP 或 PP 可能更合适。

### 3. 当前项目的盲点

AGENTS.md 和 PROJECT_LOG.md 都没记录这台 H100 的 NVLink 拓扑。这是个待补的检查项：

```bash
ssh root@10.16.11.24 'nvidia-smi topo -m'
```

如果没 NVLink，TP=2 的 all-reduce 走 PCIe，吞吐会受影响，benchmark 里 TPOT 偏高可能部分来自这里。

---

## 六、CUDA_VISIBLE_DEVICES 原理

### 1. 作用

`CUDA_VISIBLE_DEVICES` 是环境变量，控制进程能看到哪些 GPU。

```bash
# 只让进程看到 GPU 0 和 1
CUDA_VISIBLE_DEVICES=0,1 python script.py

# 进程内部看到 GPU 编号会被重新映射
# 物理 GPU 0 → 逻辑 GPU 0
# 物理 GPU 1 → 逻辑 GPU 1
# 物理 GPU 2,3 不可见
```

### 2. 重新映射机制

```text
物理 GPU: 0  1  2  3
CUDA_VISIBLE_DEVICES=2,3
进程看到: 
  逻辑 GPU 0 = 物理 GPU 2
  逻辑 GPU 1 = 物理 GPU 3
```

进程内部永远从 0 开始编号，不知道物理编号是什么。这就是为什么 `--tensor-parallel-size 2` 对应"容器内可见的 GPU 数"，而不是物理 GPU 编号。

### 3. Docker 里的配合

```bash
# --gpus all 让容器能访问所有 GPU
# CUDA_VISIBLE_DEVICES 限制实际用哪些
docker run --gpus all --env CUDA_VISIBLE_DEVICES=0,1 ...
```

当前项目双模型并行：

```text
Qwen 容器:   --gpus all + CUDA_VISIBLE_DEVICES=0,1 → 容器内看到 2 张卡
Agents 容器: --gpus all + CUDA_VISIBLE_DEVICES=2,3 → 容器内看到 2 张卡
```

两个容器物理上隔离，各自 TP=2，互不干扰。

### 4. 常见坑

- `CUDA_VISIBLE_DEVICES` 写错（如 `0,1,2,3` 给一个 TP=2 容器），vLLM 会用 4 张卡但 TP=2 只用 2 张，其余空闲但被占用。
- 编号从 0 开始，不是 1。
- Docker `--gpus '"device=0,1"'` 也能限制，但和 `CUDA_VISIBLE_DEVICES` 叠加时容易混淆，项目统一用后者。

---

## 七、TP 的性能影响

### 1. TP 降低单请求延迟

单卡 decode：每步要把全部权重读一遍。TP=2 时每卡只读一半权重，访存减半，单步更快。

```text
单卡:  TPOT ∝ 权重读取时间 = 权重大小 / HBM带宽
TP=2:  TPOT ∝ (权重大小/2) / HBM带宽 + all-reduce 时间
```

当 all-reduce 时间 < 节省的权重读取时间，TP 就能加速。NVLink 下通常成立，PCIe 下不一定。

### 2. TP 不一定提升吞吐

吞吐 = 并发数 × 单请求速度 / 延迟。TP 提升单请求速度，但卡数多了，能同时服务的请求数受总显存限制。

```text
单卡:  1 张卡所有显存给 KV Cache，能放 N 个请求
TP=2:  2 张卡，权重省了一半但总显存也翻倍，能放约 2N 个请求
       但吞吐 = 2N × (单请求速度)
```

如果 TP 通信开销大，单请求速度提升不到 2 倍，吞吐可能不如 DP（2 份单卡模型各跑各的）。这就是为什么"模型能放进单卡时，DP 通常比 TP 吞吐高"。

### 3. 当前项目的取舍

| 方案 | 单请求延迟 | 吞吐 | 能否跑双模型 |
|---|---|---|---|
| TP=4 单模型 | 最低 | 最高 | ❌ 只能跑一个 |
| TP=2 × 2 双模型 | 中等 | 中等 | ✅ 两个模型同时服务 |
| TP=1 × 4 四模型 | 最高 | 最低 | ✅ 但 27B 单卡装不下 |

双模型并行（TP=2 × 2）是为了"同时提供两个模型"，牺牲单模型峰值换取服务多样性，适合 opencode 这类需要大小模型配合的场景。

---

## 八、NCCL 与通信库

### 1. NCCL（NVIDIA Collective Communications Library）

NVIDIA 官方的多卡通信库，TP all-reduce 默认走 NCCL。

特点：

- 针对 NVIDIA GPU 和 NVLink 优化。
- 自动选择 Ring/Tree 等算法。
- vLLM、PyTorch DDP 默认用 NCCL。

### 2. 通信组（communicator）

NCCL 用 communicator 管理一组参与通信的 GPU。TP=2 时，两个模型各自有一个 communicator：

```text
Qwen communicator: GPU 0, 1
Agents communicator: GPU 2, 3
```

两组互不干扰，可以并行通信。

### 3. 通信调试

```bash
# 查看 NCCL 通信日志
NCCL_DEBUG=INFO vllm serve ...

# 关键信息:
# - 使用的传输方式（NVLink/PCIe/网络）
# - Ring 拓扑
# - 通信耗时
```

如果 benchmark 发现 TP 比预期慢，开 NCCL_DEBUG 看是不是走了 PCIe 而不是 NVLink。

---

## 九、放到当前项目里看

### 1. 双模型并行的通信隔离

```text
GPU 0 ⇄ GPU 1 (NVLink)  → Qwen TP=2 all-reduce
GPU 2 ⇄ GPU 3 (NVLink)  → Agents TP=2 all-reduce
```

两组通信互不干扰，各自走自己的 NVLink 通道。如果 GPU 0-1 之间是 NVLink，GPU 2-3 之间也是 NVLink，性能最佳。但如果 GPU 1-2 之间才是 NVLink，GPU 0-1 之间是 PCIe，TP=2 的 Qwen 性能会差一截。

### 2. 为什么不用 TP=4

TP=4 单模型：

- 单请求延迟最低（4 卡分权重）。
- 吞吐最高（4 卡显存全给一个模型）。
- 但同时只能服务一个模型。

双模型并行 TP=2 × 2：

- 单请求延迟中等（2 卡分权重）。
- 吞吐中等。
- 能同时提供两个模型。

这是**延迟 vs 灵活性**的取舍。opencode 场景需要大模型（Qwen）做复杂任务 + 小模型（Agents）做快速补全，双模型并行更合适。

### 3. 待检查项

- [ ] H100 服务器的 NVLink 拓扑（`nvidia-smi topo -m`）
- [ ] TP=2 的 all-reduce 实际走 NVLink 还是 PCIe
- [ ] 两个模型并行时，两组通信是否真的互不干扰

---

## 总结

| 概念 | 作用 | 项目里的体现 |
|---|---|---|
| TP | 权重切到多卡，每卡算一部分 | TP=2，每卡一半权重 |
| DP | 多份模型各跑各的 | 不适用（单卡装不下） |
| PP | 按层切，流水线 | 不适用（延迟高） |
| EP | MoE 专家切到多卡 | 未来 MoE 模型可用 |
| 列并行/行并行 | TP 的切分方式，每层 2 次 all-reduce | 64 层 × 2 = 128 次 all-reduce |
| NVLink | 高带宽互联，TP 性能关键 | 待检查服务器拓扑 |
| CUDA_VISIBLE_DEVICES | 容器内 GPU 隔离 | 双模型并行的基础 |
| NCCL | 通信库，执行 all-reduce | vLLM 默认用 NCCL |
