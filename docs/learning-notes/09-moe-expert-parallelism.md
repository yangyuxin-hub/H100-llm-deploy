# MoE（Mixture of Experts）与专家并行

## 一句话理解

MoE 模型把一个大的 FFN 换成多个小 FFN（专家），每个 token 只激活其中几个，用"总参数大、激活参数小"换取能力与速度的平衡；专家并行（EP）把不同专家放到不同卡上，让 token 路由到对应专家所在的卡计算。

---

## 一、为什么有 MoE

### 1. 稠密模型的瓶颈

传统 Transformer 是稠密模型（dense）：每个 token 都要过所有参数。

```text
token → attention → FFN（全部参数参与） → 下一层
```

问题：

- 模型越强，参数越多，计算量线性增长。
- 175B 模型的 FFN 计算量是 7B 的 25 倍，推理慢 25 倍。
- 但很多参数对单个 token 贡献有限（冗余）。

### 2. MoE 的思路

把 FFN 拆成 N 个小 FFN（专家），每个 token 只选 K 个专家计算：

```text
token → attention → router 选择 K 个专家 → 只算这 K 个专家 → 下一层
```

- 总参数：N 个专家的参数总和（很大）。
- 激活参数：每个 token 实际算的 K 个专家参数（小）。
- 能力来自"专家分工"，不同 token 走不同路径。

### 3. 稠密 vs MoE 对比

| 维度 | 稠密模型 | MoE 模型 |
|---|---|---|
| 总参数 | P | N × P_expert |
| 激活参数 | P | K × P_expert |
| 计算量 | ∝ P | ∝ K × P_expert |
| 显存 | ∝ P | ∝ N × P_expert |
| 能力 | 受限于 P | 受限于 N × P_expert |

关键：**MoE 用更多总参数换更强能力，但单 token 计算量只和激活参数有关**。

---

## 二、MoE 的工作机制

### 1. Router（路由器）

每个 MoE 层有一个 router（门控网络），决定 token 去哪个专家：

```text
router_output = softmax(W_router · token)   # N 维概率
top_k_experts = topk(router_output, k)       # 选概率最高的 K 个
```

- W_router 是一个小矩阵（`hidden_size × N`）。
- 输出是 N 个专家的权重。
- 选 top-K 个，通常 K 远小于 N。

### 2. 加权计算

```text
for expert in top_k_experts:
    expert_output = expert(token)
    final_output += router_weight[expert] × expert_output
```

被选中的 K 个专家各算一遍，按 router 权重加权求和。

### 3. load balancing（负载均衡）

如果所有 token 都选同一个专家，其他专家闲置，MoE 退化。所以训练时要加 load balancing loss，鼓励 token 均匀分布到专家。

推理时仍有不均衡问题：

- 某些热门专家被频繁选中，计算排队。
- 某些冷门专家几乎不用，显存浪费。

### 4. 当前项目的 MoE 模型

| 模型 | 总参数 | 激活参数 | 专家数 N | 激活数 K | 层数 |
|---|---|---|---|---|---|
| Agents-A1-FP8 | ~200B+ | ~13B | 256 | 8 | 40 |
| DeepSeek-V4-Flash-DSpark | 284B | 13B | 256 | 6 | 43 |
| Qwen3.6-27B-FP8 | 27B（稠密） | 27B | - | - | 64 |

对比：

- Qwen3.6-27B 是稠密模型，每个 token 都过全部 27B 参数。
- Agents-A1 总参数大得多，但激活参数只有 13B，decode 时访存量小。
- 这就是为什么 MoE 模型 decode 阶段更快（每 token 实际计算量小）。

---

## 三、MoE 的显存特点

### 1. 总参数常驻显存

虽然每个 token 只激活 K 个专家，但**所有 N 个专家的权重都要常驻显存**（因为不知道下个 token 会选哪个）。

```text
显存 ∝ N × P_expert（总参数）
计算 ∝ K × P_expert（激活参数）
```

Agents-A1：总参数 200B+（FP8 后约 36GB 权重），但激活参数只有 13B。

### 2. 与稠密模型的对比

| 模型 | 权重显存（FP8） | 单 token decode 计算量 |
|---|---|---|
| Qwen3.6-27B（稠密） | 27 GB | 27B 参数 |
| Agents-A1（MoE） | 36 GB | 13B 参数（8/256 激活） |

Agents-A1 权重比 Qwen3.6 大 33%，但 decode 计算量只有一半。这就是 MoE 的"用显存换计算"。

### 3. 对部署的影响

- **显存压力大**：MoE 模型总参数大，TP 切分后每卡仍要放 N/TP 个专家。
- **decode 快**：激活参数小，访存瓶颈下 decode 吞吐高。
- **prefill 慢**：prefill 时所有 token 都要算，router 开销 + 专家调度开销。
- **长上下文**：KV Cache 和稠密模型一样（attention 部分不变），但专家调度更复杂。

当前项目 Agents-A1 每卡 75.3GB 显存（TP=2），比 Qwen3.6 的 71.6GB 高，就是因为总参数更大。

---

## 四、专家并行（EP）

### 1. EP 的思路

TP 把每个专家的权重切到多卡，EP 把**不同专家整块放到不同卡**：

```text
TP=2（每个专家切两半）:
  GPU 0: expert_0 左半, expert_1 左半, ..., expert_255 左半
  GPU 1: expert_0 右半, expert_1 右半, ..., expert_255 右半
  → 每个专家都算，但只算一半

EP=2（每个专家整块放一张卡）:
  GPU 0: expert_0 到 expert_127（完整）
  GPU 1: expert_128 到 expert_255（完整）
  → 每个专家只在一台卡上，token 要路由到对应卡
```

### 2. EP 的工作流程

```text
1. token 在所有卡上都有副本
2. router 计算每个 token 去哪个专家
3. 根据 token 要去的专家，发到对应卡（all-to-all 通信）
4. 每张卡只算自己负责的专家
5. 结果发回原卡（又一次 all-to-all）
6. 加权求和
```

### 3. EP vs TP

| 维度 | TP | EP |
|---|---|---|
| 切分对象 | 每个专家的权重 | 不同专家整体 |
| 通信 | all-reduce（每层 2 次） | all-to-all（每层 2 次） |
| 通信量 | hidden_size × batch × seq | token 数 × hidden_size |
| 适合 | 专家数少 | 专家数多 |

MoE 模型专家多（256 个），EP 更合适：

- TP=4：每卡 64 个专家，但每个专家都要 all-reduce，通信多。
- EP=4：每卡 64 个专家，token 只发给对应卡，通信更高效。

### 4. EP + TP 混合

大规模 MoE 部署通常 EP + TP 混合：

```text
8 卡部署 256 专家 MoE:
  EP=4, TP=2
  GPU 0,1: expert 0-63（TP=2 切分）
  GPU 2,3: expert 64-127（TP=2 切分）
  GPU 4,5: expert 128-191（TP=2 切分）
  GPU 6,7: expert 192-255（TP=2 切分）
```

### 5. 当前项目为什么用 TP 不用 EP

当前项目 Agents-A1 用 TP=2，不用 EP，原因：

1. **vLLM 对 EP 支持较新**：当前 v0.24.0 的 EP 支持不如 TP 成熟。
2. **专家数多但 TP=2 仍可行**：256 专家 TP=2，每卡 128 个专家，显存放得下。
3. **all-to-all 通信复杂**：EP 要 all-to-all，对 NVLink 拓扑要求更高。
4. **双模型并行已占满 4 卡**：Agents-A1 只有 2 卡可用，EP=2 收益有限。

未来如果单独部署 MoE 模型用 4 卡，EP=2 + TP=2 可能比纯 TP=4 更优。

---

## 五、MoE 的 decode 优势与 MTP

### 1. MoE decode 计算密度低

稠密模型 decode：每步算全部参数，计算密集。

MoE decode：每步只算 K/N 个专家，计算稀疏。

```text
Qwen3.6-27B decode: 每步 27B 参数计算
Agents-A1 decode:   每步 13B 参数计算（8/256 激活）
```

### 2. 为什么 MoE 更适合 MTP

MTP（投机解码）用 draft 模型一次预测多个 token，再验证。加速效果取决于：

- decode 计算密度低 → 单步快 → draft 生成快。
- 验证阶段可以 batch 处理多个候选 token，摊销开销。

MoE 模型 decode 计算密度低，MTP 加速效果更明显：

- 稠密模型：decode 已经算很多，MTP 的相对加速有限。
- MoE 模型：decode 算得少，MTP draft 快，加速比高。

当前项目笔记里提到"MoE 模型 decode 计算密度低，MTP 加速效果更明显"，就是这个原因。

### 3. MoE 的 prefill 瓶颈

MoE prefill 时所有 token 都要路由 + 计算：

- router 计算：N × hidden_size 矩阵乘。
- 专家计算：K 个专家 × batch 个 token。
- all-to-all 通信（如果用 EP）。

所以 MoE 模型 prefill 可能比同等激活参数的稠密模型慢，因为 router 和调度开销。

---

## 六、DeepSeek-V4 的 FP4 专家

### 1. expert_dtype: fp4

DeepSeek-V4-Flash-DSpark 的 config 里有 `expert_dtype: fp4`，专家权重用 4-bit 量化。

这与 NVFP4 类似但不同：

| 格式 | 来源 | 特点 |
|---|---|---|
| NVFP4 | NVIDIA ModelOpt | E2M1 + per-group scale，H100 要反量化 |
| DeepSeek FP4 | DeepSeek 自研 | DSpark 的一部分，有自己的量化方案 |

### 2. DSpark 是什么

DSpark 不是新模型，是 DeepSeek-V4-Flash 的**推测解码模块**：

- 类似 MTP，但是 DeepSeek 自己的实现。
- 与模型权重一起发布（`dspark_*` 字段）。
- 需要特定推理框架支持。

### 3. 部署挑战

- `DS_HF_OVERRIDES` 要处理 FP4 expert 配置。
- vLLM 对 DeepSeek FP4 的支持程度待验证。
- DSpark 推测解码路径与 MTP 不同，可能需要专门适配。

当前项目 DeepSeek-V4-Flash-DSpark 尚未测试，属于待探索项。

---

## 七、MoE 推理的工程问题

### 1. 专家负载不均

某些专家被频繁选中，其他闲置。导致：

- 某些卡（EP 下）计算排队，其他卡空闲。
- 显存占用不均（热门专家的激活值多）。

解决：

- 训练时加 load balancing loss（已做）。
- 推理时动态调整专家分配（复杂，少用）。
- EP 下用 expert replication（复制热门专家到多卡）。

### 2. 专家交换开销

如果专家数多到单卡放不下，要动态加载/卸载专家（expert offloading）：

- 把不常用专家放 CPU 内存。
- 需要时换到 GPU。
- 有 PCIe 传输开销。

当前项目 256 专家 TP=2 下每卡 128 个专家，36GB 权重能放进 80GB H100，不需要 offloading。

### 3. router 的精度

router 是小矩阵乘，但对精度敏感（选错专家影响大）。通常 router 保持高精度（BF16/FP32），不量化。

### 4. KV Cache 不受 MoE 影响

MoE 只改 FFN，attention 部分不变。所以：

- KV Cache 大小计算和稠密模型一样（用 num_kv_heads、num_layers 等）。
- KV Cache 量化（fp8）同样适用。
- 长上下文机制（RoPE、FlashAttention）同样适用。

---

## 八、放到当前项目里看

### 1. 双模型并行的 MoE vs 稠密

| 维度 | Qwen3.6-27B（稠密） | Agents-A1（MoE） |
|---|---|---|
| 总参数 | 27B | ~200B+ |
| 激活参数 | 27B | ~13B |
| 权重显存（FP8, 每卡 TP=2） | 13.5 GB | 18 GB |
| 总显存（每卡） | 71.6 GB | 75.3 GB |
| decode 计算量 | 27B | 13B（更快） |
| prefill 复杂度 | 标准 | 加 router 开销 |
| MTP 加速效果 | 中等 | 更明显 |

### 2. 为什么双模型搭配合理

- Qwen3.6-27B（稠密）：全参数计算，能力强，适合复杂任务。
- Agents-A1（MoE）：激活参数小，decode 快，适合高吞吐补全。

opencode 场景：

- 复杂代码生成 → Qwen3.6（质量优先）。
- 快速补全/轻量任务 → Agents-A1（速度优先）。

### 3. 显存账对比

Qwen3.6-27B-FP8（TP=2，每卡）：

```text
权重: 13.5 GB
KV Cache (256K, fp8): 动态
框架 + CUDA graph + 激活: ~10 GB
总占用: 71.6 GB / 80 GB
```

Agents-A1-FP8（TP=2，每卡）：

```text
权重: 18 GB（256 专家 FP8）
KV Cache (256K, fp8): 动态
框架 + CUDA graph + 激活: ~12 GB
总占用: 75.3 GB / 80 GB
```

Agents-A1 显存更紧，因为总参数大。但激活参数小，decode 计算量低。

---

## 总结

| 概念 | 作用 | 项目里的体现 |
|---|---|---|
| MoE | 总参数大、激活参数小 | Agents-A1、DeepSeek-V4 |
| Router | 选 top-K 专家 | 256 选 8（Agents-A1） |
| 总参数 vs 激活参数 | 显存 vs 计算量 | 200B 总参，13B 激活 |
| EP | 专家分到不同卡 | 当前未用，未来可探索 |
| load balancing | 防止专家不均 | 训练时处理 |
| MoE + MTP | decode 快，MTP 加速更明显 | Agents-A1 启用 MTP |
| DeepSeek FP4 | 自研 4-bit 量化 | 待测试 |
| DSpark | DeepSeek 推测解码 | 待测试 |
