# Attention Backend 与 CUDA Graph

## 一句话理解

Attention backend 是 vLLM 执行 attention 计算的具体 kernel 实现，不同 backend 在速度、显存、兼容性上各有取舍；CUDA graph 把一串 GPU kernel 调用录制成一张"图"一次性提交，消除 CPU→GPU 的 launch 开销；torch.compile/inductor 在编译期做算子融合优化，但可能改变数值行为，这就是 NVFP4 项目里 NaN 的根源。

---

## 一、Attention Backend 是什么

### 1. 为什么有多个 backend

Attention 是大模型推理里最复杂的计算，有 self-attention、cross-attention、GQA、MQA、PagedAttention、FlashAttention 等各种变体。每种变体在不同 GPU、不同精度、不同序列长度下，最优 kernel 实现不一样。

vLLM 把 attention 抽象成 backend 接口，让不同实现可插拔：

```text
vLLM 调度器 → attention backend → 具体 kernel（FlashAttention/Triton/FlashInfer）
```

### 2. 选择依据

| 维度 | 考虑点 |
|---|---|
| GPU 架构 | H100 用 FlashAttention-3，A100 用 FlashAttention-2 |
| 精度 | FP8 路径要 backend 支持 FP8 |
| 模型结构 | 普通 attention vs GDN/线性注意力 |
| 序列长度 | 长 prefill 要 tiling，decode 要访存优化 |
| PagedAttention 支持 | 必须支持 block table 跳跃读取 |

---

## 二、主流 Attention Backend 详解

### 1. FlashAttention

**核心思想**：用 tiling 把 attention 计算的显存从 O(n²) 降到 O(n)，不把完整注意力矩阵写回 HBM。

**版本演进**：

| 版本 | 特点 | 适用 |
|---|---|---|
| FlashAttention v1 | 首次提出 tiling + online softmax | A100 |
| FlashAttention v2 | 更好的并行度，减少非矩阵乘法计算 | A100/A10 |
| FlashAttention v3 | 针对 Hopper（H100）优化，用 TMA、异步 copy、FP8 | H100 |

**H100 上的优势**：

- 用 TMA（Tensor Memory Access）硬件单元做异步访存。
- Warp Group Specialization：不同 warp group 分别做 GEMM 和 softmax，重叠执行。
- FP8 支持：v3 原生支持 FP8 Tensor Core。

vLLM 里的 backend 名：

```text
--attention-backend FLASH_ATTN
```

### 2. FlashInfer

NVIDIA/CMU 开发的高性能推理库，专门为 LLM 推理优化。

**特点**：

- 针对推理场景（decode batch、prefill batch、混合）专门优化。
- 支持 PagedAttention 原生。
- 支持 KV Cache 量化（fp8/int4）。
- 针对 Hopper 有专门 kernel。

**当前项目的坑**：

FlashInfer 的 GDN（Gated DeltaNet）prefill kernel 在 H100 上有数值精度问题（flashinfer issue #2490），gating space 不匹配导致全 NaN。这就是为什么 NVFP4 项目里要强制 `--gdn-prefill-backend triton`，绕开 FlashInfer 的 GDN 路径。

vLLM 里的 backend 名：

```text
--attention-backend FLASHINFER
```

### 3. Triton

Triton 是 OpenAI 开发的 GPU 编程语言/编译器，用 Python 写高性能 kernel，不用 CUDA C。

**特点**：

- 开发快，易调试。
- 性能接近手写 CUDA，但灵活性高。
- vLLM 用 Triton 写了 fallback kernel（如 TRITON_ATTN、triton GDN backend）。
- JIT 编译：第一次运行时编译，有冷启动开销。

**当前项目的使用**：

- `QWEN_ATTENTION_BACKEND=TRITON_ATTN`：用 Triton attention，性能略逊 FlashAttention，但兼容性好。
- `--gdn-prefill-backend triton`：GDN prefill 用 Triton，绕开 FlashInfer 的 NaN bug。

### 4. XFormers

Meta 开发的注意力库，早期 vLLM 默认 backend。

**现状**：

- 性能不如 FlashAttention v2/v3。
- vLLM 新版已不默认用。
- 主要用于老 GPU 或兼容性场景。

### 5. backend 选型决策树

```text
是否 H100？
├── 是 → 是否用 FP8？
│        ├── 是 → FLASH_ATTN (v3, 原生 FP8)
│        └── 否 → FLASH_ATTN (v3, BF16)
├── 否（A100/A10）→ FLASH_ATTN (v2)
└── 特殊模型（GDN/线性注意力）→
     ├── prefill: triton（绕开 FlashInfer bug）
     └── decode: FLASH_ATTN 或 TRITON_ATTN
```

当前项目 Qwen3.6-27B-FP8 用 `FLASH_ATTN`，NVFP4 项目里也是 `FLASH_ATTN` + GDN prefill 用 triton。

---

## 三、GDN（Gated DeltaNet）是什么

### 1. 为什么 Qwen3.6 有 GDN

Qwen3.6 不是纯 Transformer，是**混合架构**：

- 部分 layer 用标准 self-attention（full attention）。
- 部分 layer 用 Gated DeltaNet（线性注意力变体）。

GDN 的特点：

- 线性复杂度 O(n)，不是 O(n²)。
- 适合长上下文。
- 但数值行为和标准 attention 不同，对 kernel 实现敏感。

### 2. 为什么 GDN 容易出 NaN

GDN 里有 gating 机制，涉及 `exp(g)` 操作：

```text
gating = exp(g)   # g 稍大就溢出
```

- 如果 g 的数值范围没控制好，`exp(g)` 容易溢出到 inf。
- inf 传播开后变成 NaN。
- FlashInfer 的 GDN kernel 在 Hopper 上 gating space 处理有 bug，导致全 NaN。

vLLM 的 workaround：用 `torch.exp(g)` 替代 kernel 内部的 exp，但不够彻底。彻底解法是用 Triton 重写 GDN prefill kernel（`--gdn-prefill-backend triton`）。

### 3. GDN 与 attention backend 的关系

GDN 不是标准 attention，所以 FlashAttention backend 不能直接跑。vLLM 为 GDN 单独做了 backend 接口：

```text
--gdn-prefill-backend triton      # prefill 阶段的 GDN
--gdn-prefill-backend flashinfer  # 默认，但有 NaN bug
```

这就是为什么 NVFP4 项目的配置里同时有 `--attention-backend FLASH_ATTN`（标准 attention 层用）和 `--gdn-prefill-backend triton`（GDN 层用）。

---

## 四、CUDA Graph 原理

### 1. 普通 kernel launch 的开销

正常执行一次 forward：

```text
CPU: 发出 kernel A → 发出 kernel B → 发出 kernel C → ...
GPU:                       执行 A → 执行 B → 执行 C
```

每个 kernel 都要 CPU 发起一次，有 **launch 开销**（约 5-10 μs/kernel）。decode 阶段每步几百个 kernel，光 launch 开销就几十毫秒，严重拖慢 TPOT。

```text
decode 一步: 200 个 kernel × 10 μs = 2 ms 纯 launch 开销
如果实际计算只要 3 ms，launch 占了 40%
```

### 2. CUDA Graph 的做法

把一串 kernel 录制成"图"，之后一次提交整张图：

```text
录制阶段（capture）:
  CPU: 执行 forward，记录所有 kernel → 生成 graph

执行阶段（replay）:
  CPU: 一次调用 cuGraphLaunch
  GPU: 按 graph 依次执行所有 kernel，无需 CPU 介入
```

效果：

- launch 开销从 N×10μs 降到 1×10μs。
- decode 步骤从几毫秒降到几百微秒。
- 对 kernel 多、每步计算量小的 decode 阶段提升巨大。

### 3. 为什么不是所有场景都能用 CUDA Graph

**捕获要求**：

- 所有 kernel 的参数在 capture 时就要确定（地址、shape）。
- 动态 shape（如变长 batch、变长序列）不能直接 capture。

**推理场景的挑战**：

- prefill：输入长度变化，shape 动态，难 capture。
- decode：每步 batch 里请求数变化（continuous batching），shape 也变。

vLLM 的解决办法：**用固定 shape 的"虚拟 batch"capture，运行时用 pointer 重定向**。

### 4. decode-only CUDA Graph

vLLM 的 CUDA graph 模式：

| 模式 | 含义 | 适用 |
|---|---|---|
| `NONE` | 不用 CUDA graph | 调试 |
| `FULL_DECODE_ONLY` | 只对 decode 阶段 capture full graph | decode shape 相对固定 |
| `FULL_AND_PIECEWISE` | decode 用 full graph，prefill 用 piecewise graph | 最优，但依赖 torch.compile |

当前 NVFP4 项目用 `FULL_DECODE_ONLY`：

```bash
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

原因：`FULL_AND_PIECEWISE` 依赖 torch.compile，而 torch.compile 对 NVFP4 + GDN 模型有 NaN bug，不能用。`FULL_DECODE_ONLY` 只 capture decode 的 graph，不依赖 piecewise compile，绕开 bug 同时拿到 decode 加速。

### 5. CUDA Graph 的代价

- **显存开销**：graph 本身要存 kernel 参数和中间 buffer，当前项目约 2.52 GiB。
- **启动时间**：capture 要几秒到几十秒。
- **灵活性下降**：batch size 变化时要重新 capture（vLLM 会 capture 多个 batch size 的 graph pool）。
- **KV Cache 下降**：graph pool 占显存，留给 KV Cache 的空间少。当前项目 KV cache 从 4,407,792 降到 4,155,578 tokens（-5.7%）。

### 6. 实测效果

来自 `03-qwen3nvfp4-optimization.md`：

| 配置 | 512 tokens 单流吞吐 |
|---|---|
| enforce-eager（无 CUDA graph） | 32.4 tok/s |
| FULL_DECODE_ONLY CUDA graph | 162.0 tok/s |

5 倍提升。这就是为什么即使有 5.7% KV cache 代价，也要开 CUDA graph。

---

## 五、torch.compile 与 inductor

### 1. torch.compile 是什么

PyTorch 2.0 引入的 JIT 编译器，把 PyTorch 代码编译成优化后的 kernel。

```python
model = torch.compile(model)  # 一行代码开启
```

vLLM 用 torch.compile 做两件事：

1. **算子融合**：把多个小算子合成一个大 kernel，减少 launch 和访存。
2. **piecewise CUDA graph**：配合 `FULL_AND_PIECEWISE` 模式，prefill 阶段也能用 graph。

### 2. inductor 后端

torch.compile 默认用 inductor 后端，工作流程：

```text
PyTorch 代码 → FX Graph（中间表示）→ inductor 优化 → 生成 Triton/C++ kernel
```

inductor 的优化包括：

- 算子融合（如 `x + y * z` 融合成一个 kernel）。
- 内存规划（减少临时 buffer）。
- 生成 Triton kernel（针对当前 shape 特化）。

### 3. 为什么 inductor 会改变数值行为

算子融合不是无脑合并，会重排计算顺序：

```text
原代码: a = exp(x); b = a + bias; c = matmul(b, W)
融合后: c = fused_exp_add_matmul(x, bias, W)
```

融合后：

- 中间结果 `a`、`b` 不写回 HBM，在寄存器/SRAM 里算。
- 计算顺序可能变（如用 Tensor Core 的矩阵乘法累加顺序和逐元素不同）。
- 精度可能不同（FP32 累加 vs FP16 累加）。

### 4. NVFP4 项目的 NaN 根因

来自 `03-qwen3nvfp4-optimization.md` 的二分实验定位：

```text
torch.compile + inductor 对 NVFP4 + GDN 模型:
  - inductor 融合时改变了 Marlin FP4 或 GDN kernel 的数值行为
  - 导致 logits 产生 NaN
  - 采样器遇到 NaN 退化为输出 token 0 ("!")
```

关键证据：

- 实验 F：`mode=NONE`（禁用 torch.compile）→ 正常 ✓
- 实验 A-E：各种 CUDA graph 配置 + torch.compile → NaN ✗
- 结论：不是 CUDA graph 的问题，是 torch.compile/inductor 的问题。

### 5. 解耦方案

不能完全不用编译优化（会损失 CUDA graph 性能），所以要解耦：

| 组件 | 控制 | 当前配置 |
|---|---|---|
| torch.compile | `TORCH_COMPILE_DISABLE=1` | 禁用（绕 NaN） |
| CUDA graph | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | 保留（decode 加速） |

这样：

- torch.compile 不跑，inductor 不融合，数值行为不变。
- CUDA graph 仍然 capture decode 阶段的现有 kernel，launch 开销消除。
- 代价：prefill 阶段没有 piecewise graph 加速，但 prefill 本来就是计算密集，launch 开销占比小。

---

## 六、mm-encoder-attn-backend 是什么

### 1. 多模态模型的 attention

Qwen3.6-27B 是多模态模型，有 vision encoder。vision encoder 里也有 attention 计算，但和语言模型的 attention 特点不同：

- vision：图像 patch 序列，长度固定，batch 处理。
- language：文本 token，长度变化，PagedAttention。

所以 vLLM 为 vision encoder 单独配置 attention backend：

```bash
--mm-encoder-attn-backend FLASH_ATTN
```

### 2. 当前项目的处理

当前项目用 `--language-model-only` 跳过 vision encoder，所以 `--mm-encoder-attn-backend` 实际不生效。但配置里保留，是为了万一要用 vision 时不报错。

---

## 七、backend 与项目的对应关系

### 1. NVFP4 项目的 backend 配置

```bash
--attention-backend FLASH_ATTN          # 标准 attention 层
--mm-encoder-attn-backend FLASH_ATTN    # vision encoder（跳过但保留配置）
--gdn-prefill-backend triton            # GDN 层 prefill（绕 FlashInfer NaN）
```

三层 backend 分别对应三种 attention 变体：

| 层类型 | backend | 原因 |
|---|---|---|
| 标准 self-attention | FLASH_ATTN | H100 最优，支持 FP8 |
| GDN（线性注意力） | triton | FlashInfer 有 NaN bug，Triton 正确 |
| Vision attention | FLASH_ATTN | 跳过但配置保留 |

### 2. FP8 双模型项目的 backend 配置

Qwen3.6-27B-FP8 和 Agents-A1-FP8 都是标准 Transformer（无 GDN），直接用默认 backend（H100 上自动选 FLASH_ATTN 或 FlashInfer），不需要特殊配置。

### 3. 排查 backend 问题的方法

```bash
# 查看当前生效的 backend
docker logs <container> 2>&1 | grep -i "attention backend"

# 强制指定 backend 做对比
--attention-backend FLASH_ATTN
--attention-backend TRITON_ATTN
--attention-backend FLASHINFER

# 查看 vLLM 支持的 backend
docker exec <container> python3 -c "
from vllm.attention.backends import AttentionBackendManager
print(AttentionBackendManager.list_backends())
"
```

---

## 八、放到当前项目里看

### 1. 为什么 FP8 项目比 NVFP4 简单

| 维度 | FP8 项目 | NVFP4 项目 |
|---|---|---|
| 模型结构 | 标准 Transformer | 混合（attention + GDN） |
| backend 选择 | 默认（FLASH_ATTN） | 要分别配 attention + GDN |
| torch.compile | 可用 | 禁用（NaN bug） |
| CUDA graph | FULL_AND_PIECEWISE（最优） | FULL_DECODE_ONLY（折中） |
| 性能 | 147 tok/s（MTP2） | 162 tok/s（MTP3 + decode graph） |

FP8 项目模型结构简单，没有 GDN，没有 NVFP4 反量化，所以 torch.compile 和 CUDA graph 都能正常用，配置简单很多。

### 2. 学习收获

从 NVFP4 踩坑到 FP8 切换，学到的：

1. **模型结构影响 backend 选择**：GDN 这种非标准 attention 要专门处理。
2. **量化路径影响数值稳定性**：NVFP4 + torch.compile 会触发 NaN，FP8 不会。
3. **CUDA graph 和 torch.compile 可以解耦**：禁用 torch.compile 不等于禁用 CUDA graph。
4. **backend 选择是性能和兼容性的权衡**：FlashInfer 快但有 bug，Triton 稳但略慢。

### 3. 待探索项

- [ ] 测试 FlashInfer 修复后的 GDN prefill kernel（关注 flashinfer PR #3680）
- [ ] 测试 vLLM 新版对 NVFP4 + torch.compile 的修复（关注 PR #37356）
- [ ] 对比 FLASH_ATTN vs FlashInfer 在 FP8 项目里的性能差异

---

## 总结

| 概念 | 作用 | 项目里的体现 |
|---|---|---|
| FlashAttention | H100 最优 attention kernel | FP8 项目默认用 |
| FlashInfer | 推理优化库，但有 GDN bug | NVFP4 项目要绕开 |
| Triton backend | 兼容性好，JIT 编译 | GDN prefill 用它绕 NaN |
| GDN | Qwen3.6 的线性注意力层 | 需要 triton backend |
| CUDA Graph | 消除 kernel launch 开销 | decode 提速 5 倍 |
| FULL_DECODE_ONLY | 只 capture decode graph | NVFP4 项目的折中方案 |
| FULL_AND_PIECEWISE | decode + prefill 都 capture | FP8 项目可用 |
| torch.compile/inductor | 算子融合优化 | NVFP4 项目要禁用（NaN） |
| TORCH_COMPILE_DISABLE | 禁用 torch.compile 但保留 CUDA graph | NVFP4 的关键 workaround |
