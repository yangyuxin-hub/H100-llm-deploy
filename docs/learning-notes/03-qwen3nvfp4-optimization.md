# Qwen3.6-27B-NVFP4 在 H100 上的部署优化记录

> SSH 连接方式和服务器环境信息见 `AGENTS.md` 的「H100 服务器连接」章节。

## 模型与环境

| 项 | 值 |
|---|---|
| 模型 | nvidia/Qwen3.6-27B-NVFP4 |
| 量化 | ModelOpt 混合量化（FP8 + W4A16_NVFP4） |
| 架构 | Qwen3_5ForConditionalGeneration（hybrid: GDN linear attention + full attention） |
| GPU | 4× H100 80GB HBM3 |
| Driver | 550.144.03（不升级，用 CUDA 12.9 原生兼容） |
| vLLM | 0.24.0（vllm/vllm-openai:v0.24.0-cu129-ubuntu2404, CUDA 12.9） |
| 部署方式 | Docker 容器，模型权重只读挂载 |

---

## 遇到的问题

### 问题 1：输出全是 "!"（NaN logits）

**症状**：
- `/v1/chat/completions` 返回 200 个 "!" 在 reasoning 字段，content 为 null
- `/v1/completions` 直接报错：`Out of range float values are not JSON compliant: nan`
- 模型推理产生 NaN logits，采样器退化为输出 token id 0
- token id 0 = "!"（已验证 `/detokenize [0,1,2,3]` → `!"#$`）

**根因（通过二分实验定位）**：

vLLM 0.24.0 的 **torch.compile / inductor 编译阶段**对这个 NVFP4 + GDN 模型有 bug：

1. inductor 做算子融合时改变了 Marlin FP4 或 GDN kernel 的数值行为
2. 导致 logits 产生 NaN
3. 采样器遇到 NaN 退化为输出 token 0（"!"）

这不是 attention backend 或 KV cache 量化的问题，是 torch.compile 本身的问题。

**社区佐证**：
- vLLM issue #47367：GLM5.2-nvfp4 + v0.24.0 同样输出 "!"
- vLLM issue #38527：Qwen3.5-35B-A3B-FP8 同样输出 "!"
- vLLM issue #24870：Qwen3-Next 80B 在 Hopper 上输出 "!"，确认 logits 为 NaN
- vLLM PR #42076：GDN KKT kernel 在 Hopper 上精度问题（已修 Triton 路径，FlashInfer 路径未修）
- vLLM PR #37356：FlashInfer NVFP4 未初始化 buffer 导致 NaN 传播（Draft 未合并）
- HF discussion：nvidia/Qwen3.6-27B-NVFP4 #10，多人遇到同样问题

**早期修复**：强制 `--enforce-eager`（禁用 torch.compile 和 CUDA graph）。

**当前更优修复**：用 `TORCH_COMPILE_DISABLE=1` 单独禁用 torch.compile，并通过 `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` 恢复 decode-only CUDA graph。这样绕开 inductor NaN，同时保留 decode 阶段 CUDA graph 性能。

### 问题 2：FlashInfer GDN prefill kernel 产生 NaN

**症状**：vLLM auto 选择 FlashInfer GDN prefill backend 时，输出全是 "!"。

**根因**：FlashInfer GDN prefill kernel 在 Hopper 上有数值精度问题（flashinfer issue #2490，gating space 不匹配导致全 NaN）。vLLM 有 `torch.exp(g)` workaround 但不彻底。

**修复**：强制 `--gdn-prefill-backend triton`。

---

## 二分实验记录

为定位 NaN 根因，做了 9 组实验：

| 实验 | 配置 | 正确性 | 结论 |
|---|---|---|---|
| 基线 | TRITON + enforce-eager | ✅ | 基准配置 |
| 低风险优化 | +async +prefix-cache +fastsafetensors | ✅ | 提升 <3%，误差范围 |
| 实验A | FLASH_ATTN + GDN triton + CUDA graph | ❌ NaN | 排除 attention backend 嫌疑 |
| 实验B | bfloat16 KV cache + CUDA graph | ❌ NaN | 排除 fp8 KV cache 嫌疑 |
| 实验D | 小 max-len(32K) + CUDA graph | ❌ NaN | 排除 max-len 过大嫌疑 |
| 实验E | PIECEWISE graph（只 prefill） | ❌ NaN | 排除 decode graph 捕获嫌疑 |
| 实验F | `mode=NONE`（禁用 torch.compile） | ✅ | **定位根因：torch.compile/inductor** |
| 实验G | TRITON + enforce-eager + MTP | ✅ | MTP 可用，单请求 +115% |
| 实验H | FLASH_ATTN + enforce-eager | ✅ | **FLASH_ATTN 本身没问题** |
| 实验I | FLASH_ATTN + enforce-eager + MTP | ✅ | 旧最优 |
| 实验J | `TORCH_COMPILE_DISABLE=1` + `FULL_DECODE_ONLY` CUDA graph + MTP3 | ✅ | **当前最优** |

**排除项**（都不是 NaN 根因）：
- ✗ FlashAttn backend（实验H 证实 enforce-eager 下正常）
- ✗ FlashInfer backend（同样在 enforce-eager 下可用，但 GDN prefill 必须用 triton）
- ✗ fp8 KV cache 量化（实验B 排除）
- ✗ max-model-len 过大（实验D 排除）
- ✗ decode FULL graph 捕获（实验E 排除）

**最终根因**：torch.compile / inductor 编译阶段对 NVFP4 + GDN 模型有 bug。CUDA graph 本身不是根因；decode-only CUDA graph 在禁用 torch.compile 后可用。

---

## 当前最优配置

cu129 原生版本（已验证正确性和性能）：

```bash
docker run -d \
  --name qwen-nvfp4-vllm \
  --runtime nvidia --gpus all --ipc=host -p 8000:8000 \
  -v /mnt/nvme0/models:/mnt/nvme0/models:ro \
  --env VLLM_DEEP_GEMM_WARMUP=skip \
  --env TORCH_COMPILE_DISABLE=1 \
  vllm/vllm-openai:v0.24.0-cu129-ubuntu2404 \
  --model /mnt/nvme0/models/Qwen3.6-27B-NVFP4 \
  --served-model-name qwen3.6-27b-nvfp4 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --quantization modelopt \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --kv-cache-dtype auto \
  --attention-backend FLASH_ATTN \
  --mm-encoder-attn-backend FLASH_ATTN \
  --gdn-prefill-backend triton \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"FLASH_ATTN"}'
```

### 相比 cu130 + cuda-compat 的改进

| 项 | cu130 + cuda-compat | cu129 原生 | 变化 |
|---|---|---|---|
| Docker 镜像 | vllm/vllm-openai:latest (CUDA 13.0.2) | vllm/vllm-openai:v0.24.0-cu129-ubuntu2404 (CUDA 12.9) | 切换 |
| cuda-compat 层 | 需要 (VLLM_ENABLE_CUDA_COMPATIBILITY=1) | **不需要** | 简化 |
| 启动时间 | ~280s | ~180s | -36% |
| 单请求 TPS | 33.7 | 31.2 | -7%（误差范围） |
| 5 并发吞吐 | 78.5 | 80.2 | +2% |
| MTP 接受率 | - | 94.3% | 良好 |
| linear_backend | 强制 triton | auto（可选） | 改善 |

**关键收益**：去掉 cuda-compat 层后，启动快 100 秒，原生兼容 driver 550，无需额外的用户态库路径配置。

### 参数分类说明

| 类型 | 参数 | 作用 |
|---|---|---|
| 基础 | `--quantization modelopt` | NVFP4 量化加载 |
| | `--tensor-parallel-size 4` | 4 卡张量并行 |
| | `--max-model-len 32768` | 32K 上下文（为 MTP 腾显存） |
| | `--gpu-memory-utilization 0.90` | 显存占用 90% |
| 功能性 | `--reasoning-parser qwen3` | thinking 字段解析 |
| | `--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder` | 工具调用 |
| | `--trust-remote-code` | 信任模型代码 |
| bug workaround | `--enforce-eager` | **必须**：禁用 torch.compile，绕过 NaN bug |
| | `--gdn-prefill-backend triton` | **必须**：绕过 FlashInfer GDN NaN |
| | `--attention-backend FLASH_ATTN` | 可选 triton 或 flash_attn（后者略快） |
| | `--mm-encoder-attn-backend FLASH_ATTN` | 同上 |
| | `--disable-custom-all-reduce` | driver 550 不支持 custom all-reduce |
| 环境 | `VLLM_DEEP_GEMM_WARMUP=skip` | 跳过 DeepGEMM 预热 |
| 性能优化 | `--speculative-config` | MTP 推测解码，decode 提速 2× |

---

## 性能数据

### 测试方法

- 5 个中文 prompt（诗、量子计算、Python、中国历史、机器学习）
- 串行：1 并发，max_tokens=512
- 并发：5 并发，max_tokens=256
- temperature=0.7

### 各配置性能对比

| 配置 | 单请求 TPS | 5并发吞吐 | 相对基线 |
|---|---|---|---|
| 基线（TRITON + enforce-eager） | 14.2 | 57.2 | 1.0× |
| 实验G（TRITON + MTP） | 30.6 | 71.8 | 2.15× / 1.25× |
| 实验H（FLASH_ATTN，无 MTP） | 15.1 | 65.5 | 1.06× / 1.14× |
| 实验I（FLASH_ATTN + MTP，cu130+compat） | 33.7 | 78.5 | 2.37× / 1.37× |
| **cu129 原生（FLASH_ATTN + MTP2）** | **31.2** | **80.2** | **2.20× / 1.40×** |
| **cu129 原生（FLASH_ATTN + MTP3，单用户，eager）** | **32.4（512 tokens）** | 未测 | **2.28× / -** |
| **cu129 原生（MTP3 + decode-only CUDA graph）** | **162.0（512 tokens）** | 未测 | **11.4× / -** |

### cu129 vs cu130 对比说明

cu129 原生版本在并发吞吐上略优于 cu130+compat（80.2 vs 78.5），单请求 TPS 略低（31.2 vs 33.7）但在误差范围内。关键优势是去掉了 cuda-compat 层，启动快 100 秒，配置更简单。

### 性能瓶颈分析

当前 33.7 tok/s 的限制因素：

1. **CUDA graph 无法开启**（最大瓶颈，-30~50%）
   - torch.compile/inductor 对 NVFP4+GDN 模型有 bug
   - enforce-eager 模式下每步几百次 kernel launch，CPU→GPU 通信开销占 30-50%
   - 需等 vLLM 修复 inductor 对 Marlin/GDN 的编译支持

2. **Marlin FP4 weight-only 路径**
   - H100 无原生 FP4 计算单元，权重每次要反量化
   - 比 FP8 原生计算慢

3. **MTP 并发场景提升有限**
   - MTP 主要加速 decode 阶段
   - 并发时 prefill 成为瓶颈，MTP 提升从 +115% 降到 +25%

---

## 理论吞吐估算与实测对比

> 本节用标准 roofline 公式估算 decode/prefill 吞吐上界，再与实测对比，定位瓶颈所在。
> 注意：本节实测数据分两批——**FP8 容器**（2026-07-06 在线 benchmark，`qwen3.6-27b-fp8` 容器，MTP2）和 **NVFP4 容器**（本文件其他章节记录的，MTP3 + decode-only CUDA graph）。两者测试条件不同，不能直接对比绝对值。

### 1. Decode 阶段（访存瓶颈）

逐 token 生成时，每生成一个 token 都要把整个模型权重从 HBM 读一遍，所以：

```
单流 decode 吞吐 ≈ HBM带宽 / 每卡权重读取量
```

**TP=4 时**：权重分摊到 4 卡，每卡只读 1/4 权重，但 4 卡同步生成同一个 token 流（不是 4 个独立流），所以单流吞吐 = 单卡带宽 / 单卡权重，**比 TP=1 提升约 4 倍**（受 all-reduce 通信打折）。这一点与"TP=4 带宽也 ×4 结果不变"的直觉说法不同——TP=4 单流 decode 确实更快，因为每卡访存量降低了。

| 量化 | 每卡权重 (TP=4) | H100 HBM 带宽 | 理论 decode 上界 |
|---|---|---|---|
| FP8 (1 byte/param) | 27 / 4 = 6.75 GB | 3.35 TB/s | 3.35e12 / 6.75e9 ≈ **496 tok/s** |
| NVFP4 (0.5 byte/param) | 13.5 / 4 = 3.375 GB | 3.35 TB/s | 3.35e12 / 3.375e9 ≈ **992 tok/s** |

**实测对比**：

| 配置 | 实测单流 decode | 理论上界 | 效率 | 差异来源 |
|---|---|---|---|---|
| FP8 + MTP2（2026-07-06 在线） | 147.3 tok/s | 496 tok/s | ~30% | attention + KV cache 读写 + all-reduce + MTP draft 模型 + HTTP/API |
| NVFP4 + MTP3 + decode-only CUDA graph | 162.0 tok/s | 992 tok/s | ~16% | 上述全部 + Marlin FP4 反量化开销（H100 无原生 FP4 计算单元） |

**关键观察**：
- FP8 实测 147 tok/s 对应 TPOT 6.50ms，其中纯权重读取约 2.01ms（6.75GB / 3.35TB/s），仅占 31%，其余 69% 是 attention/KV/通信/MTP draft 开销
- NVFP4 理论上界是 FP8 的 2 倍（权重减半），但实测反而只快 10%（162 vs 147），因为 Marlin 反量化吃掉了大部分收益。这与"性能瓶颈分析"中"Marlin FP4 weight-only 路径比 FP8 原生计算慢"的结论一致
- **结论**：在 H100 上，NVFP4 的访存优势被反量化开销抵消。要兑现 NVFP4 理论吞吐，需要原生 FP4 计算单元（如 Blackwell 架构）

### 2. Prefill 阶段（算力瓶颈）

Prompt 处理是算力密集型，每 token FLOPs ≈ 2 × P（权重乘加）+ attention 计算：

```
prefill 吞吐 ≈ GPU总算力(TOPS) / (2 × 参数量 + seq_len × d_model)
```

简化估算（忽略 attention 项，长 prompt 时 attention 占比上升）：

| 量化 | H100 算力（TP=4） | 理论 prefill 上界 |
|---|---|---|
| FP8 稠密 | 1979 × 4 = 7916 TFLOPS | 7916e12 / (2 × 27e9) ≈ **147K tok/s** |
| FP16 稠密（NVFP4 反量化后走 FP16） | 989 × 4 = 3956 TFLOPS | 3956e12 / (2 × 27e9) ≈ **73K tok/s** |

**实测对比**（FP8，长文档场景并发 1，输入 8192 tok）：

| 指标 | 实测 | 理论 | 效率 |
|---|---|---|---|
| TTFT 中位 | 342.2 ms | 8192 / 147K ≈ 55.7 ms | ~16% |
| 实测 prefill 吞吐 | 8192 / 0.342 ≈ 24K tok/s | 147K tok/s | ~16% |

**差异来源**：
- attention 计算（2 × seq_len × d_model）在 8K 上下文下占比不小，简化公式高估了
- KV cache 写入访存开销
- TP all-reduce 通信（每层一次）
- chunked prefill 调度开销
- 容器内其他进程竞争

### 3. 多请求并发（系统吞吐）

batch 摊销权重读取，公式：

```
系统吞吐 ≈ HBM带宽 / (每卡权重/batch + KV_cache单token读量)
```

两个约束：
- **显存容量限制 batch 上限**：`max_batch ≈ (总显存 - 权重 - 框架开销) / (KV_cache单请求)`
- **KV cache 访存瓶颈**：batch 足够大时，KV cache 读取超过权重，转为 KV-cache 访存瓶颈

**实测并发吞吐**（FP8，2026-07-06 在线 benchmark，完整数据见 `logs/bench/SUMMARY.md`）：

| 场景 | 并发 | output tok/s | 是否线性扩展 |
|---|---|---|---|
| 对话 500→512 | 1 | 147.3 | 基准 |
| 对话 500→512 | 4 | 501.4 | 3.4×（近线性） |
| 对话 500→512 | 8 | 1013.4 | 6.9×（近线性） |
| 对话 500→512 | 16 | 1797.7 | 12.2×（近线性，受 max-num-seqs=16 限制未饱和） |
| 长文档 8K→512 | 1 | 135.0 | 基准 |
| 长文档 8K→512 | 16 | 1226.9 | 9.1×（仍有扩展空间） |

**关键观察**：
- 并发 16 仍未饱和，说明 GPU 算力和访存都未到瓶颈，是 `max-num-seqs=16` 调度上限卡住了
- 对话场景并发 16 时 total tok/s 3553，远低于理论 prefill+decode 上界，说明还有较大容量空间
- 长文档场景 total tok/s 20856（含 prefill），主要受 prefill 算力限制

### 4. Little's Law（端到端容量规划）

```
并发数 = 系统吞吐(tokens/s) × 平均单请求延迟(s)
```

以 FP8 对话场景为例：单请求延迟 ≈ TTFT + output_len × TPOT = 0.07 + 512 × 0.0065 ≈ 3.4s，并发 16 时系统吞吐 1798 tok/s → 可支撑并发 = 1798 × 3.4 / 512 ≈ 12 个同时在线用户。实测并发 16 能跑通，说明调度上限还有余量。

### 5. 公式使用建议

- **decode 估算**：用 `HBM带宽 / 每卡权重` 算上界，实测效率 30%（FP8）/ 16%（NVFP4）属正常范围
- **prefill 估算**：用 `算力 / 2P` 算上界，实测效率 15-20% 属正常（attention + 通信开销大）
- **容量规划**：先测单流 TPOT 和 TTFT，再用 Little's Law 估算并发上限
- **诊断**：若实测效率 <10%，排查 CUDA graph / kernel launch / 通信 / 量化路径

---

## 后续可尝试的优化

### 优先级 1：恢复 256K 上下文

当前为 MTP 腾显存降到 32K。测试 256K + MTP 是否可行：
- MTP 只增加 1 层（mtp_num_hidden_layers=1），显存增量小
- 可能 256K 也能放下
- 测试方法：改 `--max-model-len 262144`，观察 OOM 或正常

### 已验证：增加推测 token 数到 3

2026-07-03 单用户测试中，`num_speculative_tokens: 3` 比 2 更快：
```json
{"method":"mtp","num_speculative_tokens":3,"attention_backend":"FLASH_ATTN"}
```

结果：
- 64 tokens：34.64 tok/s（MTP2 为 29.86 tok/s）
- 256 tokens：33.22 tok/s（MTP2 为 28.76 tok/s）
- 512 tokens：32.40 tok/s（MTP2 为 28.25 tok/s）
- 流式 TTFT 仍约 0.11-0.20 秒
- KV cache 从 4,848,571 tokens 降到 4,407,792 tokens，下降约 9.1%

结论：单用户最快响应场景优先使用 MTP3；如果转向高并发或长上下文，需要重新评估这 9% KV cache 代价。

### 已验证：禁 torch.compile + decode-only CUDA graph

完整 `VLLM_COMPILE + FULL_AND_PIECEWISE` 会触发 NaN / 连续 `!`。可行路径是：

```bash
--env TORCH_COMPILE_DISABLE=1
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

关键日志：
- `TORCH_COMPILE_DISABLE is set, disabling torch.compile`
- `cudagraph_mode=<CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>`
- `Capturing CUDA graphs (decode, FULL)`
- `Graph capturing finished in 13 secs, took 2.52 GiB`

性能：
- 64 tokens：131.26 tok/s
- 256 tokens：162.07 tok/s
- 512 tokens：162.04 tok/s
- 标准脚本重启后的首个请求可能被推理期 Triton JIT 拉低，观察到 512 token 冷请求为 86.3 tok/s；连续热身后恢复到 155-165 tok/s
- 流式 TTFT：约 0.112-0.139 秒

代价：
- KV cache 从 MTP3 eager 的 4,407,792 tokens 降到 4,155,578 tokens，约下降 5.7%
- decode graph pool 额外占用约 2.52 GiB

### 优先级 3：尝试 CUDA graph 部分开启

当前完全禁用 torch.compile。尝试只禁用 inductor 但保留 CUDA graph：
```bash
-cc '{"backend":"","cudagraph_mode":"FULL_AND_PIECEWISE"}'
```
- 空字符串 backend 可能跳过 inductor 编译但保留 graph
- 未测试，可能触发同样 NaN

### 优先级 4：监控 vLLM 版本更新

关注以下 PR/issue 的合并状态：
- vLLM PR #37356：FlashInfer NVFP4 NaN 修复（Draft）
- flashinfer PR #3680：GDN g_space 彻底修复（Open）
- vLLM PR #45320：NVFP4 MoE 缺失 scale 校验（Open）

升级到包含修复的版本后，重新测试：
1. 去掉 `--enforce-eager`（开 CUDA graph）
2. 去掉 `--gdn-prefill-backend triton`（用 FlashInfer）
3. 预期单请求 TPS 可达 60-80 tok/s

### 优先级 5：prefix-caching 场景优化

对于重复 system prompt 的对话场景，加 `--enable-prefix-caching`：
- 首次请求正常速度
- 后续相同前缀请求 prefill 阶段加速 50-90%
- 当前测试用不同 prompt 看不到效果

### 优先级 6：向社区反馈

在 vLLM 开新 issue，标题：
`[Bug]: Qwen3.6-27B-NVFP4 on H100 (TP=4, vLLM 0.24.0) outputs all "!" (NaN logits) with torch.compile/inductor; fixed by --enforce-eager`

引用：#47367, #42076, #24870, #37356, flashinfer #2490, HF discussion #10

---

## 常用调试命令

```bash
# 查看当前容器启动参数
docker inspect qwen-nvfp4-vllm --format "{{json .Config.Cmd}}" | python3 -m json.tool

# 查看日志中的 NaN 相关信息
docker logs qwen-nvfp4-vllm 2>&1 | grep -iE "nan|inf|error|exception"

# 查看吞吐统计
docker logs qwen-nvfp4-vllm 2>&1 | grep "Engine 000" | tail -15

# 快速正确性测试
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":500,"temperature":0}'

# 检查 token 0 对应的字符（验证 "!" 问题）
curl -s http://127.0.0.1:8000/detokenize \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-nvfp4","tokens":[0,1,2,3]}'

# 绕过 chat template 的裸推理（排查 parser 问题）
curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-nvfp4","prompt":"Hello","max_tokens":20,"temperature":0}'
```

---

## 版本历史

- 2026-07-02：初版，定位 NaN 根因（torch.compile），找到最优配置（FLASH_ATTN + MTP），单请求 33.7 tok/s
- 2026-07-03：切换到 cu129 原生镜像，去掉 cuda-compat 层，启动快 36%，并发吞吐 80.2 tok/s
- 2026-07-06：新增「理论吞吐估算与实测对比」章节。用 roofline 公式估算 decode/prefill 上界，对比 FP8 在线 benchmark（147 tok/s 单流 decode，效率 30%）和 NVFP4 实测（162 tok/s，效率 16%）。修正 TP=4 单流 decode 的理论理解。完整 benchmark 数据见 `logs/bench/SUMMARY.md`

---

## 2026-07-03 cu129 镜像部署记录

### 背景

cu130 镜像（vllm/vllm-openai:latest, CUDA 13.0.2）需要 cuda-compat 层才能在 driver 550 上运行，启动慢（280s）且配置复杂。尝试 cu129 镜像（vllm/vllm-openai:v0.24.0-cu129-ubuntu2404, CUDA 12.9）原生兼容 driver 550。

### 镜像获取过程

1. daocloud 镜像源对部分 blob 返回 403 DENIED（缓存策略限制）
2. Docker Hub 官方源被墙（连接重置）
3. 通过本地 HTTP 代理（127.0.0.1:7890）配置 Docker daemon，从 Docker Hub 官方源拉取成功
4. `docker save` + `rsync` 传输到 H100（12GB，~100MB/s，2分钟）
5. 远程 `docker load` 加载镜像

### 测试结果

- **启动时间**：~180s（比 cu130+compat 快 100s）
- **正确性**：1+1=? → "2"，无 NaN 问题
- **MTP 接受率**：94.3%，mean acceptance length 2.89
- **单请求 TPS**：31.2（5 prompt 平均）
- **5 并发吞吐**：80.2 tok/s

### 配置变化

- Docker 镜像：`vllm/vllm-openai:latest` → `vllm/vllm-openai:v0.24.0-cu129-ubuntu2404`
- 去掉 `VLLM_ENABLE_CUDA_COMPATIBILITY=1`（不再需要）
- `QWEN_LINEAR_BACKEND` 从 `triton` 改为空（auto，但当前配置仍留空让其自动选择）
- 其他实验I 配置保持不变
