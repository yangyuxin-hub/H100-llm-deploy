# Qwen3.6-27B 推理吞吐优化路线调研

日期：2026-07-13

## 结论先行

在 node1 的 4×H100 80GB 上，当前最值得优先验证的不是立即从零训练 DSpark，而是按以下顺序做可复现对比：

1. **把 GPU2,3 的 TP=2 实验实例改成两个 TP=1 副本做 A/B**。27B FP8 权重约 29GB，单张 H100 80GB 可以容纳模型和可用 KV cache。对“最大化总吞吐”而言，两个无通信的副本很可能优于一个 TP=2 实例。
2. **对现有 MTP 做并发分层调优**。当前主实例实际使用 MTP3，真实日志中的平均接受长度达到 3.8-4.0；这是很强的基线。DSpark 必须在相同 workload 下超过它才值得替换。
3. **测试 ThinkingCap FP8**。它直接减少约 46% 的平均 thinking token，提升的是“每个任务完成速度”和“单位时间完成任务数”，往往比单 token 加速更有价值。
4. **测试现成 Qwen3.6 DFlash/DSpark 草稿头**，先证明收益再训练。社区已经有 stock Qwen3.6-27B 的 DFlash，以及多个 DSpark/领域微调版本。
5. **把 node1 已有 NVFP4 当成 H100 上的 W4A16/省显存方案，而不是 W4A4 加速方案**。H100 没有原生 FP4 Tensor Core；Unsloth 的 2.5× 数据来自 B200、并发 128，不能外推到 H100。

自己训练 DSpark **可行**。推荐使用“GPU2,3 分阶段离线流程”：先用两卡 verifier 提取 hidden states，停止实验容器后再用两卡 FSDP 训练草稿模型。这样不动 GPU0,1 的稳定服务。当前 NVMe 还有约 3.2TB，足以保存试验数据。

## node1 当前事实基线

以下是 2026-07-13 对远程 H100 的状态检查结果，不是本地机器状态；本次没有启停或重建容器：

| 项目 | 观察结果 |
|---|---|
| GPU | 4× NVIDIA H100 80GB HBM3，GPU 间为 NVLink `NV6` |
| vLLM | `0.24.0+cu129` |
| 稳定实例 | GPU0,1，TP=2，端口 8000，端点可访问 |
| 实验实例 | GPU2,3，TP=2，端口 8001，端点可访问 |
| FP8 权重 | 29GB |
| 本地 NVFP4 权重 | 21GB，ModelOpt mixed precision |
| DSpark 权重 | 已有 `DeepSeek-V4-Flash-DSpark`，156GB；它不是 Qwen 草稿模型 |
| 磁盘 | `/mnt/nvme0` 剩余约 3.2TB |
| 系统内存 | 503GiB，总可用约 457GiB |

稳定实例的实际 Docker 参数与本地 `config/serving.env` 有漂移：实际为 `FLASH_ATTN + MTP3`，本地配置当前写的是 `flashinfer + MTP2`；实际稳定实例也没有 `--language-model-only`。在任何重启或正式 benchmark 前应先对齐配置，否则重启后基线会变化。

`scripts/status.sh` 还把稳定实例报告为 PID 已死，但 `docker ps` 和 `/v1/models` 都证明容器及端点正常。这是 PID 文件状态与 Docker 状态不一致，不代表服务已停止。

稳定实例近期真实流量日志显示：

- Prefix cache hit rate 约 88.5%，说明 prefix caching 已经产生明显价值。
- MTP3 平均接受长度约 3.82-4.00，平均 draft acceptance 约 94%-100%。
- 单个运行请求窗口内 generation throughput 约 294-350 tok/s；这是窗口日志，不是标准 benchmark，不能与其他社区数据直接横比。

## 社区是否有人训练 DSpark

答案是肯定的，而且社区在 2026 年 7 月初快速增长：Hugging Face 的 `dspark` 标签页当前约有 36 个模型。

与 Qwen3.6-27B 直接相关的公开结果包括：

| 模型 | 目标模型 | 公开结果 | 适用性 |
|---|---|---|---|
| `z-lab/Qwen3.6-27B-DFlash` | stock Qwen3.6-27B | 2B 草稿头；约 45K 月下载；仍提示训练/引擎支持未完全稳定 | 最适合先做 stock baseline |
| `Avesed/Qwen3.6-27B-DSpark` | Avesed W4A16 target | K=7；GSM8K 接受长度 6.55，code 3.84，chat 2.54 | target 特定，不能直接假设兼容官方 FP8 |
| `pablohassan/Qwen3.6-27B-DSpark-FR` | Qwen3.6 法语领域 | 3,000 prompts，6 epochs；单流比 MTP 快 50%（GB10） | 证明领域微调有效，但硬件和语料不同 |
| `Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft` | AEON merge | 15,936 sequences，约 4,500 step 收敛；聚合吞吐比原 DFlash +11% | 训练配方信息完整，但不是 stock target |

重要限制：草稿模型通常是 **target checkpoint 和 workload 特定** 的。社区模型在另一个 merge、量化版本、语言或采样设置上接受率可能显著下降。正确流程是先在我们的官方 FP8 target 和真实 agent/coding prompt 上测接受长度，再决定复用、微调或重训。

参考：

- [vLLM Speculators 项目](https://github.com/vllm-project/speculators)
- [Speculators train.py 文档](https://docs.vllm.ai/projects/speculators/en/latest/cli/train/)
- [Qwen3.6-27B DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash)
- [Qwen3.6-27B DSpark FR 配方](https://github.com/Pablohassan/qwen3.6-27b-fr-nvfp4-dspark)
- [Qwen3.6-27B AEON DSpark](https://huggingface.co/Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft)
- [Hugging Face DSpark 模型列表](https://huggingface.co/models?other=dspark)

## 自己训练 DSpark 的资源估算

### 推荐方式：从公开 DFlash warm-start

不建议第一轮从随机初始化开始。社区已有 2B 的 `z-lab/Qwen3.6-27B-DFlash`，把它转换为 DSpark 后，只需继续训练 DFlash backbone、rank-256 Markov head 和 confidence head。公开成功案例也采用该路线。

一个现实的 pilot 配置：

| 项目 | 建议范围 |
|---|---|
| 数据 | 3,000-5,000 条真实分布 prompt，target on-policy 生成回答 |
| 序列长度 | 第一轮 1,024；验证后再扩到 2K/4K |
| 草稿规模 | 约 2B，BF16 权重约 4GB |
| 训练步数 | 先 1,000 step 看收敛；完整约 4,500-6,000 step |
| block / K | 训练 block 8-16；服务时扫描 K=4/6/8/12 |
| 质量验证 | acceptance length、端到端吞吐、最终答案一致性和工具调用成功率 |

### 显存

2B 参数草稿模型全参数训练的粗略显存账：

- BF16 参数约 4GB，BF16 gradient 约 4GB。
- FP32 master weights + Adam 状态约 24GB。
- 参数、梯度、优化器合计约 32GB，另加 activation、临时 buffer 和 CUDA workspace。
- 单张 80GB H100 理论上可做短序列训练；两张 H100 FSDP 更稳，也能提高 batch/序列长度。

这只是量级估算，具体取决于是否冻结 embedding/lm_head、optimizer（AdamW 或 Muon）、activation checkpointing、packed sequence 和 attention backend。

### hidden states 存储

Qwen3.6-27B hidden size 为 5120。若保存 5 个 verifier layer 的 BF16 hidden states，每 token 原始大小约：

```text
5120 × 5 × 2 bytes = 51,200 bytes/token，约 50KiB/token
```

若 3,000 条样本平均 1,024 token，共约 3.07M token，仅 hidden states 原始量级约 157GB；再加 token、logits/metadata、文件系统开销和 checkpoint，应预留 250-500GB。node1 目前剩余 3.2TB，磁盘不是阻塞点。

Speculators 支持三种模式：

- online：verifier 在线生成 hidden states，训练端不落盘；需要 verifier GPU 与 trainer GPU 同时存在。
- offline：先生成 hidden states，再训练；最适合当前 GPU0,1 不动、GPU2,3 分阶段使用的约束。
- hybrid：第一轮生成并缓存，后续 epoch 复用；在磁盘充足时通常最省总时间。

### node1 上的可执行资源安排

推荐不影响稳定服务的安排：

1. GPU0,1 继续提供稳定服务，不用于训练。
2. 停止 GPU2,3 当前实验容器后，用 TP=2 verifier 生成 on-policy corpus 和 hidden states。
3. 停止 verifier，只用 GPU2,3 运行 `torchrun --nproc_per_node=2` FSDP 训练。
4. 用 GPU2,3 部署草稿 + target，跑 acceptance 和端到端 A/B。

初步 pilot 预计是“小时到一天”量级，完整领域数据、多个 seed/K 和质量评测可能需要 1-3 天。这是工程估算，不是社区公开的 H100 实测时间；应在 1,000-step pilot 后用实际 step time 重算。

### 能否从零训练

资源上能，但第一阶段没有必要。从零训练需要更多数据和超参搜索，且公开 Qwen3.6 DFlash 已提供很好的初始化。只有在 warm-start 的接受率被结构或 domain ceiling 卡住后，才值得尝试随机初始化或改变 draft 层数/target layer IDs。

## 各类加速方案评估

### 1. 减少实际生成 token

这是“有效吞吐”最高优先级路线，因为计算量近似随输出 token 数下降。

**直接关闭 thinking**

- 简单问答、格式转换、检索整理、明确工具调用应使用 `enable_thinking=false`。
- 不需要换权重，风险最低。
- 当前项目已经知道此开关，但还应把它做成网关按任务路由，而不是依赖每个客户端手工设置。

**ThinkingCap FP8**

- 官方模型卡报告 out-of-domain 平均减少 45.8% thinking tokens，准确率宏平均从 81.5 降至 80.7。
- GPQA、MMLU-Pro 等任务 thinking token 减少约 44%-68%；LiveCodeBench 准确率反而提高，但 HMMT、agentic 等个别任务有下降。
- 已发布 `bottlecapai/ThinkingCap-Qwen3.6-27B-FP8`，保留 BF16 MTP head，可直接用 vLLM/SGLang 测试。
- 先测试公开 FP8，不应先自行微调。

若未来自己做 token-efficient SFT：LoRA/QLoRA 在两张 H100 上可行；27B 全参数训练在只使用 GPU2,3 的 160GB 显存下不现实，除非大量 CPU offload，吞吐会很差。node1 当前也没有 BF16 base 权重，需要额外准备约 56GB 权重。

参考：[ThinkingCap 模型卡](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B)、[ThinkingCap FP8](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B-FP8)。

### 2. 单 token 成本：FP8、NVFP4、W4A16

H100 原生支持 FP8/INT8/FP16/BF16，不支持 Blackwell 的原生 FP4 Tensor Core。vLLM 的 LLM Compressor 文档明确说明：低于 SM100 的 GPU 不执行 NVFP4 activation quantization，只退化为 weight-only。

因此：

- Unsloth 所称 Qwen3.6-27B NVFP4 2.5× 是 **1×B200、并发 128** 的结果，并要求 `vLLM>=0.25.0` 和新的 CUTLASS/FlashInfer 栈。
- 当前 H100 + vLLM 0.24.0 不具备相同 W4A4 kernel 路径，不能期待 2.5×。
- node1 已有的 21GB NVFP4 config 实际是 MLP `W4A16_NVFP4` + attention/linear attention FP8 的 mixed precision，并非 H100 W4A4。
- 它仍值得测，因为 21GB 权重比 FP8 的 29GB 小，可能允许更大的 KV cache、更多副本或单卡部署；速度必须实测，不能从 B200 推断。
- 旧日志已记录该 checkpoint 在 vLLM 0.24.0 的某些 compile/kernel 路径会输出连续 `!`，正确性 gate 必须先于性能 gate。

参考：[H100 支持的数据类型](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)、[vLLM NVFP4 说明](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/)、[Unsloth Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4)。

### 3. 投机解码：MTP、DFlash、DSpark、动态 K

投机解码主要优化低到中等 QPS、memory-bound decode。并发增大后，每个 batch 要验证 `BS × K` 个 token，额外计算可能让总吞吐下降。

**当前 MTP**

- 原生、无需额外草稿权重，是必须保留的强基线。
- 当前实际 MTP3 的真实流量接受率很高；先扫描 MTP0/2/3 与并发 1/4/8/16/32。

**DFlash/DSpark**

- 优势是一次并行提出更深的 token block，低并发可能显著超过 MTP。
- 代价是约 2B 草稿头显存、verifier hidden states 接口、新版 vLLM 兼容性和 target/domain 专用训练。
- 第一轮应测公开 DFlash；若 acceptance 不够，再 warm-start 微调 DSpark。

**动态投机深度**

- 新版 vLLM 支持按 batch size 调 K，例如低并发 K=8、中并发 K=3、高并发 K=0。
- 这与“既要低并发速度，又要高并发总吞吐”的目标最匹配。
- 当前 vLLM 0.24.0 是否覆盖 Qwen MTP/DSpark 的完整动态路径需在独立实验镜像验证，不应直接升级稳定实例。

参考：[vLLM speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/)、[Dynamic speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/dynamic_speculative_decoding/)。

### 4. 并行和副本布局

当前两个实例都是 TP=2。TP 减少单请求模型计算时间，但每层都引入跨卡通信；对于能在单卡装下的 27B FP8，最大化 aggregate throughput 时通常应优先比较 data parallel replicas。

值得测试的布局：

| 布局 | 目标 | 预期权衡 |
|---|---|---|
| 1× TP2（当前 GPU2,3） | 单请求速度 | 有 NVLink 通信，只有一个 scheduler |
| 2× TP1（GPU2、GPU3 各一实例） | 总吞吐 | 单请求较慢，但双 scheduler、无 TP all-reduce |
| vLLM DP=2 | 统一端点总吞吐 | 运维更统一，需确认 hybrid model 支持和负载均衡 |
| NVFP4/W4A16 TP1 | 单卡显存/副本密度 | 可能慢于 FP8，也可能因权重带宽下降而更快 |

若业务只有一个长请求，TP2 可能胜；若有多个并发 agent 请求，两个 TP1 副本很可能胜。这必须用 requests/s 和 aggregate output tok/s 同时判断。

### 5. prefix、prompt 和 KV 优化

- Prefix caching 已开启且实际命中率约 88.5%，优先保持并把固定 system/tool schema 的字节级前缀统一。
- 对 agent 多轮对话，避免客户端每轮改写 system prompt 或工具顺序，否则破坏 prefix cache key。
- 对超长会话做摘要、检索裁剪和工具输出截断；减少输入 token 同时降低 TTFT 和 KV 占用。
- FP8 KV 已开启。继续量化 KV 的空间有限，重点应是 workload 的上下文治理。
- 120K 单请求 prefill 在此前实测中约 26s，GPU SM 99%-100%；这类请求的优化重点是减少输入长度、缓存复用或单独的长上下文队列，而不是 DSpark。

### 6. scheduler、backend 和框架

当前已经启用 async scheduling、chunked prefill、prefix caching、FP8 KV 和 65,536 batched tokens，基础配置并不弱。下一步应通过 benchmark 扫参，而不是继续堆开关：

- `max-num-seqs`：从 16 扫到 32/64，旧 TP=4 报告显示 c16 尚未饱和。
- `max-num-batched-tokens`：长 prefill 与 decode 混跑时比较 16K/32K/64K，过大可能伤害 decode TPOT。
- MTP K：0/2/3；草稿模型 K：4/6/8/12。
- `FLASH_ATTN` 与 FlashInfer：以正确性和相同 workload 为前提比较。
- CUDA graph/compile：旧 NVFP4 已出现 NaN/连续 `!`，任何速度提升都必须过输出正确性 gate。

SGLang 值得作为实验框架横向比较，Qwen 官方支持其 MTP；TensorRT-LLM 当前 Qwen3.5 路线要求 driver 575+，node1 是 550，升级 driver 通常需要重启，违反当前绝不重启服务器的约束，因此暂不列为近期路线。

### 7. 模型路由

最大化“任务吞吐”不必所有请求都走 27B：

- 简单任务：Qwen3.6-27B non-thinking，或质量验证过的更小模型。
- 一般 coding/agent：原 FP8 + MTP。
- 复杂推理：ThinkingCap FP8，或原模型 thinking on。
- 超长上下文：独立队列和并发限制，避免阻塞短请求。

还可测试 Qwen3.6-35B-A3B 等低 active-parameter MoE，但它是模型替换而非无损加速，必须单独做 coding、tool calling 和中文质量 gate。

## 建议实验顺序

### P0：建立可信基线

固定同一份 workload，至少覆盖：

- 500 input / 512 output：普通对话。
- 8K input / 512 output：长文档。
- 真实 coding/agent prompts：thinking on，记录完成任务所需总 token。
- 并发：1、4、8、16、32。

同时记录 output tok/s、requests/s、TTFT P50/P99、TPOT P50/P99、MTP/DSpark acceptance、GPU 利用率、每任务总 token、正确率和工具调用成功率。

### P1：不下载新模型即可做

1. GPU2,3 当前 FP8 TP2：MTP0/2/3。
2. GPU2、GPU3 两个 FP8 TP1 副本：MTP0/2/3。
3. 已有 NVFP4：先正确性，再 TP1/TP2 吞吐。
4. `max-num-seqs` 16/32/64 与 batched tokens 16K/32K/64K。

### P2：下载公开权重

1. ThinkingCap FP8：比较任务完成时间、总生成 token 和质量。
2. stock Qwen DFlash：比较 MTP3 与 K=4/8/12。
3. 选择与 stock FP8 兼容的 DSpark checkpoint 做 smoke test；不兼容则进入微调。

### P3：训练

1. 收集 3K-5K 条真实分布 prompt，不保存敏感内容到仓库。
2. target on-policy 生成训练回答。
3. DFlash warm-start 转 DSpark，先 1,000-step pilot。
4. acceptance 有明显提升后再训练到 4.5K-6K step。
5. 对 K 和并发做动态策略，最终只在能同时通过质量和吞吐 gate 时保留。

## 决策门槛

建议预先定义停止条件，避免为了新技术持续投入但没有实际收益：

- **TP1 双副本**：aggregate output tok/s 或 requests/s 至少提高 20%，P99 延迟仍满足需求。
- **ThinkingCap**：任务完成 token 至少下降 30%，coding/tool calling 质量下降不超过可接受门槛。
- **现成 DSpark/DFlash**：c1 至少比 MTP3 快 20%，c16 不低于 MTP3；否则不进入训练。
- **自训 DSpark**：相对公开 warm-start 的 acceptance 或端到端吞吐至少提升 10%。
- **NVFP4/W4A16**：必须先通过中文、代码、tool calling 正确性；性能不高于 FP8 时只作为省显存/多副本方案。
- **框架替换**：同硬件同 workload 至少提高 15%，且功能兼容、稳定性和维护成本可接受。

## 最终判断

- **社区有人训练，而且 Qwen3.6-27B 已有公开基础可复用。** 现在不是“能不能做”的问题，而是“我们的真实 workload 是否值得做”。
- **两张实验 H100 足够训练约 2B 的 DSpark 草稿头**，磁盘和内存也足够；推荐离线分阶段，不占用 GPU0,1。
- **Unsloth W4A4 的核心速度优势不适用于 H100。** 在 node1 上应把 NVFP4 看作 W4A16/省显存工具，真正价值可能是单卡部署和增加副本数。
- **ThinkingCap 很值得优先测试。** 它优化的是总输出 token，因此可以与 FP8、MTP、DSpark 叠加。
- **最大化总吞吐的第一候选是 TP1 多副本，最大化低并发单请求速度的第一候选是 MTP3 对比 DFlash/DSpark。** 两个目标需要分别测量，不能用单一 tok/s 结论覆盖。
