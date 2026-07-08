# 大模型知识点学习笔记

> 完整知识体系总目录见 [INDEX.md](./INDEX.md)（含全景图、交叉索引、项目配置对应表、阅读建议）。

这个目录用于记录大模型部署和推理相关的学习笔记。

目标不是只保存命令，而是把部署过程中遇到的概念讲清楚：模型结构、推理流程、显存占用、多卡并行、vLLM、KV Cache、FP8、长上下文等。

## 当前学习主线

1. 理解大模型推理是怎么发生的。
2. 理解模型文件和权重目录里每个文件的作用。
3. 理解 vLLM 如何把模型部署成 OpenAI 兼容 API。
4. 理解 H100、FP8、TP、KV Cache 和长上下文之间的关系。
5. 结合 Qwen3.6-27B-FP8 + Agents-A1-FP8 双模型并行部署做实际记录。

## 笔记目录

### 基础与部署

1. [大模型推理基础](./01-llm-inference-basics.md)
2. [部署相关核心概念](./02-deployment-concepts.md)
3. [Qwen3.6 NVFP4 在 H100 上的优化记录](./03-qwen3nvfp4-optimization.md)(历史)
4. [双模型并行部署学习笔记](./04-dual-model-parallel-deployment.md)

### 机制与原理

5. [KV Cache、PagedAttention 与长上下文](./05-kv-cache-pagedattention-long-context.md)：KV Cache 精确公式、GQA、PagedAttention 分页管理、Continuous Batching、FlashAttention tiling、RoPE 外推、Prefix Caching。
6. [张量并行、通信与多卡部署](./06-tensor-parallelism-communication.md)：TP/DP/PP/EP 对比、列并行/行并行切分、all-reduce、NVLink vs PCIe、NCCL、CUDA_VISIBLE_DEVICES 原理。
7. [量化、FP8 与 H100 Tensor Core](./07-quantization-fp8-h100.md)：浮点格式（FP8 E4M3/E5M2、BF16、NVFP4）、量化分类（weight-only/W8A8/per-group）、H100 Tensor Core、Marlin 反量化、ModelOpt、KV Cache 量化。
8. [Attention Backend 与 CUDA Graph](./08-attention-backend-cuda-graph.md)：FlashAttention v2/v3、FlashInfer、Triton、GDN、CUDA Graph 原理、torch.compile/inductor、FULL_DECODE_ONLY、NaN bug 根因。

### 模型与生成

9. [MoE 与专家并行](./09-moe-expert-parallelism.md)：MoE 路由机制、总参数 vs 激活参数、EP 专家并行、load balancing、MoE + MTP、DeepSeek FP4/DSpark。
10. [采样策略与结构化输出](./10-sampling-structured-output.md)：temperature/top-k/top-p/min-p 数学含义、思考模型采样参数、结构化输出（grammar/JSON/regex）、Function Calling 底层、tool-call-parser。

### 框架与运维

11. [vLLM 内部架构与性能调优](./11-vllm-internals-tuning.md)：Scheduler/Worker/KV Cache Manager 三层架构、max-num-seqs/max-num-batched-tokens/gpu-memory-utilization 相互作用、chunked prefill、preemption、prefix caching、benchmark 指标解读。
12. [监控与稳定性](./12-monitoring-stability.md)：vLLM /metrics 端点、Prometheus + Grafana、dcgm-exporter、稳定性测试（长上下文/长时间并发/OOM 恢复/优雅重启）、常见稳定性问题排查。

## 后续可以补充的主题

1. `13-speculative-decoding-deep-dive.md`：MTP 投机解码的数学原理、接受率分析、draft 模型选择、与 EAGLE/Medusa 对比。
2. `14-sglang-and-other-frameworks.md`：SGLang RadixAttention、TensorRT-LLM engine 构建、TGI 对比。
3. `15-production-deployment.md`：负载均衡、多副本、灰度发布、A/B 测试、容量规划。
