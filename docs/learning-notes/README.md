# 大模型知识点学习笔记

这个目录用于记录大模型部署和推理相关的学习笔记。

目标不是只保存命令，而是把部署过程中遇到的概念讲清楚：模型结构、推理流程、显存占用、多卡并行、vLLM、KV Cache、FP8、长上下文等。

## 当前学习主线

1. 理解大模型推理是怎么发生的。
2. 理解模型文件和权重目录里每个文件的作用。
3. 理解 vLLM 如何把模型部署成 OpenAI 兼容 API。
4. 理解 H100、FP8、TP、KV Cache 和长上下文之间的关系。
5. 结合 Qwen3.6-27B-FP8 + Agents-A1-FP8 双模型并行部署做实际记录。

## 笔记目录

1. [大模型推理基础](./01-llm-inference-basics.md)
2. [部署相关核心概念](./02-deployment-concepts.md)
3. [Qwen3.6 NVFP4 在 H100 上的优化记录](./03-qwen3nvfp4-optimization.md)(历史)
4. [双模型并行部署学习笔记](./04-dual-model-parallel-deployment.md)

## 后续可以补充的主题

1. `05-kv-cache-and-long-context.md`：KV Cache、PagedAttention、长上下文显存占用。
2. `06-tensor-parallelism.md`：TP、DP、多卡切分、NCCL 通信、CUDA_VISIBLE_DEVICES。
3. `07-fp8-and-h100.md`：FP8、BF16、H100 Tensor Core。
4. `08-speculative-decoding-and-mtp.md`：MTP、speculative decoding、性能实测。
