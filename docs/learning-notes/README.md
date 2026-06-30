# 大模型知识点学习笔记

这个目录用于记录大模型部署和推理相关的学习笔记。

目标不是只保存命令，而是把部署过程中遇到的概念讲清楚：模型结构、推理流程、显存占用、多卡并行、vLLM、KV Cache、FP8、长上下文等。

## 当前学习主线

1. 理解大模型推理是怎么发生的。
2. 理解模型文件和权重目录里每个文件的作用。
3. 理解 vLLM 如何把模型部署成 OpenAI 兼容 API。
4. 理解 H100、FP8、TP=4、DP=1、KV Cache 和长上下文之间的关系。
5. 结合 Qwen3.6-27B-FP8 和 DeepSeek-V4-Flash-DSpark 做实际部署记录。

## 笔记目录

1. [大模型推理基础](./01-llm-inference-basics.md)
2. [部署相关核心概念](./02-deployment-concepts.md)

## 后续可以补充的主题

1. `03-model-files-and-safetensors.md`：模型目录、`config.json`、tokenizer、safetensors 分片。
2. `04-vllm-serving.md`：vLLM 服务流程、OpenAI 兼容 API、日志和健康检查。
3. `05-kv-cache-and-long-context.md`：KV Cache、PagedAttention、长上下文显存占用。
4. `06-tensor-parallelism.md`：TP、DP、多卡切分、NCCL 通信。
5. `07-fp8-and-h100.md`：FP8、BF16、H100 Tensor Core。
6. `08-speculative-decoding-and-mtp.md`：MTP、speculative decoding、DSpark。
