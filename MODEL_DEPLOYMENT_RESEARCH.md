# 模型选择与部署方案调研

更新时间：2026-06-30

## 结论摘要

当前仓库的方向是合理的：先用 vLLM 在 4 卡 H100 上做 Qwen3.6-27B-FP8 和 DeepSeek-V4-Flash-DSpark 的互斥部署，作为实习项目的第一条主线。

更优的推进方式不是一开始就追求最复杂的生产方案，而是分阶段验证：

1. 先把 vLLM 基线跑通，记录启动、显存、吞吐和错误。
2. Qwen 默认支持 MTP，并继续测试 text-only 等优化参数。
3. 再用 SGLang 做同模型 A/B 测试，比较吞吐、延迟、显存和稳定性。
4. 如果后续目标变成生产级极致性能，再研究 TensorRT-LLM。

## 当前模型对比

| 模型 | 定位 | 优点 | 风险 / 注意点 | 建议 |
| --- | --- | --- | --- | --- |
| Qwen3.6-27B-FP8 | 27B 稠密模型，偏代码、Agent、通用推理 | 体积小，FP8 节省显存，官方给出 vLLM / SGLang 部署命令，适合作为第一阶段基线 | 长上下文会占用大量 KV Cache；MTP 参数需要实测 | 优先跑通，作为学习 vLLM、TP、KV Cache 的主模型 |
| DeepSeek-V4-Flash-DSpark | DeepSeek-V4-Flash + DSpark 推测解码模块 | MoE，284B 总参数、13B 激活参数，支持 1M 上下文方向，适合学习 MoE 和长上下文 | DSpark 不是新模型，而是附加推测解码模块；需要确认当前 vLLM / SGLang 是否真正利用了 DSpark 加速 | 第二阶段部署，先用 64K 上下文保守启动，再逐步加长上下文 |
| Qwen3.6-35B-A3B / FP8 变体 | Qwen3.6 MoE 模型，约 35B 总参数、约 3B 激活参数 | 适合学习 MoE、小激活参数、Agentic coding；可能比 27B 更有部署研究价值 | 仍然需要完整模型权重常驻显存，不能只按 3B 激活参数估算显存 | 可作为后续替代或扩展模型，不建议打断当前已下载模型的部署主线 |
| DeepSeek-V4-Pro | 更大 DeepSeek MoE，1.6T 总参数、49B 激活参数 | 能力更强，适合研究前沿 MoE | 4 卡 H100 很可能不是轻松部署对象，工程复杂度高 | 暂不作为实习第一阶段目标 |

## 推理框架对比

| 框架 | 适合阶段 | 优点 | 代价 | 建议 |
| --- | --- | --- | --- | --- |
| vLLM | 第一阶段基线 | OpenAI 兼容 API 简单；PagedAttention、连续批处理、TP 支持成熟；Qwen 官方推荐 | 极致性能不一定总是第一；新模型特性可能需要较新版本 | 当前主线继续用 vLLM |
| SGLang | 第二阶段 A/B 测试 | Qwen 官方给出 SGLang 命令；支持 TP、DP、MTP、工具调用等参数；适合对比吞吐 | 需要额外维护一套启动脚本和参数 | 建议新增 `start_qwen_sglang.sh` 做对比实验 |
| TensorRT-LLM | 第三阶段生产优化 | NVIDIA GPU 上性能潜力高；适合稳定模型的高性能部署 | 环境、模型转换、engine 构建复杂，学习成本高 | 先学习概念，不作为当前第一阶段实现目标 |
| Hugging Face TGI | 备选学习对象 | 生产特性完整，支持 TP、连续批处理、监控等 | 对 Qwen3.6 / DeepSeek 新特性的适配需要另查支持矩阵 | 可以作为横向了解，不建议当前切换 |

## 推荐部署路线

### 阶段 1：vLLM 基线

目标：先在 H100 服务器上证明模型能稳定启动并提供 API。

建议顺序：

1. SSH 到 H100。
2. 确认 `nvidia-smi`、CUDA、Python、vLLM 版本。
3. 跑 `bash scripts/status.sh`。
4. 启动双模型：

   ```bash
   bash scripts/start_qwen_fp8_docker.sh
   bash scripts/start_agents_docker.sh
   ```

5. 用 `/v1/models` 和 `/v1/chat/completions` 做基本验证。
6. 记录启动时间、显存、吞吐和日志。

### 阶段 2：Qwen 优化实验

目标：对比不同 vLLM 参数对性能和显存的影响。

可测试参数：

- `--language-model-only`：如果只做文本任务，可以跳过视觉编码器，给 KV Cache 留更多空间。
- `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`：测试 Qwen MTP。
- `--enable-prefix-caching`：适合多轮或前缀重复场景。
- 不同 `--max-model-len`：例如 32K、64K、128K、262K。
- 不同 `--gpu-memory-utilization`：例如 0.85、0.90、0.92。

### 阶段 3：SGLang 对比实验

目标：保留 vLLM 基线，同时增加 SGLang 对比。

Qwen 官方给出的 SGLang 思路：

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.6-27B-FP8 \
  --port 8000 \
  --tp-size 8 \
  --mem-fraction-static 0.8 \
  --context-length 262144 \
  --reasoning-parser qwen3
```

如果要测试 Qwen MTP，可按官方模型卡方向加入 NEXTN speculative 参数。

### 阶段 4：TensorRT-LLM 学习

目标：理解 NVIDIA 高性能推理栈，但暂不替换当前主线。

适合学习的问题：

- 为什么 TensorRT-LLM 需要构建 engine。
- FP8 / FP4 / INT4 量化和 H100 Tensor Core 的关系。
- TensorRT-LLM、Triton、NVIDIA Dynamo / NIM 之间是什么关系。
- 为什么生产优化会牺牲一些部署灵活性。

## 需要重点验证的问题

1. H100 服务器是否为 4 张 80GB，以及实际可见 GPU 数量。
2. GPU 之间是否有 NVLink；TP=4 对通信仍然敏感。
3. 当前 vLLM 版本是否满足 Qwen3.6 推荐版本。
4. DeepSeek-V4-Flash-DSpark 的 DSpark 模块是否被推理框架实际启用。
5. 1M 上下文是否真的需要；如果只是实习第一阶段，64K / 128K 更适合调试。
6. 服务目标是单用户学习、内部工具调用，还是多并发压测。

## 推荐教程和官方文档

### vLLM

- vLLM 并行与扩展文档：<https://docs.vllm.ai/en/stable/serving/parallelism_scaling/>
- vLLM OpenAI 兼容服务：<https://docs.vllm.ai/en/stable/serving/openai_compatible_server/>
- vLLM Quickstart：<https://docs.vllm.ai/en/latest/getting_started/quickstart/>
- vLLM Benchmark CLI：<https://docs.vllm.ai/en/latest/benchmarking/cli/>
- Qwen3.5 / Qwen3.6 vLLM Recipe：<https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html>

### Qwen

- Qwen3.6 GitHub：<https://github.com/QwenLM/Qwen3.6>
- Qwen3.6-27B-FP8 Hugging Face 模型卡：<https://huggingface.co/Qwen/Qwen3.6-27B-FP8>
- Qwen vLLM 部署文档：<https://qwen.readthedocs.io/en/latest/deployment/vllm.html>
- Qwen SGLang 部署文档：<https://qwen.readthedocs.io/en/latest/deployment/sglang.html>

### DeepSeek

- DeepSeek-V4-Flash-DSpark Hugging Face 模型卡：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark>
- DeepSpec / DSpark 代码仓库：<https://github.com/deepseek-ai/DeepSpec>

### SGLang

- SGLang Server Arguments：<https://docs.sglang.io/docs/advanced_features/server_arguments>
- SGLang Multi-Node Deployment：<https://docs.sglang.io/docs/deployment/multi_node>

### TensorRT-LLM / NVIDIA

- TensorRT-LLM 文档：<https://nvidia.github.io/TensorRT-LLM/>
- TensorRT-LLM GitHub：<https://github.com/NVIDIA/TensorRT-LLM>
- NVIDIA TensorRT-LLM H100 介绍：<https://developer.nvidia.com/blog/nvidia-tensorrt-llm-supercharges-large-language-model-inference-on-nvidia-h100-gpus/>

### TGI

- Hugging Face TGI 文档：<https://huggingface.co/docs/text-generation-inference/en/index>
- TGI Tensor Parallelism：<https://huggingface.co/docs/text-generation-inference/en/conceptual/tensor_parallelism>

## 我的建议

先不要换掉当前方案。当前项目已经有模型权重、vLLM 脚本和状态检查脚本，最适合实习第一阶段。

更好的做法是把“更优方案”变成实验矩阵：

1. Qwen vLLM 标准模式。
2. Qwen vLLM + text-only。
3. Qwen vLLM + MTP。
4. Qwen SGLang 标准模式。
5. DeepSeek vLLM 64K。
6. DeepSeek vLLM 更长上下文。

每一组都记录：

- 启动是否成功
- 首 token 延迟
- 输出 tokens/s
- GPU 显存
- 最大并发
- 是否出现错误
- 配置参数

这样不仅能找到更优部署方案，也能把学习过程沉淀成一份很有价值的实习记录。
