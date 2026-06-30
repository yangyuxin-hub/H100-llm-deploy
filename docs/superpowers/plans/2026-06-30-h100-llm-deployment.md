# H100 大模型部署计划

更新日期：2026-06-30

## 目标

在远程 4 卡 H100 Ubuntu 服务器上部署 Qwen3.6-27B-FP8 和 DeepSeek-V4-Flash-DSpark，并通过 vLLM 提供 OpenAI 兼容 API 服务。

当前优先级：

1. 先跑通 Qwen3.6-27B-FP8。
2. Qwen 必须支持 MTP。
3. 最后部署 DeepSeek-V4-Flash-DSpark。

## 部署主线

1. 主推理框架使用 vLLM。
2. 服务接口使用 OpenAI 兼容 API。
3. Qwen 和 DeepSeek 互斥部署，同一时间只运行一个模型。
4. 服务端口统一使用 `8000`。
5. 4 卡 H100 优先使用 `TP=4, DP=1`，也就是 4 张 GPU 共同跑一个 Qwen 实例。
6. 先用 Python / uv 原生环境跑通现有脚本。
7. Docker 作为环境隔离和后续生产化备选方案。

## 版本信息

| 项目 | 建议版本或要求 | 说明 |
| --- | --- | --- |
| 操作系统 | Ubuntu 22.04 LTS 或 Ubuntu 24.04 LTS | 以服务器实际系统为准。 |
| GPU | 4 x NVIDIA H100 | 当前脚本按 4 卡设计。 |
| NVIDIA Driver | 以 `nvidia-smi` 正常识别 H100 为准 | Driver 必须兼容所用 CUDA / PyTorch / vLLM。 |
| CUDA | CUDA 12.x | 原生环境由 vLLM / PyTorch 后端选择；Docker 环境由镜像提供 CUDA runtime。 |
| Python | Python 3.12 推荐 | vLLM 最新文档支持 Linux + Python 3.10 到 3.13。 |
| uv | 最新稳定版 | 用于创建虚拟环境和安装 vLLM。 |
| vLLM | `>=0.19.0` | Qwen3.6 官方模型卡推荐。 |
| SGLang | `>=0.5.10` | 备选推理框架，不作为第一主线。 |
| Docker | 可选 | 如果使用容器部署，需要 Docker Engine。 |
| NVIDIA Container Toolkit | 可选 | Docker 使用 GPU 时需要。 |
| vLLM Docker 镜像 | `vllm/vllm-openai:latest` 用于首次验证 | 跑通后再固定具体镜像 tag。 |
| Qwen 模型 | `Qwen/Qwen3.6-27B-FP8` | 本地路径为 `models/Qwen3.6-27B-FP8`。 |
| Qwen 精度 | FP8 | 模型权重为 fine-grained FP8。 |
| Qwen 上下文 | 原生 262144 tokens | 初次部署先使用原生 262K，不直接扩到 1M。 |
| Qwen 并行策略 | `TP=4, DP=1` | 先保证 MTP 和长上下文稳定；`TP=2, DP=2` 后续做吞吐实验。 |
| Qwen MTP | 默认开启 | 使用 `qwen3_next_mtp` speculative 配置。 |
| DeepSeek 模型 | `deepseek-ai/DeepSeek-V4-Flash-DSpark` | 本地路径为 `models/DeepSeek-V4-Flash-DSpark`。 |

## 本项目关键配置

这些值来自 `config/serving.env`：

| 配置 | 当前值 | 含义 |
| --- | --- | --- |
| `PROJECT_ROOT` | `/home/yangyuxin/llm-deploy` | 项目根目录。 |
| `MODEL_ROOT` | `/home/yangyuxin/llm-deploy/models` | 模型权重目录。 |
| `MODEL_PATH_QWEN` | `models/Qwen3.6-27B-FP8` | Qwen 权重目录。 |
| `QWEN_SERVED_NAME` | `qwen3.6-27b-fp8` | API 请求里使用的模型名。 |
| `SERVE_PORT` | `8000` | vLLM 服务端口。 |
| `TENSOR_PARALLEL_SIZE` | `4` | 使用 4 卡张量并行。 |
| `DATA_PARALLEL_SIZE` | `1` | 只启动 1 份模型实例。 |
| `QWEN_MAX_MODEL_LEN` | `262144` | Qwen 原生上下文长度。 |
| `QWEN_GPU_MEM_UTIL` | `0.90` | vLLM 可使用的 GPU 显存比例。 |
| `QWEN_REASONING_PARSER` | `qwen3` | Qwen thinking / reasoning 解析器。 |
| `QWEN_TOOL_PARSER` | `qwen3_coder` | Qwen 工具调用解析器。 |
| `QWEN_ENABLE_MTP` | `1` | Qwen 默认开启 MTP。 |
| `QWEN_SPECULATIVE_CONFIG` | `{"method":"qwen3_next_mtp","num_speculative_tokens":2}` | Qwen 官方 MTP speculative 配置。 |

## 部署前理解

### TP=4 和 DP=1 是什么

`TP=4` 是 tensor parallel size 等于 4。

意思是：一个 Qwen 服务同时使用 4 张 H100，每张 GPU 负责模型的一部分计算。

它不是 4 张卡各跑一个 Qwen，而是 4 张卡共同跑一个 Qwen。

`DP=1` 是 data parallel size 等于 1，表示只启动 1 份 Qwen 实例。

4 卡情况下先使用：

```text
TP=4, DP=1
```

不优先使用 `DP=4`，因为 `DP=4` 等价于 4 份单卡模型实例，更吃单卡显存，不适合作为 262K 长上下文和 MTP 的第一方案。

不优先使用 `DP=2`，因为 `DP=2` 通常会配成 `TP=2, DP=2`，也就是 2 份实例、每份 2 卡。它更适合高并发吞吐实验，但单个实例的显存余量比 `TP=4, DP=1` 少。

### 为什么 Qwen 先跑

1. Qwen3.6-27B-FP8 权重约 29GB，比 DeepSeek 更容易先跑通。
2. Qwen 官方给了 vLLM、SGLang、MTP、text-only 的部署参考。
3. Qwen 适合先验证环境、脚本、API 和多卡通信。

### 为什么先不扩到 1M 上下文

Qwen3.6 原生支持 262K，上下文可以通过 YaRN 扩展到约 1M。

初次部署先用 262K，原因是：

1. 原生配置风险更低。
2. 262K 已经足够验证长上下文能力。
3. 1M 会显著增加 KV Cache 显存压力。
4. YaRN 还需要额外配置，适合后续实验。

## H100 到手后的第一步：连接服务器

1. SSH 连接远程服务器。

   ```bash
   ssh 用户名@服务器地址
   ```

2. 进入项目目录。

   ```bash
   cd /home/yangyuxin/llm-deploy
   ```

3. 确认当前目录正确。

   ```bash
   pwd
   ls
   ```

期望看到：

```text
/home/yangyuxin/llm-deploy
AGENTS.md
PROJECT_LOG.md
config
models
scripts
```

## 第二步：检查服务器基础环境

1. 查看 Ubuntu 版本。

   ```bash
   lsb_release -a
   ```

2. 查看内核版本。

   ```bash
   uname -a
   ```

3. 查看磁盘空间。

   ```bash
   df -h
   ```

4. 查看内存。

   ```bash
   free -h
   ```

要记录的信息：

1. Ubuntu 版本。
2. 磁盘剩余空间。
3. 内存大小。
4. 项目目录所在磁盘是否足够存模型、日志和缓存。

## 第三步：检查 GPU 和驱动

1. 查看 GPU。

   ```bash
   nvidia-smi
   ```

2. 查看 GPU 数量。

   ```bash
   nvidia-smi -L
   ```

3. 查看 GPU 拓扑。

   ```bash
   nvidia-smi topo -m
   ```

期望看到：

1. 服务器能识别 4 张 H100。
2. `nvidia-smi` 能显示 Driver Version 和 CUDA Version。
3. 没有 GPU 掉卡。
4. GPU 之间通信拓扑正常。

如果 `nvidia-smi` 不可用，先不要部署模型，先解决驱动或服务器环境问题。

## 第四步：准备 Python / vLLM 环境

推荐先用 Python / uv 原生环境跑通，因为项目现有脚本直接调用 `vllm serve`。

1. 查看 Python。

   ```bash
   python3 --version
   ```

2. 安装或检查 uv。

   ```bash
   uv --version
   ```

3. 如果没有 uv，安装 uv。

   ```bash
   python3 -m pip install --user uv
   ```

4. 创建 Python 3.12 虚拟环境。

   ```bash
   uv venv --python 3.12 --seed .venv
   ```

5. 进入虚拟环境。

   ```bash
   source .venv/bin/activate
   ```

6. 安装 vLLM。

   ```bash
   uv pip install vllm --torch-backend=auto
   ```

7. 检查 vLLM 版本。

   ```bash
   vllm --version
   ```

期望结果：

1. Python 版本为 3.12。
2. `vllm --version` 可用。
3. vLLM 版本不低于 `0.19.0`。

## 第五步：Docker 备选检查

如果服务器要求用 Docker，或者原生环境依赖冲突，再走 Docker 方案。

1. 检查 Docker。

   ```bash
   docker --version
   ```

2. 检查 NVIDIA Container Toolkit。

   ```bash
   nvidia-container-cli --version
   ```

3. 拉取 vLLM OpenAI 服务镜像。

   ```bash
   docker pull vllm/vllm-openai:latest
   ```

4. 测试 Docker 是否能访问 GPU。

   ```bash
   docker run --rm --gpus all --ipc=host --entrypoint nvidia-smi vllm/vllm-openai:latest
   ```

说明：

1. Docker 方案后续可以更稳定复现环境。
2. 但当前项目脚本是原生 `vllm serve` 方案。
3. 初次部署优先使用原生环境，更方便看日志和调试。
4. 如果后续使用 Docker 做正式部署，要把 `latest` 固定成实际验证通过的镜像 tag。

## 第六步：检查模型权重

1. 运行项目状态检查。

   ```bash
   bash scripts/status.sh
   ```

2. 检查 Qwen 模型目录大小。

   ```bash
   du -sh models/Qwen3.6-27B-FP8
   ```

3. 检查 Qwen 权重文件。

   ```bash
   find models/Qwen3.6-27B-FP8 -maxdepth 1 -name "*.safetensors" | wc -l
   ```

4. 检查关键配置文件。

   ```bash
   ls models/Qwen3.6-27B-FP8/config.json
   ls models/Qwen3.6-27B-FP8/tokenizer.json
   ls models/Qwen3.6-27B-FP8/model.safetensors.index.json
   ```

期望结果：

1. Qwen 模型目录存在。
2. Qwen 模型目录约 29GB。
3. 能看到 `layers-*.safetensors`、`outside.safetensors`、`mtp.safetensors`。
4. `config.json`、tokenizer 文件、index 文件都存在。

## 第七步：启动 Qwen 稳定基线

1. 停掉已有服务。

   ```bash
   bash scripts/stop.sh
   ```

2. 启动 Qwen。

   ```bash
   bash scripts/start_qwen.sh
   ```

3. 查看日志。

   ```bash
   tail -f logs/qwen3.6-27b-fp8.log
   ```

4. 查看服务状态。

   ```bash
   bash scripts/status.sh
   ```

期望结果：

1. vLLM 进程启动成功。
2. 健康检查通过。
3. `http://127.0.0.1:8000/v1/models` 可访问。
4. 4 张 H100 都有显存占用。

当前基线配置：

```text
框架：vLLM
模型：Qwen3.6-27B-FP8
接口：OpenAI-compatible API
端口：8000
TP：4
DP：1
max_model_len：262144
gpu_memory_utilization：0.90
reasoning_parser：qwen3
tool_call_parser：qwen3_coder
MTP：enabled
```

## 第八步：验证 OpenAI 兼容 API

1. 查看模型列表。

   ```bash
   curl http://127.0.0.1:8000/v1/models
   ```

2. 发送最小对话请求。

   ```bash
   curl http://127.0.0.1:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "qwen3.6-27b-fp8",
       "messages": [
         {"role": "user", "content": "用一句话介绍你自己。"}
       ],
       "temperature": 0.7,
       "top_p": 0.8,
       "max_tokens": 256
     }'
   ```

3. 测试流式输出。

   ```bash
   curl http://127.0.0.1:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "qwen3.6-27b-fp8",
       "messages": [
         {"role": "user", "content": "写一个 Python 快速排序函数。"}
       ],
       "temperature": 0.6,
       "top_p": 0.95,
       "max_tokens": 1024,
       "stream": true
     }'
   ```

要记录：

1. 请求是否成功。
2. 首 token 延迟。
3. 输出速度。
4. 是否出现 `<think>` 内容。
5. 返回格式是否符合 OpenAI Chat Completions。

## 第九步：记录显存和性能

1. 启动后查看 GPU 显存。

   ```bash
   nvidia-smi
   ```

2. 发请求时观察 GPU。

   ```bash
   watch -n 1 nvidia-smi
   ```

3. 记录日志。

   ```bash
   tail -100 logs/qwen3.6-27b-fp8.log
   ```

每次实验记录：

1. vLLM 版本。
2. Python 版本。
3. Driver Version。
4. CUDA Version。
5. GPU 型号和数量。
6. 启动耗时。
7. 每张 GPU 显存占用。
8. 首 token 延迟。
9. 输出 tokens/s。
10. 是否报错。

## 第十步：Qwen text-only 优化

适用场景：

1. 只做文本对话。
2. 只做代码生成。
3. 只做 Agent / 工具调用。
4. 不需要图片或视频输入。

优化动作：

1. 在基线服务稳定后，测试 `--language-model-only`。
2. 对比开启前后的显存占用。
3. 对比开启前后的启动速度。
4. 对比开启前后的请求速度。
5. 如果不需要多模态输入，将 text-only 作为 Qwen 常用方案。

观察重点：

1. 显存是否下降。
2. KV Cache 可用空间是否增加。
3. 长上下文请求是否更稳定。
4. 普通文本输出质量是否保持正常。

## 第十一步：Qwen MTP 验证

适用场景：

1. Qwen 必须支持 MTP。
2. 先验证 `TP=4, DP=1` 下的 MTP。
3. 再决定是否测试 `TP=2, DP=2`。

推荐先测试官方 Qwen MTP 配置：

```text
speculative_config：{"method":"qwen3_next_mtp","num_speculative_tokens":2}
```

验证步骤：

1. 停止当前 Qwen 服务。
2. 确认 `QWEN_ENABLE_MTP=1`。
3. 启动 Qwen 服务。
4. 发送短文本请求。
5. 发送代码生成请求。
6. 发送较长上下文请求。
7. 记录 MTP 开启后的速度和稳定性。

判断标准：

1. 如果 `TP=4, DP=1` + MTP 稳定，作为 Qwen 默认方案。
2. 如果长上下文请求报错，先降低上下文长度到 128K。
3. 如果需要更高并发，再测试 `TP=2, DP=2`。

## Qwen 采样参数建议

### 通用思考任务

适合复杂问答、推理、方案分析。

```text
temperature=1.0
top_p=0.95
top_k=20
presence_penalty=0.0
repetition_penalty=1.0
```

### 精确代码任务

适合代码生成、代码修改、WebDev。

```text
temperature=0.6
top_p=0.95
top_k=20
presence_penalty=0.0
repetition_penalty=1.0
```

### 普通指令任务

适合不需要长思考的问答和格式化输出。

```text
temperature=0.7
top_p=0.80
top_k=20
presence_penalty=1.5
repetition_penalty=1.0
```

## DeepSeek 部署计划

DeepSeek 放在 Qwen 跑通之后。

1. 停止 Qwen 服务。
2. 检查 DeepSeek 权重完整性。
3. 使用保守上下文长度启动 DeepSeek。
4. 检查 vLLM 日志。
5. 验证 OpenAI 兼容 API。
6. 记录启动耗时、显存占用和基础请求结果。

当前 DeepSeek 策略：

1. 默认上下文先用 `65536`。
2. 不直接上 1M 上下文。
3. 使用 `TP=4, DP=1` 作为第一方案。
4. 使用 `DS_HF_OVERRIDES` 处理 `expert_dtype` 兼容问题。

## 常见问题处理顺序

### vLLM 命令不存在

1. 确认已经进入虚拟环境。
2. 运行 `which vllm`。
3. 运行 `vllm --version`。
4. 如果没有安装，重新安装 vLLM。

### GPU 数量不足

1. 运行 `nvidia-smi -L`。
2. 确认是否真的有 4 张 H100。
3. 如果只有部分 GPU 可见，检查 `CUDA_VISIBLE_DEVICES`。
4. 如果服务器没有识别全部 GPU，联系机器管理员。

### 模型权重不完整

1. 运行 `bash scripts/status.sh`。
2. 检查模型目录大小。
3. 检查 `.safetensors` 数量。
4. 重新下载或同步缺失文件。

### 启动时显存不足

1. 先确认没有其他进程占用 GPU。
2. 降低 `gpu_memory_utilization`。
3. 降低 `max_model_len` 到 128K。
4. 测试 text-only 模式。
5. 最后再考虑其他部署框架。

### API 无法访问

1. 检查 vLLM 进程是否还在。
2. 检查日志是否报错。
3. 检查端口 `8000` 是否被占用。
4. 检查 curl 地址是否为 `127.0.0.1:8000`。

## 安全注意事项

1. 初次部署只在服务器本机访问 `127.0.0.1:8000`。
2. 不要直接把端口暴露到公网。
3. 如果需要外部访问，先加防火墙、API key 或反向代理。
4. 不要把 SSH key、token、服务器地址写进项目文档。

## 项目记录

每次部署或测试后更新 `PROJECT_LOG.md`。

建议记录格式：

```text
日期：
服务器：
模型：
部署方式：
Python 版本：
vLLM 版本：
Driver Version：
CUDA Version：
启动参数：
启动结果：
显存占用：
API 测试结果：
问题：
解决方法：
学到的概念：
```

## 参考资料

1. Qwen3.6-27B-FP8 模型卡：`https://huggingface.co/Qwen/Qwen3.6-27B-FP8`
2. vLLM Quickstart：`https://docs.vllm.ai/en/latest/getting_started/quickstart/`
3. vLLM Docker：`https://docs.vllm.ai/en/stable/deployment/docker/`
4. NVIDIA Container Toolkit：`https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html`
5. CUDA Compatibility：`https://docs.nvidia.com/deploy/cuda-compatibility/index.html`
