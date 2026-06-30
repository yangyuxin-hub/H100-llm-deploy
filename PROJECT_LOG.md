# 大模型部署实习项目记录

## 项目背景

这个仓库用于记录我的实习项目：在 H100 GPU 服务器上部署大模型，并在实践过程中学习大模型推理相关的底层知识。

H100 服务器需要通过 SSH 远程连接。本地工作区主要用于整理部署脚本、配置文件、模型文件、项目笔记和学习记录。真正的服务启动、GPU 状态检查和推理验证，应当在远程 H100 服务器上完成。

## 主要目标

在 4 卡 H100 环境中使用 vLLM 部署大模型，对外提供 OpenAI 兼容的 API 服务，并理解部署过程中涉及的核心推理概念。

当前目标模型：

- Qwen3.6-27B-FP8
- DeepSeek-V4-Flash-DSpark

当前部署模式为互斥部署：同一时间只运行一个模型。4 卡环境下，Qwen 优先使用 `TP=4, DP=1`，并默认开启 MTP。

## 当前状态

截至 2026-06-30：

- 模型权重已经下载到本地 `models/` 目录。
- `Qwen3.6-27B-FP8` 已有 `config.json` 和 safetensors 权重文件，总大小约 29 GB。
- `DeepSeek-V4-Flash-DSpark` 已有 `config.json` 和 48/48 个 safetensors 分片，总大小约 156 GB。
- 当前本地会话中没有正在运行的 vLLM 服务。
- 当前本地环境无法查看 H100 GPU 状态，因为这里没有可用的 `nvidia-smi`。
- 下一步真实部署操作应在 SSH 连接到 H100 服务器后进行。

## 重要文件

- `config/serving.env`：共享部署配置。
- `scripts/download_models.sh`：从 ModelScope 下载模型权重。
- `scripts/start_qwen.sh`：启动 Qwen vLLM 服务。
- `scripts/start_deepseek.sh`：启动 DeepSeek vLLM 服务。
- `scripts/status.sh`：检查运行服务、模型权重完整性和 GPU 状态。
- `scripts/stop.sh`：停止当前运行的服务。
- `logs/`：运行日志和下载日志。
- `runtime/`：运行时 PID 文件。
- `models/`：已下载的模型权重，不要随意删除。
- `docs/superpowers/plans/2026-06-30-h100-llm-deployment.md`：H100 大模型部署执行计划。
- `docs/learning-notes/`：大模型知识点学习笔记，包括推理流程、部署概念、KV Cache、TP、FP8 等内容。

## 下一步任务

1. SSH 连接到 H100 服务器。
2. 确认远程运行环境：
   - `nvidia-smi`
   - Python 环境
   - vLLM 版本
   - CUDA / 驱动兼容性
3. 执行状态检查：

   ```bash
   bash scripts/status.sh
   ```

4. 启动其中一个模型：

   ```bash
   bash scripts/start_qwen.sh
   ```

   或：

   ```bash
   bash scripts/start_deepseek.sh
   ```

5. 验证 API 端点：

   ```bash
   curl http://127.0.0.1:8000/v1/models
   ```

6. 记录部署结果：
   - 启动耗时
   - GPU 显存占用
   - 错误日志，如果有
   - 成功请求示例
   - 修改过的配置项

## H100 机器分配前可以做的事

当前还没有拿到 H100 服务器时，重点是把“部署前准备、脚本检查、学习路线、实验记录模板”先做好。这样机器一分配下来，就可以直接验证，而不是现场临时补材料。

### 1. 整理部署前检查清单

- 明确远程服务器需要确认的信息：
  - GPU 数量和型号：`nvidia-smi`
  - H100 显存规格：80GB / 其他
  - NVIDIA Driver 版本
  - CUDA 版本
  - Python 版本
  - vLLM 版本
  - 是否有 Docker / nvidia-container-toolkit
  - 磁盘空间是否足够放模型权重和日志
- 提前准备一组服务器登录后的检查命令。

### 2. 检查和完善本地脚本

- 阅读并理解：
  - `config/serving.env`
  - `scripts/status.sh`
  - `scripts/start_qwen.sh`
  - `scripts/start_deepseek.sh`
  - `scripts/stop.sh`
- 确认脚本里的路径是否适合未来 H100 服务器。
- 明确哪些配置以后可能要根据服务器修改，例如：
  - `PROJECT_ROOT`
  - `MODEL_ROOT`
  - `TENSOR_PARALLEL_SIZE`
  - `QWEN_MAX_MODEL_LEN`
  - `DS_MAX_MODEL_LEN`
  - `*_GPU_MEM_UTIL`
- 修改脚本后运行：

  ```bash
  bash -n scripts/*.sh
  ```

### 3. 准备镜像和环境方案

- 明确部署时是直接在宿主机安装，还是使用 Docker 容器。
- 如果使用 Docker，提前列出需要拉取的镜像：
  - CUDA / PyTorch 基础镜像
  - vLLM OpenAI 镜像
  - 可能的 SGLang 镜像或自建镜像
- 记录镜像版本，不要只写 `latest`。
- 了解 `nvidia-container-toolkit` 的作用：让 Docker 容器可以访问 GPU。

### 4. 设计验证请求和压测方法

- 准备最小健康检查：

  ```bash
  curl http://127.0.0.1:8000/v1/models
  ```

- 准备一个最小 chat completion 请求。
- 学习 vLLM 的 benchmark 工具：
  - `benchmark_serving.py`
  - 首 token 延迟
  - 输出 tokens/s
  - 并发数
  - 请求成功率
- 先设计记录表，等拿到 H100 后直接填数据。

### 5. 学习底层概念

优先学习和当前项目直接相关的概念：

- FP8：为什么 H100 适合 FP8 推理。
- Tensor Parallelism：为什么 27B / MoE 模型需要多卡切分。
- KV Cache：为什么上下文越长显存越大。
- PagedAttention：vLLM 为什么能高效管理 KV Cache。
- MoE：DeepSeek 的“总参数”和“激活参数”为什么不同。
- Speculative Decoding：DSpark / MTP 这类加速方法在做什么。
- OpenAI-compatible API：为什么部署后可以用 `/v1/models`、`/v1/chat/completions`。

### 6. 做项目记录和汇报材料

- 继续维护 `PROJECT_LOG.md`。
- 阅读 `MODEL_DEPLOYMENT_RESEARCH.md`，把不懂的概念单独记下来。
- 准备一页实习汇报思路：
  - 项目目标
  - 当前进展
  - 部署架构
  - 待验证问题
  - 学习收获
- 机器分配前的成果可以是：
  - 部署脚本准备完成
  - 模型权重准备完成
  - 调研文档完成
  - 实验指标设计完成
  - H100 到手后的执行清单完成

## 学习目标

我希望不只是把模型跑起来，也要理解它为什么能跑起来。

重点学习内容：

- 模型权重是如何组织的：`config.json`、tokenizer 文件、safetensors 分片。
- 大模型推理流程：tokenizer、embedding、prefill、decode、KV Cache。
- FP8 是什么，以及为什么能降低显存占用。
- vLLM 如何加载模型并提供 OpenAI 兼容 API。
- 张量并行：为什么 4 卡环境下优先使用 `TP=4, DP=1`。
- 模型加载和推理时的 GPU 显存占用。
- KV Cache：为什么长上下文会消耗大量显存。
- 上下文长度的取舍，特别是 64K、256K 和 1M 上下文。
- 吞吐、延迟、batch size 和并发之间的关系。
- DeepSeek 的推测解码 / DSpark 机制。
- 如何阅读日志并排查模型启动失败。

## 模型下载格式说明

当前下载下来的模型目录，本质上是 Hugging Face Transformers 风格的模型仓库。可以把一个模型目录理解成一个“模型包”，里面不只有权重，还包括模型结构配置、分词器、生成参数、README 和许可证。

### 一个模型目录里通常有什么

以 `models/Qwen3.6-27B-FP8/` 和 `models/DeepSeek-V4-Flash-DSpark/` 为例：

| 文件 / 类型 | 作用 |
| --- | --- |
| `config.json` | 模型结构配置。告诉推理框架这个模型有多少层、hidden size 多大、attention heads 多少、上下文长度多少、量化方式是什么。vLLM 加载模型时会先读它。 |
| `configuration.json` | 通常是模型仓库额外提供的配置说明，可能给特定框架或远程代码使用。 |
| `generation_config.json` | 默认生成参数，例如温度、采样、特殊 token 等。实际部署时也可以在请求里覆盖。 |
| `tokenizer.json` | 分词器主体。负责把文字变成 token id，也负责把模型输出的 token id 还原成文字。 |
| `tokenizer_config.json` | 分词器配置。记录 special tokens、chat template 相关设置等。 |
| `vocab.json` / `merges.txt` | BPE 类分词器的词表和合并规则。Qwen 目录里有这些文件。 |
| `chat_template.jinja` | 聊天格式模板。告诉框架如何把 `role=user/assistant/system` 的消息拼成模型真正看到的 prompt。 |
| `*.safetensors` | 真正的模型权重。可以理解成巨大的参数矩阵文件。 |
| `model.safetensors.index.json` | 权重索引。告诉框架“某个参数名在哪个 safetensors 文件里”。 |
| `README.md` | 模型卡，说明模型能力、参数量、上下文长度、推荐部署方式等。 |
| `LICENSE` | 许可证。说明模型怎么用、能不能商用等。 |

### safetensors 是什么

`.safetensors` 是一种保存模型权重的文件格式。它的作用类似以前常见的 `.bin` / `.pt`，但更适合分发大模型权重。

可以把它理解成：

```text
safetensors = 模型参数矩阵的安全存储格式
```

它里面保存的是一堆 tensor，例如：

- embedding 权重
- attention 里的 q/k/v/o 权重
- FFN / MLP 权重
- norm 权重
- MoE expert 权重
- lm_head 输出层权重

推理框架启动时，会把这些权重加载到 CPU / GPU 内存里，然后才能进行推理。

### 为什么权重要拆成很多个文件

大模型权重太大，单个文件会非常大，不方便下载、校验、断点续传和加载。所以通常会拆成多个分片。

当前两个模型的分片方式不同：

```text
Qwen3.6-27B-FP8:
  layers-0.safetensors
  layers-1.safetensors
  ...
  layers-63.safetensors
  mtp.safetensors
  outside.safetensors

DeepSeek-V4-Flash-DSpark:
  model-00001-of-00048.safetensors
  model-00002-of-00048.safetensors
  ...
  model-00048-of-00048.safetensors
```

Qwen 的命名更接近“按层拆分”；DeepSeek 的命名是 Hugging Face 常见的“第几个分片 / 总共多少分片”。

### model.safetensors.index.json 是干什么的

这个文件是“权重地图”。推理框架不会靠猜文件名加载权重，而是读这个索引。

它大概长这样：

```json
{
  "weight_map": {
    "embed.weight": "model-00001-of-00048.safetensors",
    "layers.0.attn.wq_a.weight": "model-00002-of-00048.safetensors"
  }
}
```

意思是：

```text
参数 embed.weight 在 model-00001-of-00048.safetensors 里
参数 layers.0.attn.wq_a.weight 在 model-00002-of-00048.safetensors 里
```

所以只要索引文件和 safetensors 分片都完整，vLLM / Transformers 就能知道怎么把整个模型拼起来。

### Qwen 目录里的特殊文件

Qwen3.6-27B-FP8 目录里有：

- `layers-0.safetensors` 到 `layers-63.safetensors`：64 层语言模型主体权重。
- `outside.safetensors`：不属于单个普通层的权重，例如 embedding、lm_head、视觉编码器部分等。
- `mtp.safetensors`：MTP，也就是 multi-token prediction 相关权重，用于后续 speculative decoding / 加速实验。
- `preprocessor_config.json`、`video_preprocessor_config.json`：因为 Qwen3.6-27B 是带 Vision Encoder 的模型，这些文件和图像 / 视频预处理有关。
- `chat_template.jinja`：聊天消息格式模板。

Qwen 的 `config.json` 里能看到：

- `num_hidden_layers: 64`：语言模型有 64 层。
- `max_position_embeddings: 262144`：原生上下文长度 262K。
- `language_model_only: false`：这个包不只是纯文本语言模型，还包含视觉相关结构。
- `mtp_num_hidden_layers: 1`：包含 MTP 相关结构。

### DeepSeek 目录里的特殊文件

DeepSeek-V4-Flash-DSpark 目录里有：

- `model-00001-of-00048.safetensors` 到 `model-00048-of-00048.safetensors`：48 个权重分片。
- `model.safetensors.index.json`：权重索引。
- `encoding/`：DeepSeek 自己提供的编码相关代码和测试。
- `inference/`：官方最小推理示例代码。

DeepSeek 的 `config.json` 里能看到：

- `model_type: deepseek_v4`：模型类型是 DeepSeek V4。
- `num_hidden_layers: 43`：模型层数。
- `n_routed_experts: 256`：MoE 里有 256 个路由专家。
- `num_experts_per_tok: 6`：每个 token 会激活 6 个专家。
- `max_position_embeddings: 1048576`：最大上下文 1M。
- `expert_dtype: fp4`：专家参数使用 FP4，这也是脚本里要用 `DS_HF_OVERRIDES` 处理的原因。
- `dspark_*` 字段：和 DSpark 推测解码模块有关。

### 下载完整性怎么看

不要只看目录存在，要看关键文件是否齐全。

Qwen 至少要有：

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors.index.json
layers-*.safetensors
outside.safetensors
mtp.safetensors
```

DeepSeek 至少要有：

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors.index.json
model-00001-of-00048.safetensors
...
model-00048-of-00048.safetensors
```

当前项目里可以直接用：

```bash
bash scripts/status.sh
```

它会检查模型目录、`config.json` 和 safetensors 文件数量。

### 一句话总结

```text
config.json 决定模型长什么样；
tokenizer.json 决定文字怎么变成 token；
safetensors 保存真正的模型参数；
model.safetensors.index.json 告诉框架每个参数在哪个权重分片里；
README.md 告诉人这个模型是什么、怎么用。
```

## 学习日志模板

每次学习或部署后，可以使用下面的格式记录。

```markdown
## YYYY-MM-DD

### 今天做了什么

- 

### 观察到什么现象

- 

### 遇到的问题 / 报错

- 

### 学到了什么

- 

### 下一步

- 
```

## 记录原则

- 部署记录和学习记录先统一写在这个文件里，等内容太多时再拆分。
- 如果解决了一个错误，要同时记录错误现象和解决方法。
- 尽量记录具体证据：命令、日志、GPU 显存数字、配置变更。
- 不要把密钥、token、私有 SSH key 或服务器敏感信息写进仓库。
