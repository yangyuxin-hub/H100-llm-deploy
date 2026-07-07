# 大模型部署实习项目记录

## 项目背景

这个仓库用于记录我的实习项目：在 H100 GPU 服务器上部署大模型，并在实践过程中学习大模型推理相关的底层知识。

H100 服务器需要通过 SSH 远程连接。本地工作区主要用于整理部署脚本、配置文件、模型文件、项目笔记和学习记录。真正的服务启动、GPU 状态检查和推理验证，应当在远程 H100 服务器上完成。

## 主要目标

在 4 卡 H100 环境中使用 vLLM 部署大模型，对外提供 OpenAI 兼容的 API 服务，并理解部署过程中涉及的核心推理概念。远程 node1 已通过 `nvidia-smi -L` 验证为 4 张 NVIDIA H100 80GB HBM3。

当前目标模型：

- Qwen3.6-27B-FP8（主模型,端口 8000,GPU 0-1）
- Agents-A1-FP8（小模型,端口 8001,GPU 2-3）

当前部署模式为双模型并行部署：两个模型同时运行,各占 2 张 H100（TP=2）。两个模型都启用 MTP 投机解码和 Function Calling。详见 2026-07-07 章节。

## 当前状态

截至 2026-07-07：

- 当前目标模型：**Qwen3.6-27B-FP8** + **Agents-A1-FP8**（双模型并行部署）。
- 远程 H100 服务器 node1：4× H100 80GB HBM3，driver 550.144.03，不升级不重启。
- 远程 `/mnt/nvme0/models` 下有 4 个模型：Qwen3.6-27B-FP8、Qwen3.6-27B-NVFP4、Agents-A1-FP8、DeepSeek-V4-Flash-DSpark。
- vLLM Docker 镜像 `vllm/vllm-openai:v0.24.0-cu129-ubuntu2404`（v0.24.0, CUDA 12.9，原生兼容 driver 550）已加载到远程。
- **双模型并行部署已验证通过**：Qwen3.6-27B-FP8 (GPU 0-1, 端口 8000) + Agents-A1-FP8 (GPU 2-3, 端口 8001)。
- One API 网关 (http://10.30.75.58:18082/) 已配置两个渠道。
- opencode 已接入两个模型,配置在 `~/.config/opencode/opencode.json`。
- 部署和学习笔记见 `docs/learning-notes/04-dual-model-parallel-deployment.md`。

### Qwen3.6-27B-NVFP4 当前配置（cu129 原生版本，已验证）

- 镜像：vllm/vllm-openai:v0.24.0-cu129-ubuntu2404（CUDA 12.9，无需 cuda-compat 层）
- TP=4，max-model-len=32768，gpu-memory-utilization=0.90
- `--quantization modelopt` + `TORCH_COMPILE_DISABLE=1`（绕过 torch.compile NaN bug）
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`（启用 decode-only CUDA graph）
- `--attention-backend FLASH_ATTN` + `--gdn-prefill-backend triton`（绕过 FlashInfer GDN NaN）
- `--speculative-config` MTP 推测解码（num_speculative_tokens=3，单用户吞吐更优）
- 性能：单请求 512 tokens 平均 162.0 tok/s，256 tokens 平均 162.1 tok/s，流式 TTFT 约 0.11-0.14s，启动健康检查约 210s

### 已知限制

- torch.compile/inductor 仍不可用：完整 `VLLM_COMPILE + FULL_AND_PIECEWISE` 会导致 NaN / 连续 `!`；当前只启用 decode-only CUDA graph
- max-model-len 从 256K 降到 32K：为 MTP 腾显存
- 直接按官方形式使用模型 ID `nvidia/Qwen3.6-27B-NVFP4` 启动时，容器需要访问 Hugging Face 拉取 `config.json`；当前远程网络请求失败，未进入模型加载阶段
- DeepSeek-V4-Flash-DSpark 尚未测试

## 2026-07-03 官方基础命令复测记录

### 目标

用户要求不加额外 workaround，按官方基础方式启动：

```bash
vllm serve nvidia/Qwen3.6-27B-NVFP4 \
  --port 8000 \
  --quantization modelopt \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

Docker 镜像按项目确认使用：

```bash
vllm/vllm-openai:v0.24.0-cu129-ubuntu2404
```

### 执行过程

先停止原有容器，只做容器级 stop/rm，没有重启服务器，也没有改动模型权重：

```bash
cd /root/llm-deploy && bash scripts/stop.sh
docker stop qwen-nvfp4-vllm
docker rm qwen-nvfp4-vllm
```

随后按 Docker 镜像 entrypoint（`["vllm","serve"]`）启动：

```bash
docker run -d \
  --name qwen-nvfp4-vllm \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  vllm/vllm-openai:v0.24.0-cu129-ubuntu2404 \
  nvidia/Qwen3.6-27B-NVFP4 \
  --port 8000 \
  --quantization modelopt \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

### 结果

- 容器 ID：`490a133e65e2039a8a65b55800fa558f4e428aa2e26970ca1a2689f5818f535b`
- 容器约 270 秒后退出，`exit=1`，`oom=false`
- GPU 未进入加载阶段，4 张 H100 仍约 `5 MiB / 81559 MiB`
- 日志确认 vLLM 收到的非默认参数只有：
  - `model_tag='nvidia/Qwen3.6-27B-NVFP4'`
  - `model='nvidia/Qwen3.6-27B-NVFP4'`
  - `max_model_len=262144`
  - `quantization='modelopt'`
  - `reasoning_parser='qwen3'`

### 错误信息

关键日志：

```text
'[Errno 99] Cannot assign requested address' thrown while requesting HEAD https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4/resolve/main/config.json
OSError: Can't load the configuration of 'nvidia/Qwen3.6-27B-NVFP4'
```

### 判断

这次官方基础命令没有跑到模型加载，也没有触发 256K 显存压力或推理正确性验证；失败点在容器内通过 Hugging Face 模型 ID 获取 `config.json`。如果继续坚持“官方参数不加 workaround”，下一步应只解决模型来源问题：要么让容器能访问 Hugging Face，要么把官方命令里的模型 ID 换成本地已下载路径 `/mnt/nvme0/models/Qwen3.6-27B-NVFP4`。后者不是完全同一条官方命令，但不改变 vLLM 运行参数。

### 离线等价复测：官方参数 + 本地 snapshot 路径

为排除远程无网络带来的 Hugging Face 访问失败，使用本地完整模型目录替换 repo id，但不增加 MTP、workaround、TP 或其他 vLLM 参数：

```bash
docker run -d \
  --name qwen-official-clean \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v /mnt/nvme0/models:/mnt/nvme0/models:ro \
  vllm/vllm-openai:v0.24.0-cu129-ubuntu2404 \
  /mnt/nvme0/models/Qwen3.6-27B-NVFP4 \
  --port 8000 \
  --quantization modelopt \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

结果：

- 容器 ID：`00a81b568a1817cc73e4a333f9ac658e4dc0f04d4600560a6e72b59e95610a8b`
- 容器退出，`exit=1`，`oom=false`
- 已进入模型加载阶段，说明本地 snapshot 文件足够，联网问题已绕过
- 单进程/默认 TP=1 下模型加载占用约 `20.0 GiB`
- KV cache 规划显示：
  - `GPU KV cache size: 1,496,213 tokens`
  - `Maximum concurrency for 262,144 tokens per request: 5.71x`

关键错误：

```text
ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (976).
Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed.
Please lower max_num_seqs to at most 976 or increase gpu_memory_utilization.
```

判断：

- 这次不是文件缺失，也不是 Hugging Face 网络问题。
- 在完全官方 vLLM 参数、默认 TP=1、默认 `max_num_seqs=1024` 下，vLLM 的 CUDA graph capture 因 Mamba cache block 数不足而拒绝启动。
- 如果继续沿“官方参数尽量不污染”的方向测试，下一步最小变量只有两类：
  1. 资源适配：添加 `--tensor-parallel-size 4`，让 4 张 H100 参与部署，但仍不加 bug workaround。
  2. 保持 TP=1：按报错降低 `--max-num-seqs` 到 `976` 或更低，或提高 `--gpu-memory-utilization`。

### 离线等价复测：官方参数 + 本地 snapshot 路径 + TP=4

用户确认 `--tensor-parallel-size 4` 是 4 卡硬件资源适配，不属于污染配置。继续使用独立容器名 `qwen-official-clean`，不写入 runtime PID，不使用项目启动脚本：

```bash
docker run -d \
  --name qwen-official-clean \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v /mnt/nvme0/models:/mnt/nvme0/models:ro \
  vllm/vllm-openai:v0.24.0-cu129-ubuntu2404 \
  /mnt/nvme0/models/Qwen3.6-27B-NVFP4 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --quantization modelopt \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

结果：

- 容器 ID：`2c1372c839650bbd1fe0d0654a52f538083b3eef23a8e9d432ce69d0d69e70b7`
- `/v1/models` 健康检查通过，约 `210s` ready
- 返回 `max_model_len=262144`
- GPU 显存约 `76419-76429 MiB / 81559 MiB`
- 因为没有写 runtime PID，`scripts/status.sh` 的“运行中的服务”部分不显示该手工容器，但 GPU 显存可见其占用
- vLLM 配置确认：
  - `tensor_parallel_size=4`
  - `quantization=modelopt_mixed`
  - `max_seq_len=262144`
  - `speculative_config=None`
  - `enforce_eager=False`
  - 默认 `torch.compile/inductor`
  - 默认 `cudagraph_mode=FULL_AND_PIECEWISE`
  - 默认 `Using FlashInfer GDN prefill kernel`
- KV cache：
  - `GPU KV cache size: 7,859,720 tokens`
  - `Maximum concurrency for 262,144 tokens per request: 29.98x`
- engine 初始化：
  - `init engine ... took 108.66 s`
  - `compilation: 81.33 s`

短推理验证：

```bash
POST /v1/chat/completions
model="/mnt/nvme0/models/Qwen3.6-27B-NVFP4"
prompt="用一句中文回答：1+1等于几？"
max_tokens=64
temperature=0
```

结果：HTTP 成功返回，但输出仍为连续 `!`：

```json
{
  "content": "",
  "reasoning_prefix": "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
  "completion_tokens": 64,
  "all_bang": true
}
```

判断：

- `--tensor-parallel-size 4` 解决了 TP=1 下 Mamba cache blocks 不足导致的启动失败。
- 官方参数 + TP=4 + 本地 snapshot 可以启动到 ready，但推理正确性失败，复现连续 `!`。
- 这进一步说明此前 workaround 不是“脚本污染”造成的，而是在这台 H100 + vLLM 0.24.0 + Qwen3.6 NVFP4 上为了正确输出需要绕开默认 `torch.compile/inductor` 或 FlashInfer GDN 路径。

## 2026-07-03 Qwen NVFP4 单用户吞吐优化记录

### 背景

远程曾运行手工实验容器 `qwen-nvfp4-expC3`。该容器使用 `--compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE"}'`，API 可访问但推理输出为连续 `!`，属于已知 torch.compile / CUDA graph 导致的 NaN logits 路径，不能作为有效吞吐基线。

同时发现远程 `/root/llm-deploy/config/serving.env` 仍是旧 FP8/latest 配置，本地项目配置才是当前 NVFP4/cu129 正确配置。已将本地项目同步到远程，不同步 `models/`、`logs/`、`runtime/`。

### 执行的命令

```bash
cd /root/llm-deploy && bash scripts/stop.sh
docker stop qwen-nvfp4-expC3
docker rm qwen-nvfp4-expC3
rsync -az --exclude 'models/' --exclude 'logs/' --exclude 'runtime/' /home/yangyuxin/llm-deploy/ root@10.16.11.24:/root/llm-deploy/
cd /root/llm-deploy && bash -n scripts/*.sh
cd /root/llm-deploy && bash scripts/start_qwen_docker.sh
```

### MTP2 基线

- 配置：`num_speculative_tokens=2`，`--enforce-eager`，`FLASH_ATTN`，`--gdn-prefill-backend triton`。
- 正确性：不再输出连续 `!`。
- 启动：健康检查约 200 秒；engine 初始化 67.90 秒。
- KV cache：4,848,571 tokens；32K 请求最大并发估算 147.97x。
- 单用户非流式吞吐：
  - 64 tokens：29.86 tok/s
  - 256 tokens：28.76 tok/s
  - 512 tokens：28.25 tok/s
- 流式 TTFT：约 0.11-0.20 秒。

### MTP3 实验

只把 `num_speculative_tokens` 从 2 改成 3，其余参数保持不变。

- 正确性：正常，不再输出连续 `!`。
- 启动：健康检查约 200 秒；engine 初始化 68.86 秒。
- KV cache：4,407,792 tokens；32K 请求最大并发估算 134.52x。
- 单用户非流式吞吐：
  - 64 tokens：34.64 tok/s（较 MTP2 +16.0%）
  - 256 tokens：33.22 tok/s（较 MTP2 +15.5%）
  - 512 tokens：32.40 tok/s（较 MTP2 +14.7%）
- 流式 TTFT：约 0.11-0.20 秒，基本不变。
- 代价：KV cache 下降约 9.1%，对单用户优先目标可以接受。

### 当前结论

在单用户最快响应目标下，MTP3 比 MTP2 更合适。已将 `config/serving.env` 的 `QWEN_SPECULATIVE_CONFIG` 更新为 `num_speculative_tokens=3`。仍需注意：CUDA graph / torch.compile 路径会触发 NaN 输出，不能作为有效优化，除非后续 vLLM 修复并重新验证正确性。

## 2026-07-03 Qwen NVFP4 CUDA graph 解耦实验

### 背景

用户明确要求为了性能必须启用 CUDA graph。根据 vLLM 官方文档，`--enforce-eager` 会同时关闭 `torch.compile` 和 CUDA graph；但 `TORCH_COMPILE_DISABLE=1` 只关闭 `torch.compile`。同时 vLLM 新 CUDA graph 设计支持 `FULL_DECODE_ONLY`，即只对 decode 阶段捕获 full CUDA graph，不要求 piecewise compile。

### 实验配置

只改变编译/graph 相关参数，其余保留 MTP3 最优配置：

```bash
--env TORCH_COMPILE_DISABLE=1
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

不再使用 `--enforce-eager`。

### 关键日志证据

- `TORCH_COMPILE_DISABLE is set, disabling torch.compile`
- `cudagraph_mode=<CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>`
- `Capturing CUDA graphs (decode, FULL)`
- `Graph capturing finished in 13 secs, took 2.52 GiB`
- `GPU KV cache size: 4,155,578 tokens`
- `Maximum concurrency for 32,768 tokens per request: 126.82x`

### 正确性

短请求验证结果不再是连续 `!`：

```json
{"not_all_bang": true, "completion_tokens": 96}
```

### 性能

相同 prompt、MTP3、单用户非流式：

- 64 tokens：131.26 tok/s
- 256 tokens：162.07 tok/s
- 512 tokens：162.04 tok/s
- 标准脚本重启后的首个 512 token 冷请求曾测得 86.3 tok/s，日志显示 `_zero_kv_blocks_kernel`、`_compute_slot_mapping_kernel`、`eagle_*`、`_causal_conv1d_fwd_kernel` 等 Triton kernel 在推理期 JIT；连续热身后 512 token 回到 154.96 / 164.51 / 156.53 tok/s。

流式 TTFT：

- 5 次约 0.112-0.139 秒
- 128 token 流式总时延约 0.71-0.79 秒

### 当前结论

可行解不是“完整打开 torch.compile + CUDA graph”，而是 **关闭 torch.compile，打开 decode-only CUDA graph**。这避免了 inductor NaN，同时恢复了 decode 阶段 graph 性能。代价是 CUDA graph pool 额外占用约 2.52 GiB，KV cache 从 MTP3 eager 的 4,407,792 tokens 降到 4,155,578 tokens，约下降 5.7%，对单用户最快响应目标可以接受。

## 重要文件

- `config/serving.env`：共享部署配置。
- `scripts/download_models.sh`：从 ModelScope 下载模型权重。
- `scripts/start_qwen.sh`：启动 Qwen vLLM 服务。
- `scripts/start_deepseek.sh`：启动 DeepSeek vLLM 服务。
- `scripts/status.sh`：检查运行服务、模型权重完整性和 GPU 状态。
- `scripts/stop.sh`：停止当前运行的服务。
- `logs/`：运行日志和下载日志。
- `runtime/`：运行时 PID 文件。
- `models/`：本地已下载的模型权重，不要随意删除。
- `/mnt/nvme0/models`：远程 H100 服务器上的模型权重根目录。
- `docs/superpowers/plans/2026-06-30-h100-llm-deployment.md`：H100 大模型部署执行计划。
- `docs/learning-notes/`：大模型知识点学习笔记，包括推理流程、部署概念、KV Cache、TP、FP8 等内容。

## 下一步任务

1. **恢复 256K 上下文**：当前为 MTP 腾显存降到 32K，测试 256K + MTP 是否可行。
2. **继续验证 MTP3 泛化性**：当前单用户测试 MTP3 优于 MTP2，后续需在更多 prompt、低温/高温、并发场景下确认。
3. **监控 vLLM 版本更新**：关注 torch.compile NaN 修复（PR #37356 等），升级后重新测试 CUDA graph。
4. **测试 DeepSeek-V4-Flash-DSpark**：NVFP4 跑通后，切到 DeepSeek 验证。
5. **向社区反馈**：在 vLLM 开 issue 报告 Qwen3.6-27B-NVFP4 + H100 + torch.compile NaN 问题。

## 2026-07-01 远程初检记录

### 执行的命令

```bash
hostname
pwd
nvidia-smi -L
df -h /mnt/nvme0
mkdir -p /mnt/nvme0/models
find /root /home /mnt/nvme0 -maxdepth 4 -type d -name llm-deploy
rsync 本地项目到 /root/llm-deploy
cd /root/llm-deploy && bash -n scripts/*.sh
cd /root/llm-deploy && bash scripts/status.sh
```

### 观察结果

- 远程主机名为 `node1`。
- GPU 为 4 张 NVIDIA H100 80GB HBM3。
- `/mnt/nvme0` 容量约 3.5T，当前占用约 25G。
- `/mnt/nvme0/models` 已创建；初检时 Qwen 和 DeepSeek 权重还没有放进去。
- 项目文件已同步到 `/root/llm-deploy`。
- `scripts/status.sh` 显示当前没有运行中的 vLLM 服务，4 张 GPU 基本空闲。
- 远程 Python 是 3.6.8，`uv` 和 `vllm` 当前不可用。

### 初步判断

- 远程硬件满足 4 卡 H100 部署基线。
- 当前阻塞点不是 GPU，而是模型权重尚未到远程模型目录，以及 Python / vLLM 环境尚未准备。
- 部署过程中明确不允许重启机器，后续只做用户态环境安装、模型同步和服务启动验证。

### 下一步

- 优先把 Qwen3.6-27B-FP8 同步到 `/mnt/nvme0/models/Qwen3.6-27B-FP8`，先跑通较小模型。
- 准备 Python 3.12 / uv / vLLM 环境。
- 再运行 `bash scripts/status.sh` 验证权重和环境。

## 2026-07-01 Qwen 权重同步记录

### 执行的命令

```bash
rsync -az --info=progress2 models/Qwen3.6-27B-FP8/ root@node1:/mnt/nvme0/models/Qwen3.6-27B-FP8/
cd /root/llm-deploy && bash scripts/status.sh
```

### 结果

- Qwen3.6-27B-FP8 已同步到远程 `/mnt/nvme0/models/Qwen3.6-27B-FP8`。
- 远程目录大小约 29G。
- 远程目录下有 81 个顶层文件，其中 66 个 `*.safetensors`。
- `config.json` 和 `tokenizer.json` 存在。
- `scripts/status.sh` 显示 `start_qwen.sh [权重 ✓]`，DeepSeek 仍为 `[权重 ✗]`。
- 同步过程中没有重启机器，也没有启动 vLLM 服务。

### 下一步

- 准备 Python 3.12 / uv / vLLM 环境。
- 启动 Qwen 前再次运行 `bash scripts/status.sh`。
- Qwen 跑通后再考虑同步 DeepSeek 权重。

## 2026-07-01 DeepSeek 权重同步记录

### 执行的命令

```bash
rsync -a --partial --info=progress2 models/DeepSeek-V4-Flash-DSpark/ root@node1:/mnt/nvme0/models/DeepSeek-V4-Flash-DSpark/
cd /root/llm-deploy && bash scripts/status.sh
```

### 结果

- DeepSeek-V4-Flash-DSpark 已同步到远程 `/mnt/nvme0/models/DeepSeek-V4-Flash-DSpark`。
- `rsync` 传输约 166.9GB，用时约 42 分 17 秒。
- 远程目录大小约 156G。
- 远程目录下有 48 个 `*.safetensors` 分片，符合 `DS_SHARDS_EXPECTED=48`。
- `config.json` 和 `tokenizer.json` 存在。
- `scripts/status.sh` 显示 `start_deepseek.sh [权重 ✓]`。
- 同步后 `/mnt/nvme0` 使用约 210G，剩余约 3.3T。
- 同步过程中没有重启机器，也没有启动 vLLM 服务。

### 下一步

- 先准备 Python 3.12 / uv / vLLM 环境。
- 按优先级先启动 Qwen3.6-27B-FP8 做基线验证。
- Qwen 跑通后再切换到 DeepSeek，切换前执行 `bash scripts/stop.sh`。

## 2026-07-01 vLLM 0.24.0 环境安装与 Qwen 兼容性测试

### 执行的命令

```bash
cd /root/llm-deploy && bash scripts/status.sh
cd /root/llm-deploy && bash scripts/start_qwen.sh
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions
curl http://127.0.0.1:8000/v1/completions
cd /root/llm-deploy && bash scripts/stop.sh
```

### 环境结果

- 未重启机器，未升级系统 driver / CUDA / glibc。
- 远程用户态安装了 `uv`、Python 3.12.13 和 vLLM 0.24.0。
- vLLM 0.24.0 使用的 PyTorch/CUDA 用户态库比当前 NVIDIA Driver 550.144.03 更新，直接运行会出现 CUDA runtime/driver 不匹配。
- 通过用户态 `cuda-compat-12-9`、补充 `LD_LIBRARY_PATH`、降级 `llguidance` 到兼容 CentOS 8 glibc 的 wheel 后，`vllm` 可以导入并启动到较后阶段。
- `scripts/start_qwen.sh` / `scripts/start_deepseek.sh` 增加了虚拟环境和 CUDA 用户态库路径设置；配置集中放在 `config/serving.env`。

### Qwen3.6-27B-FP8 测试结果

- 模型路径：`/mnt/nvme0/models/Qwen3.6-27B-FP8`。
- 使用 TP=4，DP=1，默认 MTP 配置。
- 首次可启动配置：
  - `QWEN_LINEAR_BACKEND=triton`
  - `QWEN_ATTENTION_BACKEND=TRITON_ATTN`
  - `QWEN_MM_ENCODER_ATTN_BACKEND=TRITON_ATTN`
  - `QWEN_GDN_PREFILL_BACKEND=triton`
  - `DISABLE_CUSTOM_ALL_REDUCE=1`
  - `VLLM_DEEP_GEMM_WARMUP=skip`
- 该配置服务可以健康检查通过，启动约 280 秒；日志显示：
  - 模型加载约 7.48 GiB/GPU。
  - KV cache 约 3,121,613 tokens。
  - 262,144 tokens/request 时最大并发约 11.91x。
- 但实际推理不可用：`/v1/chat/completions` 和 `/v1/completions` 都输出连续 `!`，不是正常文本。

### 已验证的失败路径

- `linear_backend=auto` / DeepGEMM 路径：报 `cudaErrorInsufficientDriver`。
- `linear_backend=cutlass`：启动阶段报 `cutlass_gemm_caller ... Error Internal`。
- `linear_backend=flashinfer_cutlass`：回落到 DeepGEMM 权重后处理，报 `cudaErrorInsufficientDriver`。
- `linear_backend=emulation`：该层类型没有 `emulation` kernel，报 `no 'emulation' kernel exists for this layer type`。
- `linear_backend=triton` + 关闭 MTP：仍然输出连续 `!`，说明不是 MTP 本身导致。
- `linear_backend=triton` + `--enforce-eager`：profile 阶段仍报 `cudaGetDeviceCount failed: CUDA driver version is insufficient for CUDA runtime version`。

### 当前判断

- 在“不重启、不升级驱动/CUDA/glibc”的约束下，latest vLLM 0.24.0 尚未找到能正确部署 `Qwen3.6-27B-FP8` 的 FP8 kernel 路径。
- 当前阻塞不是模型权重缺失，也不是端口/API 问题，而是 vLLM 0.24.0 的 CUDA 用户态栈和 NVIDIA Driver 550.144.03 的兼容性问题。
- 下一步可选方向：
  - 若保持不升级驱动：尝试更旧的 vLLM/PyTorch CUDA 12.1/12.4 组合，但可能不支持 Qwen3.6 架构和 MTP。
  - 若要使用 latest vLLM 0.24.0 并追求 H100 FP8 性能：需要在维护窗口升级 NVIDIA driver 到支持当前 CUDA runtime 的版本，然后重新测试 DeepGEMM/Cutlass/FlashAttention 路径。
  - 继续测试 DeepSeek 前，应先解决同类 CUDA runtime/driver 兼容问题，否则 DeepSeek 的 FP8/Flash/DeepGEMM 路径大概率也会遇到类似问题。

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
- 张量并行：为什么 4 卡环境下优先使用 `TP=4, DP=1`，以及 GPU 数量变化时如何调整。
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

## 2026-07-02 官方 vLLM Docker 镜像实验入口

### 背景

用户希望先按官方 Docker 配置做 CUDA 12.9 方向的实验，不自建镜像。当前选择是使用 vLLM 官方 OpenAI-compatible 镜像：

```bash
vllm/vllm-openai:latest
```

### 修改内容

- 在 `config/serving.env` 增加 Docker 实验配置：
  - `VLLM_DOCKER_IMAGE="vllm/vllm-openai:latest"`
  - `VLLM_DOCKER_ENABLE_CUDA_COMPATIBILITY=1`
  - `QWEN_DOCKER_CONTAINER_NAME="qwen-vllm"`
  - `DS_DOCKER_CONTAINER_NAME="deepseek-vllm"`
- 新增 `scripts/start_qwen_docker.sh`：
  - 使用官方 vLLM Docker 镜像启动 Qwen3.6-27B-FP8。
  - 通过 `-v /mnt/nvme0/models:/mnt/nvme0/models:ro` 挂载模型目录。
  - 不把模型权重打进镜像。
- 新增 `scripts/start_deepseek_docker.sh`：
  - 使用同一个官方 vLLM Docker 镜像启动 DeepSeek-V4-Flash-DSpark。
  - 同样只读挂载 `/mnt/nvme0/models`。
- 更新 `scripts/status.sh`：
  - 如果服务由 Docker 启动，显示容器状态和 `docker logs -f` 查看方式。
- 更新 `scripts/stop.sh`：
  - 如果存在容器记录，优先使用 `docker stop` 和 `docker rm` 停止并清理容器。

### 学习记录

- Docker 镜像里放的是 Python、vLLM、PyTorch、CUDA runtime 等运行环境。
- 模型权重继续放在宿主机 `/mnt/nvme0/models`，启动容器时只读挂载进去。
- Docker 不能替代宿主机 NVIDIA driver。当前 H100 服务器 driver 为 550.144.03，使用官方 latest 镜像做 CUDA 12.9 方向实验时，仍可能遇到 `cudaErrorInsufficientDriver`。
- `VLLM_ENABLE_CUDA_COMPATIBILITY=1` 是 vLLM 官方文档推荐的旧驱动兼容开关，但不能保证解决所有 FP8/JIT kernel 兼容问题。

### 验证

本地执行：

```bash
bash -n scripts/*.sh
bash scripts/status.sh
```

结果：

- shell 语法检查通过。
- 本地 `status.sh` 可正常执行。
- 本地没有 `nvidia-smi`，也没有 `/mnt/nvme0/models` 下的远程模型目录，因此本地状态不代表 H100 服务器状态。

### 下一步

在远程 H100 服务器上执行：

```bash
cd /root/llm-deploy
bash scripts/status.sh
docker run --rm --runtime nvidia --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
bash scripts/start_qwen_docker.sh
curl http://127.0.0.1:8000/v1/models
```

如果 Qwen 仍出现连续 `!` 或 `cudaErrorInsufficientDriver`，说明问题不是 Python 虚拟环境污染，而更可能是官方 latest 镜像的 CUDA/PyTorch/vLLM kernel 路径与宿主机 driver 不匹配。

### 远程同步与镜像拉取结果

2026-07-02 已将官方 Docker 启动入口同步到远程 `/root/llm-deploy`：

```bash
rsync -az --relative \
  config/serving.env \
  scripts/start_qwen_docker.sh \
  scripts/start_deepseek_docker.sh \
  scripts/status.sh \
  scripts/stop.sh \
  PROJECT_LOG.md \
  root@10.16.11.24:/root/llm-deploy/
```

远程验证命令：

```bash
cd /root/llm-deploy
bash -n scripts/*.sh
bash scripts/status.sh
docker --version
docker images vllm/vllm-openai:latest
```

观察结果：

- 远程主机为 `node1`。
- `bash -n scripts/*.sh` 通过。
- `scripts/status.sh` 显示 Qwen 和 DeepSeek 权重均可用，4 张 H100 空闲。
- `status.sh` 已显示 Docker 启动入口：
  - `bash scripts/start_qwen_docker.sh`
  - `bash scripts/start_deepseek_docker.sh`
- Docker 版本为 24.0.9。
- Docker 数据目录为 `/home/docker`，所在 `/home` 分区剩余约 93G。
- 当前远程没有 `vllm/vllm-openai:latest` 镜像。

尝试拉取官方镜像：

```bash
docker pull vllm/vllm-openai:latest
```

失败信息：

```text
Error response from daemon: Get "https://registry-1.docker.io/v2/": context deadline exceeded (Client.Timeout exceeded while awaiting headers)
```

进一步验证：

```bash
curl -I --connect-timeout 8 https://registry-1.docker.io/v2/
docker info --format '{{json .RegistryConfig.Mirrors}}'
```

结果：

- 远程访问 Docker Hub registry 超时。
- Docker registry mirror 当前为 `null`，没有配置镜像源。

当前判断：

- 官方 vLLM Docker 启动脚本和配置已经在远程。
- 官方 vLLM Docker 镜像还没有拉取成功。
- 阻塞点是远程服务器访问 Docker Hub 超时；不是模型路径、脚本语法或 Docker 磁盘空间问题。
- 下一步需要提供可访问的 Docker registry mirror，或由机房/平台侧预置 `vllm/vllm-openai:latest` 镜像后再启动。

### 本地下载再上传方案检查

用户提出“本地下载，再传上去”的方案。检查命令：

```bash
docker --version
df -h / /home /tmp
command -v podman
command -v skopeo
command -v nerdctl
command -v ctr
```

观察结果：

- 当前本地机器没有 `docker` 命令。
- 当前本地机器没有 `podman`、`skopeo`、`nerdctl`、`ctr`。
- 本地磁盘空间充足，根分区约 644G 可用。

判断：

- “本地下载 Docker 镜像再传到远程”技术上可行。
- 但当前本地环境缺少拉取和导出 OCI/Docker 镜像的工具，暂时不能直接执行 `docker pull` / `docker save`。
- 如果换到一台能访问 Docker Hub 且安装了 Docker 的机器，可以用 `docker save | ssh docker load` 方式传到远程，不需要先落地成超大 tar 文件。

### 本地用户态工具拉取失败记录

为避免等待本机 Docker 安装，尝试使用用户态 `crane` 拉取并导出官方镜像：

```bash
/tmp/crane-bin/crane pull \
  --platform=linux/amd64 \
  --cache_path=/tmp/crane-cache-vllm-fresh \
  --format=legacy \
  vllm/vllm-openai:latest \
  /tmp/vllm-openai-latest.fresh.tar
```

结果：

- 本地没有 Docker，且 `sudo -n true` 显示需要用户输入 sudo 密码，当前会话不能代装系统 Docker。
- 本地 rootless Docker 依赖组件不完整，未找到 `newuidmap`、`newgidmap`、`rootlesskit`、`slirp4netns` 等命令。
- `crane` 使用全新缓存重新拉取时仍失败：

```text
Error: saving legacy tarball /tmp/vllm-openai-latest.fresh.tar: unexpected EOF
```

判断：

- 失败点仍是大 layer 下载或导出过程中连接被截断，不是远程 Docker runtime 问题。
- 当前最稳妥路径是用户在本机输入 sudo 密码安装 Docker Engine，然后使用 `docker pull` / `docker save` 生成官方镜像 tar，再传到远程 `docker load`。

### 官方 vLLM 镜像本地中转并加载到远程

2026-07-02 继续执行本地 Docker 中转方案。

本机安装 Docker Engine：

```bash
sudo bash /home/yangyuxin/llm-deploy/scripts/install_local_docker.sh
```

观察结果：

- 本机 Docker Engine 已安装，版本为 `Docker version 29.6.1`。
- 安装过程中 Docker Hub 直连 `hello-world` 仍出现 `connection reset by peer`，说明本机到 Docker Hub 也不稳定。
- 当前 shell 尚未重新登录，普通 `docker` 命令暂时没有 docker 组权限；使用 `sg docker -c 'docker ...'` 临时进入 docker 组执行。

官方镜像直连拉取失败：

```bash
sg docker -c 'docker pull vllm/vllm-openai:latest'
```

错误信息：

```text
failed to do request: Head "https://registry-1.docker.io/v2/vllm/vllm-openai/manifests/latest": read: connection reset by peer
```

改用临时 Docker Hub 镜像地址拉取同名镜像：

```bash
sg docker -c 'docker pull docker.m.daocloud.io/vllm/vllm-openai:latest'
sg docker -c 'docker tag docker.m.daocloud.io/vllm/vllm-openai:latest vllm/vllm-openai:latest'
```

结果：

- 本机成功拉取镜像，digest 为 `sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f`。
- 本机 `docker images` 显示 `vllm/vllm-openai:latest`，大小约 29.9GB。
- 之前 `crane` 反复损坏的 `761033aa3ab9...` 大 layer 在 Docker pull 中成功下载并 `Pull complete`。

导出并上传：

```bash
sg docker -c 'docker save vllm/vllm-openai:latest -o /tmp/vllm-openai-latest.docker-save.tar'
rsync -P /tmp/vllm-openai-latest.docker-save.tar root@10.16.11.24:/mnt/nvme0/vllm-openai-latest.docker-save.tar
```

结果：

- 本地导出的 Docker tar 大小约 8.6GB。
- 本地和远程 sha256 一致：

```text
e7d1e8f5af940b49411d10e22bb932267890412507c70aca5bee75f084db1116
```

远程加载：

```bash
docker load -i /mnt/nvme0/vllm-openai-latest.docker-save.tar
docker images --format "{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}" | grep "^vllm/vllm-openai:latest"
cd /root/llm-deploy && bash scripts/status.sh
```

结果：

- 远程 `docker load` 成功：

```text
Loaded image: vllm/vllm-openai:latest
```

- 远程已有 `vllm/vllm-openai:latest`，镜像大小约 20.4GB。
- 远程当前没有运行中的服务。
- 4 张 H100 均空闲，显存约 5MiB/81559MiB。
- Qwen3.6-27B-FP8 和 DeepSeek-V4-Flash-DSpark 权重均可用。
- 下一步可以在远程执行 `bash scripts/start_qwen_docker.sh` 做官方 Docker 启动验证。

## 2026-07-02 Qwen3.6-27B-NVFP4 权重下载与上传记录

用户补充关注 NVIDIA 发布的 `nvidia/Qwen3.6-27B-NVFP4`。该模型是基于 Qwen3.6-27B 的 NVIDIA ModelOpt NVFP4 量化版本，不是蒸馏模型；推理时通常需要 vLLM 的 `--quantization modelopt` 路径。

本地下载命令使用 Hugging Face CLI。由于当前网络代理链路对 Xet/HTTP 大文件下载不稳定，实际采用禁用 Xet 的普通 HTTP 断点续传方式，逐个下载 3 个 safetensors 分片：

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 \
  hf download nvidia/Qwen3.6-27B-NVFP4 model-0000X-of-00003.safetensors \
  --local-dir models/Qwen3.6-27B-NVFP4
```

观察到的问题：

- Xet 后端在大文件重建时出现长时间停滞，临时文件不稳定落盘。
- 普通 HTTP 后端多次出现 `read operation timed out`、`peer closed connection without sending complete message body` 和 `SSL: UNEXPECTED_EOF_WHILE_READING`。
- 通过循环重试和断点续传最终完成下载。

本地完成后，目录为：

```bash
models/Qwen3.6-27B-NVFP4
```

关键文件大小：

```text
model-00001-of-00003.safetensors  9965652512
model-00002-of-00003.safetensors  9985757032
model-00003-of-00003.safetensors  1970287640
```

上传到远程 H100 服务器：

```bash
rsync -a --partial --info=progress2 \
  models/Qwen3.6-27B-NVFP4/ \
  root@10.16.11.24:/mnt/nvme0/models/Qwen3.6-27B-NVFP4/
```

结果：

- 远程目录 `/mnt/nvme0/models/Qwen3.6-27B-NVFP4` 已存在并完成同步。
- 远程目录大小约 21G。
- 远程 3 个 safetensors 分片大小与本地一致。
- 上传前后远程 `scripts/status.sh` 显示没有运行中的服务，4 张 H100 空闲。
- 上传过程没有重启机器，没有删除或移动已有模型权重。

下一步：

- 将 NVFP4 模型加入配置和状态检查，避免与原 FP8 Qwen 配置混淆。
- 新增或调整启动脚本时，需要显式使用 `--quantization modelopt`。
- 由于之前 vLLM 0.24.0 + 当前驱动在 FP8 路径上存在兼容问题，NVFP4 需要单独启动验证，不能直接假设可用。

## 2026-07-02 Qwen3.6-27B-NVFP4 官方 Docker vLLM 测试

### 测试目标

使用已经加载到远程的官方 `vllm/vllm-openai:latest` Docker 镜像测试 `Qwen3.6-27B-NVFP4`，验证 ModelOpt NVFP4 路径是否能启动并正常生成文本。

### 执行的命令

远程启动前先确认没有运行中的服务：

```bash
cd /root/llm-deploy && bash scripts/status.sh
```

使用一次性 Docker 命令启动 NVFP4，不修改现有 FP8 / DeepSeek 启动脚本：

```bash
docker run -d \
  --name qwen-nvfp4-vllm \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v /mnt/nvme0/models:/mnt/nvme0/models:ro \
  --env VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
  --env VLLM_DEEP_GEMM_WARMUP=skip \
  vllm/vllm-openai:latest \
  --model /mnt/nvme0/models/Qwen3.6-27B-NVFP4 \
  --served-model-name qwen3.6-27b-nvfp4 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --quantization modelopt \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --disable-custom-all-reduce
```

### 启动结果

- 容器名：`qwen-nvfp4-vllm`。
- 服务名：`qwen3.6-27b-nvfp4`。
- 官方镜像内 vLLM 版本：`0.24.0`。
- 官方镜像内 PyTorch / CUDA：`torch 2.11.0+cu130`，CUDA `13.0`。
- `/v1/models` 健康检查通过，返回 `max_model_len=262144`。
- `scripts/status.sh` 显示服务就绪，4 张 H100 显存约 `74411-74421 MiB / 81559 MiB`。
- 日志显示 vLLM 正确识别 ModelOpt：
  - `Detected ModelOpt fp8 checkpoint`
  - `Detected ModelOpt NVFP4 checkpoint`
  - engine 配置中 `quantization=modelopt_mixed`
- 权重加载成功：
  - checkpoint size 约 `20.42 GiB`
  - loading weights 约 `1.75s`
  - model loading 约 `5.15 GiB`/GPU，约 `3.01s`
- KV cache：
  - `GPU KV cache size: 7,542,388 tokens`
  - `Maximum concurrency for 262,144 tokens per request: 28.77x`
- 启动 profile / warmup：
  - `init engine ... took 110.64s`
  - 其中 `compilation: 80.16s`

### 推理验证结果

健康检查通过后，测试 `/v1/chat/completions`：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"用一句中文解释什么是KV Cache。"}],"max_tokens":80,"temperature":0.2}'
```

结果：HTTP 200，但输出异常，`reasoning` 字段为连续 `!`，`content=null`。

继续测试 `/v1/completions`：

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-nvfp4","prompt":"用一句中文解释什么是KV Cache。\n答案：","max_tokens":80,"temperature":0.2}'
```

结果：HTTP 200，但 `text` 仍为连续 `!`。

### 当前判断

- 官方 Docker vLLM 0.24.0 可以成功加载 `Qwen3.6-27B-NVFP4`，并且 `--quantization modelopt` 路径能识别 ModelOpt mixed FP8/NVFP4 配置。
- 这次没有复现 `cudaErrorInsufficientDriver`，说明容器 CUDA 13.0 路径至少可以完成启动、权重加载、编译和 API 服务。
- 推理结果仍然异常，表现为连续 `!`，与之前 `Qwen3.6-27B-FP8` 的异常输出相似。
- 因为 `/v1/completions` 也异常，问题不只是 `reasoning-parser qwen3` 的 chat 解析问题，更可能在 Qwen3.6 + 当前 vLLM/kernel/量化 decode 路径。
- 远程当前有一个正在运行的测试容器，占用 4 张 H100。停止命令：

```bash
cd /root/llm-deploy && bash scripts/stop.sh
```

或：

```bash
docker rm -f qwen-nvfp4-vllm
```


## 2026-07-07 双模型并行部署 + opencode 接入

### 目标

从 NVFP4 单模型互斥部署,切换为 Qwen3.6-27B-FP8 + Agents-A1-FP8 双模型并行部署,各占 2 张 H100(TP=2),不同端口。并接入 opencode 编码工具。

### 关键变更

1. **解除互斥部署模式**:两个模型同时运行,用 `CUDA_VISIBLE_DEVICES` 隔离 GPU
   - Qwen3.6-27B-FP8:GPU 0,1,端口 8000
   - Agents-A1-FP8:GPU 2,3,端口 8001

2. **新增 Agents-A1-FP8 模型**(InternScience/Agents-A1-FP8)
   - 架构:Qwen3_5MoeForConditionalGeneration(Qwen3.5 MoE)
   - FP8 量化(compressed-tensors),36GB 权重
   - 256 专家/8 激活,40 层,原生 256K 上下文
   - 从 ModelScope 下载,rsync 到远程 `/mnt/nvme0/models/Agents-A1-FP8/`

3. **One API 网关**:http://10.30.75.58:18082/
   - 两个渠道:qwen3.6-27b-fp8 (id=4)、Agents-A1-FP8 (id=5)
   - 统一入口,便于计费和管理

4. **opencode 接入**:~/.config/opencode/opencode.json
   - 两个 provider:qwen (8000)、agents (8001)
   - 主模型 qwen3.6-27b-fp8,小模型 agents-a1-fp8

### 官方配置对齐(2026-07-07)

对照两个模型的官方 HuggingFace 文档,调整了启动参数:

#### Qwen3.6-27B-FP8 配置(https://huggingface.co/Qwen/Qwen3.6-27B-FP8)

| 参数 | 值 | 说明 |
|---|---|---|
| max-model-len | 262144 (256K) | 官方原生支持 |
| tensor-parallel-size | 2 | 双模型并行,各 2 卡 |
| kv-cache-dtype | fp8 | 显存优化 |
| reasoning-parser | qwen3 | 思考链解析 |
| tool-call-parser | qwen3_coder | 官方推荐 |
| language-model-only | ✅ | 跳过 vision encoder,省显存 |
| speculative-config | qwen3_next_mtp, 2 tokens | MTP 投机解码 |
| 采样参数 | temp=0.6, top_p=0.95, top_k=20 | 精确编码模式 |
| max_tokens | 81920 | 官方建议复杂数学/编程场景 |

#### Agents-A1-FP8 配置(https://huggingface.co/InternScience/Agents-A1-FP8)

| 参数 | 值 | 说明 |
|---|---|---|
| max-model-len | 262144 (256K) | 官方推荐(原设 128K 已改) |
| tensor-parallel-size | 2 | 双模型并行 |
| kv-cache-dtype | fp8 | 官方推荐(原缺失已加) |
| reasoning-parser | qwen3 | 官方推荐 |
| tool-call-parser | qwen3_coder | 官方推荐(原 hermes/qwen3_xml 不兼容已改) |
| speculative-config | qwen3_next_mtp, 2 tokens | MTP(模型 config 有 mtp_num_hidden_layers=1) |
| 采样参数 | temp=0.85, top_p=0.95, top_k=20, presence_penalty=1.1 | 官方推荐 |
| max_tokens | 81920 | 与 Qwen 对齐 |

### Function Calling 支持

- 两个模型都加了 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
- Function Calling 测试通过(模型能返回结构化 tool_calls)
- opencode 工具调用正常(读写文件等)

### 踩坑记录

1. **One API token**:登录密码不是 API token,需要去网页「令牌」页面复制 sk-xxx 格式的 token
2. **Docker GPU 隔离**:`--gpus all` + `CUDA_VISIBLE_DEVICES` 环境变量限制具体 GPU
3. **Agents tool-call-parser**:
   - `hermes`:模型输出工具调用文本但不解析 ❌
   - `qwen3_xml`:能工作但非官方推荐 ⚠️
   - `qwen3_coder`:官方推荐,正常工作 ✅
4. **Qwen3.6 不支持 `/no_think`**:这是 Qwen3 的语法,Qwen3.6 关闭思考要用 `chat_template_kwargs: {"enable_thinking": false}`
5. **Qwen3.6 思考模型 max_tokens**:思考过程消耗大量 token,300 token 不够输出最终答案,需 ≥1000,复杂任务建议 81920
6. **opencode context 配置**:客户端 `limit.context` 要与服务端 `--max-model-len` 对齐,否则状态栏显示错误

### 当前运行状态(2026-07-07)

```
GPU 0,1 (71.6GB/81.5GB) → qwen3.6-27b-fp8 (端口 8000)
GPU 2,3 (75.3GB/81.5GB) → agents-a1-fp8-vllm (端口 8001)
```

### 文件变更

| 文件 | 变更 |
|---|---|
| config/serving.env | 新增 QWEN_FP8_* 和 AGENTS_* 配置块 |
| scripts/start_qwen_fp8_docker.sh | 新建,Qwen FP8 Docker 启动脚本 |
| scripts/start_agents_docker.sh | 新建,Agents Docker 启动脚本 |
| scripts/stop.sh | 新增 agents/qwen-fp8 case 匹配 |
| scripts/status.sh | 新增权重检查和启动提示 |
| ~/.config/opencode/opencode.json | 新建,opencode 客户端配置 |

### 验证结果

- ✅ 两个容器正常运行
- ✅ 两个端点 /v1/models 返回正确(都是 256K)
- ✅ 对话、流式输出、Function Calling 全部通过
- ✅ opencode 能调用两个模型(包括文件操作等工具调用)
- ✅ MTP 投机解码两个模型都启用

### 下一步

- 接入更多 OpenAI 兼容工具(Aider、Cline 等)
- 吞吐测试对比(MTP 加速效果)
- 长上下文压力测试(256K)
- 稳定性测试(长时间并发)

