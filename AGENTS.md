# Agent 项目说明

这个仓库是用户的实习项目：在 H100 GPU 上部署大模型，并学习大模型推理的底层知识。

## 项目目的

- 帮助用户使用 vLLM 部署 Qwen3.6-27B-FP8 和 Agents-A1-FP8(双模型并行)。
- 保留用户的学习过程,而不是只记录最终可用命令。
- 将 `PROJECT_LOG.md` 视为主要项目记忆,用于记录目标、进展、下一步任务和学习笔记。
- 默认使用中文沟通和写文档;命令、路径、模型名和必要技术术语可以保留英文。

## H100 服务器连接

- 主机：10.16.11.24（node1）
- 登录：`ssh root@10.16.11.24`（本地 `~/.ssh/config` 已配置 IdentityFile）
- 密钥：`~/.ssh/codex_llm_deploy`（私钥），公钥已在服务器 `~/.ssh/authorized_keys`
- GPU：4× H100 80GB HBM3（不是 8 卡，张量并行大小为 4）
- Driver：550.144.03，CUDA 12.9 原生兼容（cu129 镜像，无需 cuda-compat 层）
- vLLM 镜像：`vllm/vllm-openai:v0.24.0-cu129-ubuntu2404`（v0.24.0, CUDA 12.9）

## 严格安全规则（不可违反）

- **绝对不允许重启 H100 服务器**。只允许 docker 容器级别的 stop/start/rm。
- 不要删除、移动或重写 `models/` 下的模型权重，除非用户明确要求。
- 不要粘贴或保存凭据、SSH key、token 或私有服务器信息。
- 不要假设命令已经在 H100 服务器上执行，除非用户明确说明，或者命令输出能够证明。
- 区分本地机器观察结果和 H100 服务器观察结果。
- 除非用户明确改变目标,否则保持当前双模型并行部署模式(Qwen3.6-27B-FP8 + Agents-A1-FP8,各占 2 卡)。

## 操作前先做

1. 阅读 `PROJECT_LOG.md`(特别是 2026-07-07 双模型并行部署章节)。
2. 阅读 `config/serving.env`(QWEN_FP8_* 和 AGENTS_* 配置块)。
3. 检查部署状态时,优先运行或查看 `scripts/status.sh`。
4. 修改启动脚本前,先查看 `logs/` 下的相关日志。
5. 了解两个模型的官方配置:
   - Qwen3.6-27B-FP8: https://huggingface.co/Qwen/Qwen3.6-27B-FP8
   - Agents-A1-FP8: https://huggingface.co/InternScience/Agents-A1-FP8

## 安全规则

- 不要删除、移动或重写 `models/` 下的模型权重，除非用户明确要求。
- 不要粘贴或保存凭据、SSH key、token 或私有服务器信息。
- 不要假设命令已经在 H100 服务器上执行，除非用户明确说明，或者命令输出能够证明。
- 区分本地机器观察结果和 H100 服务器观察结果。
- 除非用户明确改变目标,否则保持当前双模型并行部署模式(Qwen3.6-27B-FP8 + Agents-A1-FP8,各占 2 卡)。

## 文档规则

- 项目状态变化时，更新 `PROJECT_LOG.md`。
- 记录学习价值，不只记录操作步骤。
- 调试问题时记录：
  - 执行的命令
  - 错误信息
  - 初步怀疑原因
  - 尝试的修复方法
  - 最终结果
- 部署状态尽量使用准确日期和具体数值。
- 默认用中文写说明、总结和学习笔记。

## 工程规则

- 遵循现有 Bash 脚本风格。
- 配置优先放在 `config/serving.env`，除非某个值只属于单个脚本。
- 修改 shell 脚本后，运行：

  ```bash
  bash -n scripts/*.sh
  ```

- 使用 `scripts/status.sh` 作为第一步状态检查。
- 重启任一模型前,先使用 `scripts/stop.sh <qwen-fp8|agents>` 停止对应服务。

## 常用命令

```bash
# SSH 连接 H100
ssh root@10.16.11.24

# 远程容器操作(双模型并行)
ssh root@10.16.11.24 'docker ps --filter name=qwen3.6-27b-fp8'
ssh root@10.16.11.24 'docker ps --filter name=agents-a1-fp8-vllm'
ssh root@10.16.11.24 'docker logs --tail 30 qwen3.6-27b-fp8'
ssh root@10.16.11.24 'docker logs --tail 30 agents-a1-fp8-vllm'

# 本地脚本
bash scripts/status.sh
bash scripts/start_qwen_fp8_docker.sh
bash scripts/start_agents_docker.sh
bash scripts/stop.sh qwen-fp8
bash scripts/stop.sh agents
tail -f logs/qwen3.6-27b-fp8.docker.log
tail -f logs/agents-a1-fp8.docker.log

# 端点验证
curl http://10.16.11.24:8000/v1/models   # Qwen3.6-27B-FP8
curl http://10.16.11.24:8001/v1/models   # Agents-A1-FP8

# One API 网关(http://10.30.75.58:18082/)
#   admin / d454e12b57472124
#   渠道 id=4: qwen3.6-27b-fp8  → http://10.16.11.24:8000
#   渠道 id=5: agents-a1-fp8    → http://10.16.11.24:8001

# opencode(本地安装,接入两个模型)
opencode
opencode models   # 列出可用模型
opencode run --model agents/agents-a1-fp8 "你的问题"
```
