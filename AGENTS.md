# Agent 项目说明

这个仓库是用户的实习项目：在 H100 GPU 上部署大模型，并学习大模型推理的底层知识。

## 项目目的

- 帮助用户使用 vLLM 部署 Qwen3.6-27B-FP8 和 DeepSeek-V4-Flash-DSpark。
- 保留用户的学习过程，而不是只记录最终可用命令。
- 将 `PROJECT_LOG.md` 视为主要项目记忆，用于记录目标、进展、下一步任务和学习笔记。
- 默认使用中文沟通和写文档；命令、路径、模型名和必要技术术语可以保留英文。

## 环境背景

- 真正的 H100 机器需要通过 SSH 远程连接。
- 当前本地机器不一定有 GPU。如果本地 `nvidia-smi` 不可用，不要直接判断 H100 环境有问题。
- 部署脚本面向 8 卡 H100 环境，张量并行大小为 8。
- 两个模型设计为互斥运行，并复用同一个服务端口。

## 操作前先做

1. 阅读 `PROJECT_LOG.md`。
2. 阅读 `config/serving.env`。
3. 检查部署状态时，优先运行或查看 `scripts/status.sh`。
4. 修改启动脚本前，先查看 `logs/` 下的相关日志。

## 安全规则

- 不要删除、移动或重写 `models/` 下的模型权重，除非用户明确要求。
- 不要粘贴或保存凭据、SSH key、token 或私有服务器信息。
- 不要假设命令已经在 H100 服务器上执行，除非用户明确说明，或者命令输出能够证明。
- 区分本地机器观察结果和 H100 服务器观察结果。
- 除非用户明确改变目标，否则保持当前互斥部署模式。

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
- 在 Qwen 和 DeepSeek 服务之间切换前，先使用 `scripts/stop.sh`。

## 常用命令

```bash
bash scripts/status.sh
bash scripts/start_qwen.sh
bash scripts/start_deepseek.sh
bash scripts/stop.sh
tail -f logs/qwen3.6-27b-fp8.log
tail -f logs/deepseek-v4-flash-dspark.log
curl http://127.0.0.1:8000/v1/models
```
