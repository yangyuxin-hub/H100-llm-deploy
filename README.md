# H100 大模型部署项目

在 node1（4×H100 80GB）上使用 vLLM 部署和评测大模型。当前远端记录：

| 用途 | 模型 | GPU | 端口 |
|---|---|---:|---:|
| 稳定服务 | Qwen3.6-27B-FP8 | 0,1 | 8000 |
| 实验服务 | ThinkingCap-Qwen3.6-27B-FP8 | 2,3 | 8001 |

## 目录结构

```text
llm-deploy/
├── config/       # 当前服务配置
├── scripts/      # 下载、启动、停止、状态检查
├── benchmarks/   # 评测代码、协议和本地数据缓存
├── docs/         # 学习笔记、项目日志、报告和调研
├── logs/         # 原始 benchmark/eval 输出，不提交 Git
├── models/       # 本地模型权重，不提交 Git
└── runtime/      # PID、容器信息等临时状态
```

## 主要入口

- 当前进度：[PROJECT_LOG.md](PROJECT_LOG.md)
- 服务配置：[config/serving.env](config/serving.env)
- 部署脚本：[`scripts/status.sh`](scripts/status.sh) 及同目录脚本
- 评测资产：[benchmarks/README.md](benchmarks/README.md)
- 学习笔记：[docs/learning-notes/INDEX.md](docs/learning-notes/INDEX.md)
- 历史过程：[docs/project-log/README.md](docs/project-log/README.md)

```bash
bash scripts/status.sh
bash -n scripts/*.sh benchmarks/evalscope/*.sh benchmarks/modelcard/*.sh
```

本地状态不代表 H100 远端状态。禁止重启服务器，禁止随意改动 `models/` 权重。
