# 07 常见问题与安全边界

本页讲"出问题怎么办"和"不能做什么"。跑评测的流程见 [05](05-execution-guide.md)。

## 常见问题

| 现象 | 优先判断 | 处理 |
|---|---|---|
| `/v1/models` 模型名不符 | 8001 当前运行了其他模型 | 停止本轮,不要误发请求 |
| 两模型进度同时停止 | node1 上的 runner/终端退出 | 检查 PID、runner.log,再使用同 run id `--resume` |
| 远端容器正常但 `Running=0` | 当前没有评测请求 | 检查 node1 上的评测进程和 progress 更新时间 |
| 冻结配置不一致 | 同 run id 参数变化 | 新建 run id,不强制复用 cache |
| IFBench 生成数与计分数不同 | adapter 有样本未进入指标 | 同时报告两个分母并列出差异 |
| LongBench 上下文超限 | 输入加输出超过 262,144 | 重新预处理,不静默裁剪 |
| judge 任务被阻塞 | 未配置独立 judge | smoke 可自评;full 必须独立 judge |
| 代码任务被阻塞 | 未启用或无法访问 Docker | 修复隔离 sandbox 后显式启用 |
| `BLOCKED_EXTERNAL` | 缺搜索、视频、GUI 等 worker | 单独建设环境,不用普通 adapter 替代 |
| TAU2 报空 `AssistantMessage` | 上一轮工具调用落在普通 `content` | 查看原始轨迹;ThinkingCap 检查服务端 `thinkingcap_agent.jinja` 是否加载并先重跑 smoke,不在评测端改写正文 |
| `Failed to decode the model response` 大量出现(BFCL) | 上游判定当前响应非函数调用 | Agentic 任务最后一条自然语言回答本就是评分对象,不能用该日志行数代替原始响应审计 |
| 请求被代理拦截(连接 127.0.0.1:7890) | NO_PROXY 未设 | `export NO_PROXY=...,127.0.0.1,localhost,10.16.11.24` |
| HMMT 数学输出被截断 | 旧参数 16,384 不够 | 新参数已提到 81,920;检查是否用了旧 tsv |
| BFCL 卡在 `memory snapshot prereq` 数十分钟 | `SentenceTransformer` 在无网环境下尝试下载 `all-MiniLM-L6-v2` 并指数退避重试 | 本地下载模型后上传到 `/mnt/nvme0/models/all-MiniLM-L6-v2/`,patch `bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_vector.py:26` 改用本地绝对路径 |
| P1 full 模式数据集卡住或报 `Unknown split` | `run_reference_suite.sh` 在 full 模式下未传 `local_path`,evalscope 联网拉取 HF/ModelScope 失败 | 确认 `${P1_FULL_DATA_ROOT}` 已预下载数据;runner 已修复为 smoke/full 都传 `local_path` |
| `ValueError: Value.__init__() missing 'dtype'` | evalscope 加载 `dataset_infos.json` 与 parquet 不兼容 | 删除 `dataset_infos.json`,evalscope loader 也会自动忽略 |

## 调试记录要求

调试问题时记录:

- 执行的命令
- 错误信息
- 初步怀疑原因
- 尝试的修复方法
- 最终结果
- 可比性影响
- 证据路径

## 安全边界

以下是不可违反的红线:

- **绝对不重启 H100 宿主机**;只允许按部署流程操作容器。
- 评测脚本本身不得启动、停止或重启模型服务。
- 不删除、移动或重写 `models/` 下的模型权重。
- 不保存 SSH key、token 或外部 judge 凭据。
- 代码、搜索、GUI、Android 和 Agent worker 与模型宿主机隔离。
- `BLOCKED` 表示环境或协议缺口,不表示模型能力失败;不能用近似数据或自评 judge 冒充正式结果。
- 评测脚本只请求已有 OpenAI-compatible API,不修改模型权重或服务配置。
- 修改 `thinkingcap_agent.jinja` 或重启 ThinkingCap 容器后,必须先重跑 TAU2 smoke,不能直接沿用修复前的状态。
