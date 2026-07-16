# External benchmark 运行环境边界

本文只描述 `reference_benchmarks.tsv` 中 `integration=external` 的 46 项评测所需环境，不表示这些依赖、数据或 harness 已经安装，也不表示评测已经可以正式运行。

这些任务不能直接加入当前通用 EvalScope runner 后就视为“已支持”。正式接入前至少要固定：官方或可审计的 harness revision、数据 revision、输入预处理、答案抽取、judge、资源限制和结果目录。缺少任何关键项时，runner 应返回 `BLOCKED` 或 `SKIPPED`，不能用空结果或近似任务冒充成功。

## 建议的公共目录与 worker 边界

所有路径都应由环境变量提供，不在仓库中写入凭据或机器私有路径。建议把依赖、只读数据、临时工作区和结果分开：

```text
${EXTERNAL_ENV_ROOT}/<suite>/      # harness 源码、锁文件和独立虚拟环境
${EXTERNAL_DATA_ROOT}/<suite>/     # 固定 revision 的只读数据与校验和
${EXTERNAL_WORK_ROOT}/<run-id>/    # 可删除的单次运行工作区
logs/eval/external/<run-id>/       # manifest、轨迹、stdout/stderr 和评分结果
```

建议使用五类独立 worker，而不是把全部权限集中到 H100 模型服务节点：

| Worker | 允许能力 | 默认禁止 |
|---|---|---|
| `python-api` | 读取固定数据、调用被测模型/独立 judge API | 任意宿主写入、未声明的公网访问 |
| `search-agent` | 受控公网搜索、网页抓取、调用模型和 judge | 访问内网、云元数据地址和本地凭据文件 |
| `code-sandbox` | 在一次性容器内编译、测试、启动题目服务 | privileged、宿主 Docker socket 透传、无限网络和持久写入 |
| `gui-android` | 操作可恢复的 VM/模拟器快照 | 操作宿主桌面、访问模型服务节点上的真实账号或文件 |
| `media-data` | 读取视频、3D、文档和医疗 benchmark 数据，执行解码/评分 | 修改原始数据、处理未授权数据、把受限数据写入普通日志 |

数据下载与正式评测应分成两个阶段。下载阶段可以按许可使用网络并生成 checksum；评测阶段尽量挂载只读数据、关闭无关公网访问，并把模型/judge endpoint 列入 allowlist。

## 1. 纯 Python/API 与静态数据（13 项）

项目：

- 指令与推理：`MultiChallenge`、`HMMT Nov 25`
- 多语言：`MMLU-ProX`、`NOVA-63`、`INCLUDE`、`Global PIQA`、`MAXIFE`
- 视觉推理：`DynaMath`、`VlmsAreBlind`、`BabyVision`
- 文档/OCR：`CharXiv(RQ)`、`MMLongBench-Doc`、`CC-OCR`

所需基础设施：

- 每项独立的 Python 环境和锁定的 harness commit；不能把名称相似的 EvalScope task 当成同一协议。
- 固定数据 revision、split、样本 ID、checksum，以及可重复生成的数据 manifest。
- OpenAI-compatible 文本/图片请求客户端；多模态任务还要固定图片顺序、缩放、`max_pixels`、PDF 渲染 DPI、页数和长文档截断策略。
- 固定答案抽取与聚合。MMLU-ProX 需要固定 29 种语言及平均方式；MAXIFE 需要固定 English 与 multilingual original prompts 的 23 个 setting；BabyVision 要保留模型卡中的双分数口径。
- 若 benchmark 使用 LLM judge，judge 必须与被测模型隔离，并记录模型名、服务版本、prompt、temperature、重试和原始判决。

不能只靠 `pip` 解决：

- 数据访问许可、非公开 test label、官方评分服务或 judge 配额。
- `HMMT Nov 25` 与本地 February adapter 的题集差异，`Global PIQA` 与普通 PIQA 的数据差异。
- 文档原始 PDF、OCR 标注和官方 scorer；仅安装 PDF/OCR 库不会自动得到正确数据与模型卡协议。
- 两个被测端点是否都能正确接收图片和多页文档，需要真实 transport smoke 验证；服务配置保留 vision 能力不等于链路已经验证。

建议放在 `python-api` worker。数据目录只读挂载，运行工作区按 `run-id` 隔离；API token 只从进程环境或 secret provider 注入，禁止写入 manifest、命令行参数和 stdout。

## 2. 搜索与通用 Agent（7 项）

项目：

- 通用 Agent：`VITA-Bench`、`DeepPlanning`
- 搜索 Agent：`HLE w/ tool`、`Browsecomp`、`Browsecomp-zh`、`WideSearch`、`Seal-0`

所需基础设施：

- 可审计的 agent loop：工具 schema、max turns、单步和整题 timeout、失败重试、token 预算、完整 tool-call/tool-response 轨迹。
- 搜索 backend、网页抓取器、正文抽取、URL 去重和时间戳；中文搜索任务需要明确中文检索源。
- VITA-Bench 和 DeepPlanning 所需的工具 fixture、环境/用户模拟器和最终状态判定。
- 独立 judge 服务及固定判分 prompt。
- 明确区分上下文策略：大多数搜索任务需要模型卡所述的 256K context folding；`WideSearch` 使用 256K 且不做 context management。两者不能共用一个静默裁剪实现。

不能只靠 `pip` 解决：

- 搜索服务账号、凭据、额度、公网出口和站点访问策略。
- 实时搜索索引与网页内容会变化；若没有搜索快照，结果不能严格复现。
- Agent fixture、用户模拟器、隐藏成功状态和外部 judge 预算。
- 262K 模型服务长度只是服务上限，不会自动实现 context folding，也不能保证工具轨迹在预算内安全完成。

建议使用独立 `search-agent` worker，并采用默认拒绝的网络策略：允许模型 endpoint、judge endpoint 和明确批准的搜索/网页域名；拒绝 RFC1918 内网、loopback、link-local、云元数据地址以及 `file://`。搜索凭据不得暴露给被测模型，网页内容按不可信输入处理。

## 3. 代码执行沙箱（4 项）

项目：`CodeForces`、`OJBench`、`FullStackBench en`、`FullStackBench zh`。

所需基础设施：

- 一次性容器或微型 VM；限制 CPU、RAM、PID、磁盘、wall-clock、输出大小和网络。
- 固定编译器、语言 runtime、系统包和容器镜像 digest。
- 隐藏测试、评分脚本、退出状态采集以及每题全新 workspace。
- FullStackBench 还需要浏览器、前后端服务、数据库/缓存等题目依赖，以及端口分配和服务就绪检查。

不能只靠 `pip` 解决：

- Docker/容器 runtime 权限与安全隔离；Python 包无法替代 kernel namespace、cgroup 或 VM。
- CodeForces 模型卡注明使用自有 query set，缺少该题集就无法复现其分数。
- OJBench 的隐藏测试或官方 judge；FullStackBench 的完整浏览器/服务镜像和任务资产。
- 编译和端到端测试所需的 CPU、RAM、磁盘与执行时间。

建议放在独立 `code-sandbox` worker，不要与模型服务容器共享 Docker socket。禁止 privileged、host network、宿主路径写挂载和持久化容器；默认关闭公网，仅按题目 manifest 临时开放明确依赖，并对下载内容做缓存和 checksum。

## 4. GUI、OS 与 Android（3 项）

项目：`ScreenSpot Pro`、`OSWorld-Verified`、`AndroidWorld`。

所需基础设施：

- ScreenSpot Pro 的截图资产、坐标系/缩放规范、grounding 输出解析和动作命中 scorer。
- OSWorld-Verified 的固定 VM 镜像、桌面应用版本、任务服务器、截图/键鼠执行器和每题快照恢复。
- AndroidWorld 的 Android SDK、Emulator、ADB、固定 system image、APK/应用状态和每题快照恢复。
- 动作轨迹、截图/录像、单步 timeout、总步数上限和基于环境状态而非模型自述的成功判定。

不能只靠 `pip` 解决：

- KVM/虚拟化权限、display server、VM 镜像、Android system image 和硬件加速。
- 桌面应用/APK 的固定版本、账号状态、镜像许可和任务初始快照。
- 对真实 GUI 的安全隔离；不能让 benchmark agent 操作宿主桌面或复用个人账号。

建议使用独立 `gui-android` worker，最好不部署在 H100 模型服务节点。每题从已知快照启动，任务结束后销毁；VM/模拟器只允许访问被测模型 endpoint 和任务明确需要的网络，禁止挂载仓库凭据、SSH 配置和用户主目录。

## 5. 视频、3D/空间与医疗数据（19 项）

项目：

- 空间/3D：`ERQA`、`CountBench`、`ODInW13`、`EmbSpatialBench`、`RefSpatialBench`、`LingoQA`、`Hypersim`、`SUNRGBD`、`Nuscene`
- 视频：`VideoMME(w sub.)`、`VideoMME(w/o sub.)`、`VideoMMMU`、`MLVU`、`MVBench`、`LVBench`、`MMVU`
- 医疗：`SLAKE`、`PMC-VQA`、`MedXpertQA-MM`

所需基础设施：

- 视频解码器和固定策略：backend/版本、帧率或采样点、最大帧数、分辨率、字幕编码、字幕注入位置和长视频截断。
- `VideoMME(w sub.)` 与 `VideoMME(w/o sub.)` 必须是两个独立协议，不能在结果阶段才按同一输出改名。
- 3D/空间数据的图像、深度图、点云、相机参数、检测/grounding 标注和官方 scorer。
- 医疗 VQA 的许可清单、数据版本、视觉预处理、答案规范及可审计 scorer。
- 大容量只读数据盘、足够的解码 CPU/内存，以及按样本记录媒体 checksum 和预处理参数的 manifest。

不能只靠 `pip` 解决：

- 视频、深度图和点云资产的下载体积、账号、许可和固定版本。
- Hypersim、SUNRGBD、nuScenes 等数据的目录结构、标注及使用条款；安装 SDK 不等于获得数据授权。
- ERQA、ODInW13、RefSpatialBench 等任务的 grounding/detection 标注和官方 scorer。
- SLAKE、PMC-VQA、MedXpertQA-MM 的数据许可、可能的非公开测试答案和医疗数据治理。
- 模型是否支持原生视频、深度图或点云输入。若只能抽帧成图片序列，必须把它记录为单独协议；不支持目标模态时应标记 `N/A` 或 `BLOCKED`，不能记作能力失败。

建议使用独立 `media-data` worker。`${EXTERNAL_DATA_ROOT}` 只读挂载，预处理产物写入按数据 revision 隔离的缓存，运行工作区可删除。受许可数据不进入 Git、普通日志或通用对象存储；医疗任务仅使用正式 benchmark 数据，不引入真实患者信息，结果也不得解释为临床结论。

## 接入门槛

每项 external benchmark 只有同时满足以下条件，才能从 `BLOCKED` 升级为可调度：

1. harness 来源与 revision 已固定，并通过自身 smoke test；
2. 数据许可、revision、split 和 checksum 已记录；
3. worker 类别、资源上限、网络策略和只读挂载已配置；
4. 被测模型输入协议已经用真实样本验证；
5. scorer/judge、聚合方式和模型卡口径已固定；
6. dry-run 能在不发起正式全量请求的情况下验证以上前置条件；
7. 缺少任一条件时输出明确的 `BLOCKED: <reason>`，而不是零分、空分或成功状态。
