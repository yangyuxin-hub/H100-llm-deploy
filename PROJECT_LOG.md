# 大模型部署实习项目记录

## 当前状态

截至 2026-07-16：

| 用途 | 模型 | GPU | TP | 端口 | 状态 |
|---|---|---:|---:|---:|---|
| 稳定服务 | Qwen3.6-27B-FP8 | 0,1 | 2 | 8000 | 已验证可用 |
| 实验服务 | ThinkingCap-Qwen3.6-27B-FP8 | 2,3 | 2 | 8001 | 已验证可用 |
| 历史模型 | Agents-A1-FP8 | 2,3 | 2 | 8001 | 已下线，保留配置 |

- node1：4×H100 80GB，driver 550.144.03；禁止重启宿主机；**node1 无公网**，所有评测数据、模型和依赖必须本地准备后上传。
- 两个当前服务均使用 vLLM 0.24.0、TP2、MTP3、FLASH_ATTN、FP8 KV cache 和 `max-num-seqs=16`。
- ThinkingCap 默认开启 thinking，并通过 `qwen3` parser 分离 `reasoning` 与最终 `content`。
- 本地 `scripts/status.sh` 只反映本机；远端状态需在 node1 执行或通过 Docker/API 验证。
- 七项核心评测已于 2026-07-15 15:15 全部完成，run id 为 `modelcard-core-full-20260714`；14 个模型—任务状态均为 `SUCCESS`。MMLU-Redux 通过同一 run id 的落盘缓存续跑完成，随后 C-Eval 与 LongBench v2 也正常结束；当前本地无 EvalScope 评测进程。
- P1 最终 ranked smoke 已于 2026-07-15 20:01 完成，run id 为 `p1-ranked-smoke-validated-20260715-1946`；十项、双模型共 20 个状态全部为 `SUCCESS`，无 `FAILED`。BFCL `simple_python` 双模型均为 1/1；TAU2 的 raw audit、trajectory 与 reward 链路均完整。该次 airline Task 1 双模型 Reward 都为 0，属于策略动作结果；此前 ThinkingCap 低随机性最终配置的两个独立新 run 均为 Reward 1.0，说明单题 pass@1 有采样波动，不能挑最好轮次代替 full 结果。
- P1 其余 8 项在同一最终 run 中完成双模型端到端 smoke：LiveCodeBench v6、HMMT Nov/Feb、PolyMATH、SuperGPQA、MMMLU、HLE w/ CoT、AA-LCR 的数据加载、推理、review 和报告链路均跑通。所有被测任务使用 81,920 完整输出预算；PolyMATH 的长输出自然完成，未恢复 4K/32K 截断。HLE 与 AA-LCR 当前仍只是 self-judge 链路 smoke，正式分数被独立 judge gate 阻塞。
- P1 参数按任务与模型冻结：BFCL 两模型为 `temperature=0.7, top_p=0.8, presence_penalty=1.5, enable_thinking=false`；TAU2 Qwen 为 `temperature=1.0, top_p=0.95, presence_penalty=0.5, enable_thinking=true`，ThinkingCap 为 `temperature=0.7, top_p=0.8, presence_penalty=0, enable_thinking=false`；其余 reasoning 任务通常为 `temperature=1.0`，精确代码为 `0.6`。所有被测模型任务 `max_tokens=81920`；TAU2 用户模拟器为 `temperature=0, enable_thinking=false` 且不传输出上限。P1 不传 `preserve_thinking`，不强制 `tool_choice=required`，不改写响应。
- P1 ranked full 已于 2026-07-16 重新启动，run id 为 `p1-full-20260716`。此次启动前修复了两个离线依赖问题：BFCL memory 任务所需 `all-MiniLM-L6-v2` 模型已下载到 `/mnt/nvme0/models/all-MiniLM-L6-v2/` 并 patch `bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_vector.py:26` 改用本地路径；HLE、SuperGPQA、HMMT Feb 25、MMMLU、PolyMATH、AA-LCR 六个数据集在 `run_reference_suite.sh` 的 `dataset_args_for()` 中改为 smoke/full 两种模式都传 `local_path`，避免 evalscope 在 full 模式下尝试联网下载。完整数据集已预下载到 `/root/llm-deploy/.eval-deps/data/p1-full-data/`。BFCL `memory_kv_prereq` 阶段已顺利推进，证明本地 MiniLM 补丁生效。HLE w/ CoT 与 AA-LCR 因未配置独立 judge，在 preflight 阶段会被标记为 `BLOCKED_JUDGE`，其余 8 项按 `p1_tasks.tsv` 能力优先级串行执行。

## 已完成

- Qwen 与 ThinkingCap 完成相同配置的并发 16 稳态复测：输出吞吐 2342.2 与 2354.3 tok/s，差异 0.52%，可视为持平。
- ThinkingCap 消融确认：默认温度从 1.0 降到 0.6 是 MTP 接受率上升的主因；运行配置对齐本身约带来 4% 吞吐收益。见 [MTP 接受率分析](docs/reports/archive/performance/2026-07-14-thinkingcap-mtp-acceptance-analysis.md)。
- 两模型完成 8K–140K 长上下文扫描，正式请求均零失败；详细过程见 [2026-07-13 日志](docs/project-log/2026-07-13.md)。
- ThinkingCap 完成 MMLU-Pro 280 题 pilot，exact-match 84.29%；Qwen 对照未跑完，不能形成有效 A/B。见 [能力评测报告](docs/reports/archive/evaluation-results/2026-07-13-thinkingcap-capability-eval.md)。
- ThinkingCap 评测 manifest、标准和数据准备工具已整理到 `benchmarks/thinkingcap/`(已移除,改由 `benchmarks/modelcard/` 与 `benchmarks/evalscope/` 承载)。
- 使用 `uv` 创建独立 EvalScope 1.6.1 环境：`.eval-deps/evalscope-1.6.1`，Python 3.10.20，安装 `evalscope[perf,app]`；已验证 `evalscope eval`、`evalscope perf` 和 Python import 均正常。
- 新增 `benchmarks/evalscope/` 自动化套件：按知识、数学、推理、指令、代码、长上下文、事实性和 Agent 分类运行能力评测，并支持并发、输入长度、输出长度、多轮、真实请求和 soak 性能测试。每项独立记录状态；代码执行必须显式启用 Docker sandbox，heavy 任务和缺失依赖默认跳过。
- 本地 EvalScope Docker sandbox 权限已验证：当前用户已在 `docker` 组，Docker daemon 与最小容器运行正常；在刷新组身份的会话中，`humaneval_plus`、`mbpp_plus`、`live_code_bench` 和 `swe_bench_verified_mini` 均通过 dry-run 预检，不再因 Docker socket 权限跳过。旧登录会话需重新登录或使用 `newgrp docker`/`sg docker` 获取新组身份。
- `tau2-bench` v0.2.0 源码已安装为 Python 包 `tau2==0.2.1.dev0`；其 602MB benchmark 数据保存在 `.eval-deps/tau2-data-v0.2.0`，能力评测脚本会自动设置并预检 `TAU2_DATA_DIR`。全量 dry-run 已达到 30 项全部可调度、零依赖跳过。
- 自动化套件完成端到端性能 smoke：Qwen 端点在并发 1、固定输入 1024 tokens、输出 256 tokens 下完成 3/3 请求；平均 TTFT 0.1023s、平均延迟 1.129s、输出吞吐 226.6 tok/s，EvalScope 解析的 MTP 接受率约 66.1%。该结果只验证脚本、SSH 隧道、tokenizer 和 vLLM 流式协议链路，不作为正式性能结论；原始结果在 `logs/bench/evalscope/validation-live-20260714/`。
- 整理 GLM-5.2、ThinkingCap-Qwen3.6-27B 与 Qwen3.5-122B-A10B 三张模型卡的评测集清单、交集和协议差异；共录入 122 条模型—评测记录。三者共同出现的核心集只有 GPQA-Diamond 与 HMMT Nov 2025，同名成绩仍需统一 prompt、seed、harness、judge 和输出预算后才能横比。见 [模型卡评测集清单](docs/reports/archive/evaluation-references/2026-07-14-model-card-benchmark-inventory/report.html)。
- 归档一组 Qwen3.5 模型卡横向评测分数：Language 37 项、Vision Language 45 项，共 82 个 benchmark 行，保留 6 个对照模型、缺失值、双分数和原表脚注。见 [Qwen3.5 模型卡评测分数参考表](docs/reports/archive/evaluation-references/2026-07-14-qwen35-benchmark-score-reference.md)。
- 新建七项核心跑分成绩单，集中记录 Qwen3.5-122B-A10B 官方参考分、两个本地 27B 模型实测分、任务进度、生成参数和 IFEval 复现方法。见 [七项核心跑分对比与测试配置记录](docs/reports/benchmark-comparison.md)。
- 更新 `benchmarks/modelcard/run_remaining_evaluations.sh`：默认续跑正式 run 中剩余的 MMLU-Redux、C-Eval、LongBench v2；核心 runner 在冻结配置一致时通过 EvalScope `--use-cache` 复用 MMLU-Redux 已落盘的 prediction/review，另支持 `--background` 和同 run id 重复启动保护。脚本只交付和校验，未代替用户启动。
- 将 `docs/reports/benchmark-test-process.md` 扩展为面向接手同事的七项核心评测复现手册，统一记录工具分工、冻结参数、标准流程、后台缓存续跑、进度/吞吐监控、完成验收、故障处理和单项结果模板；最终汇报仍只维护该手册与 `benchmark-comparison.md` 两个主入口。
- 将 `benchmark-test-process.md` 精简重构为完整模型卡测试流程，新增 P0–P4 分层、环境配置、数据冻结、preflight/smoke、自动化正式运行和验收主线；实时成绩继续只在 `benchmark-comparison.md` 维护。
- 根据可直接交接复现的要求，将该文档扩展为逐步操作手册：每一步补齐输入、命令、产物和通过标准，并恢复 P0/P1 参数、数据准备、smoke/full、监控、续跑、验收与故障处理细节；实时成绩仍与流程正文分离。
- 对照上述 82 项评测与当前 EvalScope 主套件、本地 1.6.1 adapter 注册表及已有结果完成覆盖差距盘点：精确接入 10 项、部分接入 2 项、可基于现有 adapter 补接 26 项、未接入 44 项；45 项视觉语言任务当前没有一项进入主套件。见 [评测覆盖差距报告](docs/reports/archive/evaluation-planning/2026-07-14-benchmark-gap-analysis/report.html)。
- 完成 7 项模型卡核心任务的双模型 smoke：除 LongBench v2 的 long 样本因输入至少 245,761 tokens、再预留 16,384 输出后超过 262,144 上限外，其余 6 项链路均完成；据此将 7 项核心任务和剩余 75 项整理为 P0–P4 执行队列，并加入视觉 gate、协议阻塞和成本止损规则。见 [模型卡评测最终方案](docs/reports/archive/evaluation-planning/2026-07-14-model-card-evaluation-plan.md)。
- 已启动 7 项 P0 核心任务的双模型正式全量运行，run id 为 `modelcard-core-full-20260714`：每个模型并发 4、同一 benchmark 两模型并行、benchmark 间串行，原始状态与结果写入 `logs/eval/modelcard/modelcard-core-full-20260714/`。LongBench v2 按真实 Qwen chat template 预先计数，503 题中保留不超过上下文预算的 400 题，另有 103 题明确记录为超限而不静默截断；EvalScope 1.6.1 本地 JSONL 加载兼容问题已修复并验证三个长度子集均可加载。详细过程见 [2026-07-14 日志](docs/project-log/2026-07-14.md)。
- IFEval 正式结果已完成并复核：ThinkingCap Prompt strict 87.80%，Qwen 84.84%，全量差 +2.96pp；但剔除任一模型达到 8,192-token 上限的 39 条配对提示后，差距缩小到 +1.20pp，95% bootstrap 区间跨 0。当前结论是 ThinkingCap 正式成绩和输出长度控制更好，但不能把全量差距全部解释为非截断样本上的基础指令遵循优势。见 [IFEval 双模型报告](docs/reports/archive/evaluation-results/2026-07-14-ifeval-comparison/report.html)。
- 七项 P0 核心正式评测全部完成：MMLU-Redux 为 Qwen 93.19%、ThinkingCap 93.19%；C-Eval 为 91.09%、90.79%；LongBench v2 在上下文预算内的 400/503 题上为 61.75%、63.25%，另 103 题因超出 262,144-token 上限未运行。完成的七项中 ThinkingCap 五项领先、一项持平、一项落后；详细分数和协议 caveat 见 [七项核心跑分对比与测试配置记录](docs/reports/benchmark-comparison.md)。
- 完成剩余 75 项 benchmark 的环境配置骨架和全量离线调度检查：23 项本地 runner/依赖可进入 preflight，5 项缺独立 judge，46 项缺外部 harness/数据/worker，1 项协议阻塞。WMT/COMET 因 torch/protobuf 版本冲突改为独立 Python 环境，共享评测环境已恢复并通过完整 import 回归；从剩余任务中抽取 HMMT Feb 25 在 Qwen 上完成 1 题端到端 smoke。见 [剩余 benchmark 环境状态](docs/reports/archive/evaluation-planning/2026-07-14-remaining-benchmark-environment-status.md)。
- 将 82 项评测的 P0–P4 划分、能力覆盖、全部 benchmark、执行 gate、环境可执行性和阶段产出整理为面向 leader 的统一表格，并明确区分业务优先级与当前环境状态。见 [模型卡 82 项评测优先级汇总](docs/reports/archive/evaluation-planning/2026-07-14-model-card-evaluation-priority-summary.md)。
- 简化 `docs/reports/` 最终入口：只保留持续维护的[跑分对比表](docs/reports/benchmark-comparison.md)和[测试过程与参数](docs/reports/benchmark-test-process.md)；早期 pilot、单项详报、规划、参考资料和性能分析移入 `docs/reports/archive/`。
- 新增 `scripts/run_remote_p1_agent_tests.sh` 与 `benchmarks/modelcard/run_p1_agent_evaluations.sh`(均已移除,功能合并到 ranked 入口)：本地一条命令同步必要 runner，在 node1 串行调度 BFCL-V4、TAU2-Bench，支持 preflight/smoke/full、后台运行、固定 run id 和断点续跑。远端离线环境使用 NVMe 上的 Python 3.10.20、EvalScope 1.6.1、BFCL scorer 与 TAU2 v0.2.0 数据，不依赖 node1 访问 GitHub、PyPI 或 ModelScope。
- 修复 ThinkingCap 在 TAU2 长策略提示词下把工具调用写成正文 `<invoke>`/JSON/`<tool_code>` 或丢失必填参数的兼容问题：[`config/chat_templates/thinkingcap_agent.jinja`](config/chat_templates/thinkingcap_agent.jinja) 在不修改 TAU2 官方提示的前提下，于服务层 system suffix 重申 canonical tool protocol 和无引号 parameter 标签。原始审计同时证明 ThinkingCap thinking 模式会偶发 reasoning-only 空响应，原生高随机性也会持续生成；最终 TAU2 因此采用低随机性 non-thinking 配置。只重启过 ThinkingCap 容器，Qwen 与 H100 宿主机未重启，评测端没有改写答案。
- 新增 `scripts/run_remote_p1_remaining_tests.sh`、`benchmarks/modelcard/run_p1_remaining_evaluations.sh`(均已移除,功能合并到 ranked 入口) 和 [`benchmarks/modelcard/prepare_p1_smoke_data.py`](benchmarks/modelcard/prepare_p1_smoke_data.py)：本地自动准备可审计的单样本 fixture 并同步到 node1 NVMe，远端按 benchmark 串行、双模型并行执行其余 8 项 P1 preflight/smoke/full。runner 现在会把 Docker sandbox 空池/执行失败和“所有输出均因 `max_tokens` 截断”转为非零退出，避免 EvalScope 退出码 0 造成假成功。
- 新增 [`scripts/run_remote_p1_ranked_tests.sh`](scripts/run_remote_p1_ranked_tests.sh) 和 [`benchmarks/modelcard/run_p1_ranked_evaluations.sh`](benchmarks/modelcard/run_p1_ranked_evaluations.sh)：一条命令按 `p1_tasks.tsv` 的能力优先级自动完成十项双模型 preflight/smoke/full，支持后台运行、固定 run id、断点续跑、单项筛选、BFCL subset、跨 run 全局锁、远端已有 P1 进程保护，以及原始/去重状态双文件。`DRY_RUN`、`BLOCKED_*` 和 `FAILED` 不会永久阻止后续真实重试。
- P1 新自动化入口已完成本地语法/manifest 校验、node1 远端 20/20 dry-run，以及最终 20/20 真实 smoke。`run_remote_p1_ranked_tests.sh` 会同步 runner 与固定 fixture、拒绝并行旧任务、使用跨 run 锁和新 run id，并输出 append-only `suite_status.tsv` 与去重 `latest_status.tsv`。

## 下一步

1. 最终 smoke 已 20/20 通过；下一步启动 ranked full。BFCL 按 20 个无凭据子集报告，不能写成官方完整 V4 overall；TAU2 核对 airline/retail/telecom 共 269 题、逐题轨迹和 raw audit。
2. 在 ThinkingCap 的 TAU2 full 中验证修复覆盖三个领域；若出现新的协议失败或持续生成，保留原始响应和轨迹，不在评测端静默改写，也不挑选最好重试轮次替代 pass@1。
3. 为 HLE w/ CoT 和 AA-LCR 配置独立 judge 后再跑正式评测；self-judge smoke 只证明链路可用，不能作为模型成绩。
4. 在 GPU2,3 对比 TP2 与两个 TP1 副本，以及 MTP0/2/3；不影响 GPU0,1 稳定服务。
5. 补测真实 coding/agent workload，比较成功率、tool calling、完成 token 和 time-to-correct。

## 文档入口

- 按日期操作记录：[`docs/project-log/`](docs/project-log/README.md)
- 推理学习笔记：[`docs/learning-notes/`](docs/learning-notes/INDEX.md)
- 评测与性能报告：[`docs/reports/`](docs/reports/README.md)
- 吞吐优化调研：[`docs/research/2026-07-13-inference-throughput-options.md`](docs/research/2026-07-13-inference-throughput-options.md)
- 原始性能/能力输出：`logs/bench/`、`logs/eval/`（不提交 Git）
