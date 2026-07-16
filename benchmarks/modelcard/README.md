# 模型卡评测配置

本目录承载三模型模型卡对比的可追溯配置，不代表所有 benchmark 都已经可以正式运行。

- `core_tasks.tsv`：第一阶段 7 项核心任务配置；
- `run_core_evaluations.sh`：核心任务双模型 runner；
- `run_remaining_evaluations.sh`：续跑 MMLU-Redux、C-Eval、LongBench v2，支持部分 cache 和后台运行；
- `p1_tasks.tsv`：P1 十项按能力价值排序后的冻结协议、任务型生成参数、题量和阻塞状态；
- `prepare_p1_data.sh`：下载并校验固定 revision 的 HMMT Nov 2025 数据；
- `run_p1_evaluations.sh`：按 benchmark 串行、双模型并行的 P1 runner；
- `run_p1_ranked_evaluations.sh`：十项统一自动调度、后台运行、续跑和状态去重入口；

P1 默认按以下能力优先级运行：

1. Agent/工具调用：BFCL-V4、TAU2-Bench；
2. 代码：LiveCodeBench v6；
3. 高难数学：HMMT Nov 25、HMMT Feb 25、PolyMATH；
4. 专业与多语言知识：SuperGPQA、MMMLU；
5. 尚缺独立 judge：HLE w/ CoT、AA-LCR。

生成参数按任务和模型分别冻结。BFCL 两模型都使用 Qwen 官方 non-thinking 配置（`temperature=0.7, top_p=0.8, presence_penalty=1.5`），避免实测 reasoning loop/伪 XML 并专注原生 Function Calling。TAU2 的 Qwen 保留 thinking 和 `presence_penalty=0.5`，因为实测 non-thinking 会退化为文本伪 tool call；ThinkingCap 使用低随机性的 non-thinking Function Calling 采样（`temperature=0.7, top_p=0.8, top_k=20, presence_penalty=0`），`presence_penalty=0` 避免惩罚复制必填 ID。TAU2 单轮输出上限为 8,192 tokens，避免已观测到的 81,920-token runaway 长时间占用服务；其他被测模型任务仍使用完整 81,920-token 输出预算。TAU2 被测 agent 的非法空响应不重试、不改写，作为该样本 0 分保留在 pass@1 分母中并继续后续样本。不传 `preserve_thinking`，不强制 `tool_choice=required`。TAU2 用户模拟器固定为 `temperature=0`、关闭 thinking、无输出上限。完整参数以 `p1_tasks.tsv` 为唯一机器可读来源。

P1 首次运行前：

```bash
bash benchmarks/modelcard/prepare_p1_data.sh
bash benchmarks/modelcard/run_p1_evaluations.sh --mode preflight
bash benchmarks/modelcard/run_p1_evaluations.sh --mode smoke
```

日常使用优先调用统一自动化入口：

```bash
# 在 node1 一键同步并后台跑新配置 smoke
bash scripts/run_remote_p1_ranked_tests.sh \
  --mode smoke \
  --run-id p1-ranked-smoke-20260715 \
  --background

# 查看统一状态
ssh root@10.16.11.24 'tail -n 30 /mnt/nvme0/llm-deploy-eval-logs/modelcard/p1-ranked/p1-ranked-smoke-20260715/suite_status.tsv'

# 同一 run id 断点续跑；runner 自动跳过 SUCCESS
bash scripts/run_remote_p1_ranked_tests.sh \
  --mode smoke \
  --run-id p1-ranked-smoke-20260715 \
  --background \
  --no-prepare-data
```

`latest_status.tsv` 会按 benchmark/model/mode 保留最后一次状态并恢复优先级顺序，适合自动验收；`suite_status.tsv` 保留所有尝试，适合审计。`BLOCKED_*` 和 `FAILED` 在续跑时会重新检查，配置好 judge 或修复环境后无需换 run id；dry-run 记录也不会阻止后续真实运行。

`run_p1_evaluations.sh --mode smoke` 会为多 subset 任务固定一个代表 subset，再运行 1 题；PolyMATH 因 adapter 固定加载四个难度层级，会运行英文的 4 题。HLE w/ CoT 和 AA-LCR 在 smoke 中显式使用被测模型自评，只验证 judge 链路，分数不能进入正式模型卡；正式运行仍要求独立 judge。LiveCodeBench v6 只有通过 runner 内置的 Docker sandbox gate 后才会调度。正式全量用 `--mode full`，并可用同一 `--run-id --resume` 恢复。修改生成参数后必须使用新 run id 重跑 smoke，旧 smoke 只能证明数据、请求、执行和评分链路曾经跑通。

共享环境检查和 82 项离线 dry-run 入口见 [`../evalscope/README.md`](../evalscope/README.md)。

当前正式 run 直接后台续跑：

```bash
bash benchmarks/modelcard/run_remaining_evaluations.sh --background
```

查看后台日志与状态：

```bash
tail -f logs/eval/modelcard/modelcard-core-full-20260714/resume-launcher.log
tail -n 20 logs/eval/modelcard/modelcard-core-full-20260714/suite_status.tsv
```

默认顺序是 MMLU-Redux、C-Eval、LongBench v2。MMLU-Redux 会复用已保存的 prediction/review cache；单跑一项时增加 `--benchmark ceval`。冻结配置不一致时脚本会拒绝复用，避免把不同协议的结果混在一起。
