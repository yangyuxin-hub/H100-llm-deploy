# 06 观测与验收

本页讲"怎么观测"和"跑完怎么验收":输出目录结构、每阶段看什么、日志路径速查、判断卡住的信号、性能观测、验收清单、写卡模板。

## 输出目录结构

P0 和 P1 的结果都在 node1 上。P0 输出目录:

```text
# node1 上 /root/llm-deploy/logs/eval/modelcard/<run-id>/
logs/eval/modelcard/<run-id>/
├── environment.json
├── commands.log
├── suite_status.tsv
├── issue_log.tsv
└── <benchmark>/<model>/
    ├── command.sh
    ├── runner.log
    ├── progress.json
    ├── frozen_config/
    ├── predictions/
    ├── reviews/
    └── reports/
```

P1 输出位于 node1 的 `/mnt/nvme0/llm-deploy-eval-logs/modelcard/p1-ranked/<run-id>/`,结构与上面一致。

## 日志路径速查表

P0 和 P1 的结果都在 node1 上。P0 命令在 node1 上直接执行(先 `ssh root@10.16.11.24` 进去);P1 从本地触发,用 `ssh root@10.16.11.24` 远程查看。

| 我想看 | 路径 / 命令 |
|---|---|
| P0 整体状态 | 在 node1 上:`logs/eval/modelcard/<run-id>/suite_status.tsv` |
| P0 单任务进度 | 在 node1 上:`logs/eval/modelcard/<run-id>/<benchmark>/<model>/progress.json` |
| P0 单任务日志 | 在 node1 上:`logs/eval/modelcard/<run-id>/<benchmark>/<model>/runner.log` |
| P0 续跑后台日志 | 在 node1 上:`logs/eval/modelcard/<run-id>/resume-launcher.log` |
| P1 整体状态 | `ssh root@10.16.11.24 'tail -n 30 /mnt/nvme0/llm-deploy-eval-logs/modelcard/p1-ranked/<run-id>/suite_status.tsv'` |
| P1 最新状态(去重) | node1 同目录下 `latest_status.tsv` |
| 单项报告 | `<run-dir>/<benchmark>/<model>/reports/*.json` |
| 单项预测 | `<run-dir>/<benchmark>/<model>/predictions/*.jsonl` |
| 单项评分 | `<run-dir>/<benchmark>/<model>/reviews/*.jsonl` |
| 实际执行命令 | `<run-dir>/<benchmark>/<model>/command.sh` 或 `<run-dir>/commands.log` |
| 冻结配置 | `<run-dir>/<benchmark>/<model>/frozen_config/` |
| 运行环境 | `<run-dir>/environment.json` |

`<run-dir>` 对 P0 是 `logs/eval/modelcard/<run-id>`(在 node1 的 `/root/llm-deploy/` 下),对 P1 是 `/mnt/nvme0/llm-deploy-eval-logs/modelcard/p1-ranked/<run-id>`。

## 每阶段观测点

| 阶段 | 关键观测 | 通过信号 |
|---|---|---|
| dry-run | `suite_status.tsv` 全为 `DRY_RUN` 或 `BLOCKED_*` | 命令和配置能生成,无意外报错 |
| preflight | `suite_status.tsv` 的 `BLOCKED_*` | 需要的环境/数据/judge 就绪;`BLOCKED_EXTERNAL` 不强行解除 |
| smoke | `progress.json`、`reports/*.json`、`suite_status.tsv` | `SUCCESS`;无 HTTP 错误/空输出/全截断 |
| full | `progress.json` 推进、远端 `Running>0` | `progress=100%` 且 report 题量符合预期 |
| 续跑 | `suite_status.tsv` 跳过 `SUCCESS` | 未完成项重新检查,不重复已完成项 |

查看状态和进度(在 node1 上执行,P0 直接看;P1 用 `ssh root@10.16.11.24` 远程看):

```bash
RUN_DIR="logs/eval/modelcard/${RUN_ID}"
tail -n 30 "${RUN_DIR}/suite_status.tsv"
tail -f "${RUN_DIR}/mmlu_pro/qwen/runner.log"
jq . "${RUN_DIR}/mmlu_pro/qwen/progress.json"
```

## 判断卡住的信号

| 现象 | 优先判断 | 处理 |
|---|---|---|
| `progress.json` 长时间不更新 | node1 上的 runner/PID 可能退出 | 检查 PID、`runner.log` 末尾,再用同 run id `--resume` |
| node1 `Running=0` 且进度不更新 | 当前没有评测请求 | 检查 node1 上的评测进程和 progress 更新时间 |
| node1 `Running=0` 但进度在更新 | 评测在请求间隙 | 正常,继续等 |
| `progress=100%` | 不等于结果已验收 | 仍要检查 report、题量和异常 |
| `suite_status.tsv` 长时间无新行 | 整项未结束或卡住 | 看 `runner.log` 和 `issue_log.tsv` |

查看 node1 服务是否收到请求:

```bash
ssh root@10.16.11.24 'docker logs --since 1m -f qwen3.6-27b-fp8 2>&1' \
  | grep --line-buffered -E 'Avg prompt throughput|SpecDecoding metrics|Running|Waiting'
```

判断原则:

- `progress.json` 在更新:评测仍在推进。
- node1 `Running=0` 且进度长时间不更新:优先检查 node1 上的 runner/PID。
- `progress=100%` 不等于结果已验收,仍要检查 report、题量和异常。
- `suite_status.tsv` 只在整项结束后写入 `SUCCESS` 或 `FAILED`。

## 性能观测(可选)

跑 full 时如果想观察服务端吞吐和 MTP 接受率,看远端容器日志关键字段:

```bash
ssh root@10.16.11.24 'docker logs --since 5m qwen3.6-27b-fp8 2>&1' \
  | grep -E 'Avg prompt throughput|Avg generation throughput|SpecDecoding metrics|GPU KV cache usage'
```

- `Avg generation throughput`:生成 token/s,反映实际吞吐。
- `SpecDecoding metrics`:MTP 接受率和平均接受长度。
- `GPU KV cache usage`:显存占用百分比,接近 100% 会触发排队。
- `Running` / `Waiting`:当前在跑和排队请求数。

独立的性能压测(并发扫描、长上下文延迟、soak)不在本手册范围,见 `benchmarks/evalscope/run_perf_suite.sh` 和 `logs/bench/` 下的历史报告。

## 验收正式结果

一个 benchmark 只有通过以下检查才能写入模型卡。

### 状态和文件

- Qwen 和 ThinkingCap 都有对应 full phase 的 `SUCCESS`。
- prediction、review、report、runner.log、command.sh 和 frozen_config 完整。
- 实际执行命令与计划一致。

### 题量

同时记录:

```text
官方/原始题量
计划题量
生成题量
实际计分题量
排除题量及原因
```

不能把生成题量当成计分题量。例如 IFBench 可能生成 300 条但汇总只计分 294 条,必须同时保留两个分母。

查找报告:

```bash
find "${RUN_DIR}" -path '*/reports/*' -type f -name '*.json' | sort
find "${RUN_DIR}" -path '*/predictions/*' -type f -name '*.jsonl' | sort
find "${RUN_DIR}" -path '*/reviews/*' -type f -name '*.jsonl' | sort
```

### 配置一致性

两模型应具有相同的:

- dataset args、subset、split、few-shot;
- generation config、seed、timeout 和输出预算;
- prompt/chat template、答案抽取和评分器;
- judge、sandbox、用户模拟器和外部 worker 版本。

(BFCL/TAU2 的 MODEL_SPECIFIC 差异除外,依据见[附录 A](appendix-a-bfcl-tau2-rationale.md)。)

### 异常和截断

必须核对:

- HTTP 失败、超时和重试;
- 空答案、解析失败和未评分样本;
- 达到 `max_tokens` 的样本;
- 上下文超限和被过滤样本;
- judge 失败、sandbox 失败和工具调用失败。

达到 `max_tokens` 的错误不能直接解释为模型基础能力差异。LongBench v2 只验收预算内样本,同时保留超限清单。

### 可比性结论

- 两个本地模型(都部署在 node1):只有同配置、同题量后才可做 A/B。
- Qwen 122B 官方分:根据已公开协议标记 `Strict` 到 `Not reproducible`。
- smoke、自评 judge、近似数据和部分 subset 不能混入正式主分数。

## 写入模型卡

更新顺序:

1. 在 [benchmark-comparison.md](benchmark-comparison.md) 写主分数、实际计分题量和差值。
2. 本手册各拆分文档只补充可复用的新流程或限制,不维护实时进度。
3. 在 `PROJECT_LOG.md` 和当日项目日志记录运行、错误、修复和结论。
4. 原始结果继续保留在 node1 的 `logs/eval/` 或 `/mnt/nvme0/llm-deploy-eval-logs/`,不复制进报告目录。

单项结果至少记录:

```text
Benchmark：
Run ID / 完成时间：
模型与 served name：
Dataset revision / subset / split：
Few-shot / prompt / chat template：
temperature / top_p / top_k / max_tokens / seed：
计划 / 生成 / 计分 / 排除题量：
两个模型主分数和绝对差：
失败 / 超时 / 解析失败 / 截断：
是否复用 cache：
122B 官方参考分和可比性等级：
prediction / review / report / frozen_config 路径：
结论与限制：
```
