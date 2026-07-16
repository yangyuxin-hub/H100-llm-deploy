# 剩余 75 项 benchmark 环境配置状态

日期：2026-07-14

## 结论

本轮目标从重复运行 7 项核心 smoke 调整为配置模型卡清单中其余 75 项。当前已为 75 项生成独立任务目录、协议骨架、Qwen/ThinkingCap 请求配置和机器可读状态。

按环境可调度性统计：

| 状态 | 数量 | 说明 |
|---|---:|---|
| 本地 runner 已生成，可进入 preflight | 23 | 依赖和 runner 就绪；其中部分仍需显式接受 `review` 协议后才可正式运行 |
| 缺独立 judge | 5 | AA-LCR、HLE w/ CoT、ZEROBench、SimpleVQA、TIR-Bench |
| 外部 harness / 数据 / worker 阻塞 | 46 | 搜索、Agent、代码、视频、3D、GUI、Android、医疗或非公开协议 |
| 协议阻塞 | 1 | ZEROBench_sub：当前 adapter 不暴露模型卡所需子分数 |

严格按协议冻结状态统计，75 项任务骨架当前为：`READY_PREFLIGHT=5`、`SMOKE_QWEN_SUCCESS=1`、`BLOCKED_PREFLIGHT=22`、`BLOCKED_EXTERNAL=46`、`BLOCKED_PROTOCOL=1`。这里的 `BLOCKED` 表示基础设施或协议缺口，不表示模型能力失败。

## 已完成的环境

- 基础 EvalScope：`.eval-deps/evalscope-1.6.1`，EvalScope 1.6.1、Python 3.10.20、torch 2.13.0。
- Language extras：IFEval、IFBench、BFCL、sandbox、SWE-bench、TAU2。
- Vision extras：OCRBench、OmniDocBench、RefCOCO。
- Terminal-Bench 2：独立 Python 3.12 + Harbor 环境。
- WMT24++：独立 `.eval-deps/evalscope-wmt-1.6.1`，torch 1.13.1、transformers 4.33.3、unbabel-comet 1.1.3、numpy 1.26.4。
- 本地 Docker daemon、代码 sandbox 权限和 TAU2 数据目录检查通过。

`bash benchmarks/evalscope/setup_eval_environments.sh check` 已完整通过。WMT 的 Python scorer 环境已经可 import，但正式评分仍需固定并准备 XCOMET 模型 revision，不能把“包已安装”写成“正式协议已冻结”。

## WMT 环境调试记录

首次执行：

```bash
bash benchmarks/evalscope/setup_eval_environments.sh install-translation
```

原始问题：`evalscope[wmt]==1.6.1` 解析到 `unbabel-comet==1.1.3`，它要求 `torch<2`、`pytorch-lightning==1.6.4` 和 `protobuf<=3.20.1`，因此把共享环境的 torch 2.13.0 降到 1.13.1、protobuf 6.33.6 降到 3.20.1。共享环境中的 transformers 5.13.1 又要求 torch >=2.4，最终 `import comet` 报 `NameError: LRScheduler is not defined`。

修复：

1. 将共享环境恢复到 torch 2.13.0 和 protobuf 6.33.6；
2. 为 WMT 新建独立 Python 3.10 venv；
3. 固定 `transformers==4.33.3`、`setuptools==80.9.0` 和 `numpy==1.26.4`；
4. runner 遇到 `environment=wmt` 时切换到独立 EvalScope；
5. 对共享环境和 WMT 环境分别执行真实 import 回归。

最终两个环境均通过，避免老版本 COMET 依赖影响其他 benchmark。

## 剩余任务 smoke

从剩余 75 项中选择 `HMMT Feb 25`，只在 Qwen 端点运行 1 题：

```bash
bash benchmarks/evalscope/run_reference_suite.sh \
  --model qwen \
  --api-url "${QWEN_API_URL}" \
  --benchmark 'HMMT Feb 25' \
  --limit 1 \
  --eval-batch-size 1 \
  --generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"max_tokens":16384}'
```

结果：题库加载 30 条，实际尝试 1 条，成功请求 1 条，解析/评分成功 1 条，runner 退出码 0。输出为 13,483 tokens，报告和 HTML 均生成。该 1/1 分数只证明数据、请求、答案抽取和聚合链路跑通，不作为模型质量结论。

证据目录：`logs/eval/modelcard/remaining-smoke/hmmt25-20260714/`。

## 生成物

- `benchmarks/modelcard/tasks/`：剩余 75 项逐项目录；
- `benchmarks/modelcard/tasks/index.json`：统一状态索引；
- `benchmarks/modelcard/scaffold_remaining_tasks.py`：从参考 manifest 可重复生成任务骨架；
- `logs/eval/modelcard/environment-preflight/remaining-75-20260714/`：82 项离线 dry-run 证据；
- `benchmarks/evalscope/EXTERNAL_ENVIRONMENTS.md`：46 项外部 worker、安全和数据边界。

## 下一步边界

下一步应优先对 23 项本地 runner 逐项完成协议核对和数据 preflight，而不是继续扩大核心任务跑分。46 项外部任务需要用户另行授权凭据、许可数据或 VM/Android/搜索基础设施；在这些条件满足前保持 `BLOCKED_EXTERNAL`。
