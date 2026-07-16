# 评测

- `evalscope/`：EvalScope 1.6.1 能力、Agent、长上下文与性能自动化套件。
- `modelcard/`：模型卡 P0/P1 评测的冻结配置、runner 和数据准备工具。

性能与能力原始结果分别放到 `logs/bench/<model>-<date>-<experiment>/` 和 `logs/eval/`；只把重要结论写入 `PROJECT_LOG.md` 或 `docs/reports/`。

```bash
bash benchmarks/evalscope/run_all.sh --models qwen,thinkingcap --dry-run
```
