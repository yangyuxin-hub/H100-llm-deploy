# ThinkingCap 基础能力评测（2026-07-13）

## 结论

ThinkingCap-Qwen3.6-27B-FP8 在本次 MMLU-Pro pilot 子集上完成 280 题，`exact_match` 为 **84.29%**（lm-eval stderr ±2.16 个百分点）。这是 ThinkingCap 的单模型能力结果，不是与 stock Qwen 的有效 A/B 差值：用户要求先停掉 Qwen 评测，Qwen 在 280 题中完成 175 题后被中断，因此没有生成可比较的完整分数。

## 评测口径

- 框架：`lm-eval 0.4.12`，`local-chat-completions`，调用远程 OpenAI-compatible `/v1/chat/completions`。
- 任务：MMLU-Pro，14 个学科，每学科抽取 20 题，共 280 题；5-shot；seed=42。
- 解码：`temperature=0`、`max_gen_toks=8192`、`until=[]`。`until=[]` 是为避免模型思维文本中出现 `Question:` 导致提前截断；两个模型必须使用同一口径。
- 统计：按 lm-eval 的 custom extraction（`the answer is (X)`）计算 exact match；stderr 是该子集的抽样标准误，不代表完整 MMLU-Pro 官方成绩。
- 评测期间未重启或停止线上 Docker 容器；只停止了本地 Qwen 评测进程。

## 分学科结果

| 学科 | 题数 | ThinkingCap |
|---|---:|---:|
| biology | 20 | 95.0% |
| business | 20 | 85.0% |
| chemistry | 20 | 85.0% |
| computer_science | 20 | 90.0% |
| economics | 20 | 90.0% |
| engineering | 20 | 85.0% |
| health | 20 | 70.0% |
| history | 20 | 75.0% |
| law | 20 | 70.0% |
| math | 20 | 95.0% |
| other | 20 | 75.0% |
| philosophy | 20 | 90.0% |
| physics | 20 | 100.0% |
| psychology | 20 | 75.0% |
| **macro/pooled** | **280** | **84.29%** |

## 数据与复现

- 聚合结果：[results JSON](../../../../logs/eval/thinkingcap-basic-20260713/mmlu-pro-subset/thinkingcap-mmlu-pro/thinkingcap-qwen3.6-27b-fp8/results_2026-07-13T17-27-33.489556.json)
- 每题原始样本：同目录下 `samples_mmlu_pro_*.jsonl`。
- Qwen 中断日志保留在 `logs/eval/qwen-basic-20260713/mmlu-pro-subset/` 的运行输出中；该运行未产生可用于最终汇总的完整 results 文件。

## 限制与下一步

1. 20 题/学科只能作为回归 smoke/pilot，不能替代完整 MMLU-Pro；若要判定小幅回归，建议至少 100 题/学科，并固定同一题目 ID。
2. 本轮 `temperature=0` 便于回归复现，但不是模型卡中的多 seed 采样协议；正式发布报告应补 3–5 个 seed 或报告逐题配对置信区间。
3. 下一次只需重跑 Qwen 同一命令并复用 ThinkingCap 的题目顺序，即可计算 pooled、分学科以及逐题配对 delta；不要把吞吐/延迟 benchmark 混入能力分数。
