# EvalScope 1.6.1 自动化评测

本目录把能力评测和性能压测拆成可恢复的小任务。每个 benchmark 独立输出，即使某项下载失败、缺依赖或模型请求失败，其他任务仍可继续，并在 `suite_status.tsv` 中记录 `SUCCESS`、`FAILED`、`SKIPPED` 或 `DRY_RUN`。

脚本只请求已有的 OpenAI-compatible API，不会启动、停止或重启模型，也不会修改模型权重。

## 文件

- `capability_suites.tsv`：能力维度、benchmark、可选依赖和数据参数。
- `run_capability_suite.sh`：知识、数学、推理、指令、代码、长上下文、事实性和 Agent 评测。
- `run_perf_suite.sh`：固定长度、并发、上下文长度、输出长度、真实数据、多轮和 soak 测试。
- `run_all.sh`：按模型顺序运行能力和性能测试。为避免共享宿主机资源互相干扰，不并行压测两个模型。
- `reference_benchmarks.tsv`：Qwen3.5 模型卡 82 项参考评测的统一清单，记录本地 adapter、环境和协议状态。
- `setup_eval_environments.sh`：检查或安装 82 项清单中可由本地 Python 环境满足的依赖。
- `run_reference_suite.sh`：按模态、优先级或 benchmark 预检和调度 82 项参考清单。
- `config/evaluation.env`：参考套件共享的 EvalScope 路径、端点、数据目录、judge 和复现参数。

## 1. 准备端点

本地默认使用 SSH 隧道后的地址：

| 模型别名 | 默认 API 根地址 | 期望 served model |
|---|---|---|
| `qwen` | `http://127.0.0.1:18000/v1` | `qwen3.6-27b-fp8` |
| `thinkingcap` | `http://127.0.0.1:18001/v1` | `thinkingcap-qwen3.6-27b-fp8` |
| `agents` | `http://127.0.0.1:18001/v1` | `agents-a1-fp8` |

端口 8001 同一时间只能对应 ThinkingCap 或 Agents-A1。脚本开始前会读取 `/v1/models` 并要求模型名精确匹配，避免把 Agents 评测误发给 ThinkingCap。

在单独终端建立隧道，将 `<node1>` 替换成已有 SSH Host：

```bash
ssh -N \
  -L 18000:127.0.0.1:8000 \
  -L 18001:127.0.0.1:8001 \
  <node1>
```

如果 EvalScope 与 vLLM 在同一台机器运行，可以覆盖地址：

```bash
QWEN_API_URL=http://127.0.0.1:8000/v1 \
bash benchmarks/evalscope/run_capability_suite.sh --model qwen
```

## 2. 先 dry-run

```bash
bash benchmarks/evalscope/run_all.sh \
  --models qwen,thinkingcap \
  --capability-suite all \
  --limit 5 \
  --perf-suite all \
  --dry-run
```

`dry-run` 不访问模型、不下载数据，只显示将运行的命令和缺失依赖。

## 3. 推荐执行顺序

### Smoke：先验证整条链路

每个能力维度选择一个代表任务，每个 subset 默认最多 2 题；缺少 sandbox 的代码任务会标记为 `SKIPPED`。

```bash
bash benchmarks/evalscope/run_all.sh --models qwen,thinkingcap
```

### Pilot：所有标准能力各抽 5 题

```bash
bash benchmarks/evalscope/run_all.sh \
  --models qwen,thinkingcap \
  --capability-suite all \
  --limit 5 \
  --perf-suite all
```

注意：`--limit` 是“每个 subset”的上限。比如 C-Eval 有 52 个 subset，`--limit 5` 最多会产生约 260 题，并不等于整个 benchmark 只有 5 题。

### 正式全量能力评测

```bash
bash benchmarks/evalscope/run_all.sh \
  --models qwen,thinkingcap \
  --capability-suite all \
  --limit 0 \
  --perf-suite none
```

全量需要较长时间。建议先按类别单独运行并核对输出，例如：

```bash
bash benchmarks/evalscope/run_capability_suite.sh \
  --model qwen --suite knowledge --limit 20

bash benchmarks/evalscope/run_capability_suite.sh \
  --model qwen --suite long_context --limit 5
```

## 4. 可选依赖

当前基础环境安装了 `evalscope[perf,app]==1.6.1`。脚本不会自动安装依赖，避免一次性拉取大型 Docker 镜像或改变环境。

使用 `uv` 安装所需 extra：

```bash
EVAL_PY=.eval-deps/evalscope-1.6.1/bin/python

uv pip install --python "${EVAL_PY}" 'evalscope[ifeval,ifbench,multi-if]==1.6.1'
uv pip install --python "${EVAL_PY}" 'evalscope[sandbox]==1.6.1'
uv pip install --python "${EVAL_PY}" 'evalscope[bfcl]==1.6.1'
uv pip install --python "${EVAL_PY}" 'evalscope[needle-haystack,openai-mrcr]==1.6.1'
uv pip install --python "${EVAL_PY}" 'evalscope[swe-bench]==1.6.1'
uv pip install --python "${EVAL_PY}" \
  'git+https://github.com/sierra-research/tau2-bench@v0.2.0'
```

`tau2-bench` 通过普通 wheel 安装时不会包含 benchmark 数据。脚本默认从
`.eval-deps/tau2-data-v0.2.0` 读取，并自动导出 `TAU2_DATA_DIR`；也可以在运行前用同名环境变量覆盖。
目录中应包含 `tau2/domains/`。如果 GitHub 的 Git fetch 很慢，可以下载 v0.2.0 源码归档，
从解压后的源码目录安装，并将其中的 `data/` 内容复制到上述目录。

代码 benchmark 默认不会执行。只有同时满足以下条件才运行：

1. 显式传入 `--enable-sandbox`；
2. 安装 `evalscope[sandbox]`；
3. 本地 Docker daemon 可用。

```bash
bash benchmarks/evalscope/run_capability_suite.sh \
  --model qwen --suite coding --limit 5 --enable-sandbox
```

SWE-bench mini 和 tau2 属于 heavy 项，必须显式启用：

```bash
bash benchmarks/evalscope/run_capability_suite.sh \
  --model qwen --suite all --limit 2 --include-heavy
```

`swe_bench_verified_mini` 会使用或拉取 Docker 镜像；不要在未确认本地磁盘和网络条件时直接跑全量。

## 5. LLM judge

`simple_qa`、`chinese_simpleqa` 和 Needle-in-a-Haystack 需要裁判模型。默认使用被测模型自评，适合检查链路，不适合作为严肃结论。

可指定另一个已经部署、无需保存真实凭据的本地端点：

```bash
EVALSCOPE_JUDGE_MODEL=<judge-served-name> \
EVALSCOPE_JUDGE_API_URL=http://127.0.0.1:<port>/v1 \
bash benchmarks/evalscope/run_capability_suite.sh \
  --model qwen --suite factuality --limit 20
```

脚本只使用 `EMPTY` API key，不接受或记录真实 token。需要外部付费裁判服务时，应在不保存配置和日志凭据的独立环境中运行。

## 6. 性能测试

```bash
# 3 个请求的快速验证
bash benchmarks/evalscope/run_perf_suite.sh --model qwen --suite quick

# 并发 1/4/8/16
bash benchmarks/evalscope/run_perf_suite.sh --model qwen --suite concurrency

# 输入 1K/8K/32K/64K/128K
bash benchmarks/evalscope/run_perf_suite.sh --model qwen --suite context

# 输出 128/512/2K/8K
bash benchmarks/evalscope/run_perf_suite.sh --model qwen --suite output

# 所有常规性能场景
bash benchmarks/evalscope/run_perf_suite.sh --model qwen --suite all

# 额外执行并发 16、1000 请求的稳定性测试
bash benchmarks/evalscope/run_perf_suite.sh \
  --model qwen --suite all --include-soak
```

随机数据需要 tokenizer。默认使用各模型的公开模型 ID；也可以指定本地 tokenizer 目录：

```bash
bash benchmarks/evalscope/run_perf_suite.sh \
  --model qwen \
  --tokenizer-path /path/to/local/tokenizer \
  --suite concurrency
```

## 7. 输出与恢复

能力结果：

```text
logs/eval/evalscope/<run-id>/<model>/capability/
├── commands.log
├── suite_status.tsv
└── <group>/<benchmark>/
    ├── configs/
    ├── predictions/
    ├── reports/
    ├── reviews/
    └── runner.log
```

性能结果：

```text
logs/bench/evalscope/<run-id>/<model>/perf/
├── commands.log
├── suite_status.tsv
└── <scenario>/
```

失败后可以用相同 `--run-id --resume` 继续：已有 `SUCCESS` 的 benchmark 或性能场景会跳过，`FAILED` 和 `SKIPPED` 会重新检查并执行。EvalScope 输出使用 `--no-timestamp`，路径稳定。

```bash
bash benchmarks/evalscope/run_all.sh \
  --models qwen,thinkingcap \
  --capability-suite all \
  --limit 5 \
  --perf-suite all \
  --run-id <原-run-id> \
  --resume
```

## 8. 对比原则

- 两个模型使用同一个 `run-id`、EvalScope 版本、题目 limit、seed 和 generation config。
- 能力 A/B 默认 `temperature=0`、并发 1，优先保证可复现。
- 性能 A/B 按模型顺序执行，不同时压测两个模型。
- EvalScope 和已有 `lm-eval` 的 prompt、答案抽取与汇总口径不同，分数不可直接混用。
- `SKIPPED` 不等于模型失败；它表示依赖、安全条件或外部环境未满足。

## 9. 82 项参考评测环境

`reference_benchmarks.tsv` 将用户提供的 Language 37 项和 Vision Language 45 项放在同一份清单中。这个入口的目标是如实区分“当前可调度”“协议待核对”和“需要外部 harness”，不会把缺少数据、许可证、搜索服务、GUI/Android 环境或视频工具链的任务伪装成已经可运行。

### 安装与只读检查

先执行只读检查；该命令不联网，也不下载数据：

```bash
bash benchmarks/evalscope/setup_eval_environments.sh check
```

按缺失项选择安装，避免不必要地一次拉取所有大依赖：

```bash
# Language：IFEval、IFBench、BFCL、sandbox 和 SWE-bench
bash benchmarks/evalscope/setup_eval_environments.sh install-language

# Vision：OCRBench、OmniDocBench 和 RefCOCO 评分依赖
bash benchmarks/evalscope/setup_eval_environments.sh install-vision

# WMT24++ 的 COMET 依赖，体积较大；使用独立 Python 3.10 环境，
# 避免其 torch<2 / protobuf<=3.20.1 约束污染基础 EvalScope 环境。
bash benchmarks/evalscope/setup_eval_environments.sh install-translation

# Terminal-Bench 2：独立 Python 3.12 + Harbor 环境
bash benchmarks/evalscope/setup_eval_environments.sh install-terminal
```

也可以使用 `install-all` 顺序执行以上安装。它仍不会安装清单中的外部搜索、视频、GUI、Android 或受许可数据 harness，也不会向 `models/` 写入内容。共享路径和默认参数来自 `config/evaluation.env`；大数据与外部 harness 应放在 `.eval-deps/` 下。

### 状态含义

- `fixed`：本地 runner 和关键协议已固定，默认允许调度；正式结果仍应记录数据 revision、prompt 和输出预算。
- `review`：已有 adapter，但版本、subset、judge、聚合或模型卡协议尚未完全对齐；默认显示为 `BLOCKED_REVIEW`，只有显式传入 `--include-review` 才会运行。
- `external`：没有本地 EvalScope runner，或任务依赖独立搜索、代码、视频、GUI、Android、医疗数据等 harness；始终记录为 `BLOCKED_EXTERNAL`，安装 Python 包不能解除这个状态。
- `missing` / `blocked`：协议或 adapter 本身尚未准备好，记录为 `BLOCKED_PROTOCOL`。依赖、Docker、judge 或数据目录不满足时则记录为 `BLOCKED_ENV`。

这些 `BLOCKED_*` 状态表示评测基础设施缺口，不表示模型答题失败。

### dry-run、预检与正式运行

先离线查看完整 82 项计划；`dry-run` 不检查端点、不下载数据，也不请求模型：

```bash
bash benchmarks/evalscope/run_reference_suite.sh \
  --model qwen \
  --modality all \
  --priority all \
  --dry-run
```

再做环境和端点预检，但不请求题目：

```bash
bash benchmarks/evalscope/run_reference_suite.sh \
  --model qwen \
  --modality language \
  --priority P0 \
  --preflight-only
```

推荐先对单项做两题 smoke。需要 judge 的任务必须提供独立 judge；仅验证链路时才可用 `--allow-self-judge`，自评结果不能作为正式横向结论：

```bash
bash benchmarks/evalscope/run_reference_suite.sh \
  --model qwen \
  --benchmark 'MMLU-Pro' \
  --limit 2 \
  --run-id reference-smoke
```

正式运行默认只调度 `fixed` 项，`--limit 0` 表示每个 subset 全量：

```bash
bash benchmarks/evalscope/run_reference_suite.sh \
  --model qwen \
  --modality language \
  --limit 0 \
  --run-id reference-qwen-full
```

协议核对完成后，才为对应批次加入 `--include-review`。代码、SWE-bench 和 Terminal-Bench 任务还需显式传 `--enable-sandbox`，并确保 Docker daemon、权限和相应独立环境均可用。

参考套件输出位于：

```text
logs/eval/reference-82/<run-id>/<model>/
├── status.tsv
├── commands.log
├── language/<order>-<adapter>/
└── vision/<order>-<adapter>/
```

### VLM 注意事项

- 被测 vLLM 服务必须实际启用视觉输入，并通过 OpenAI-compatible Chat Completions 接受图像内容；只有 `/v1/models` 健康并不能证明图片请求可用，正式批次前应先跑一个 `--limit 1` 的 VLM smoke。
- 当前 EvalScope 视觉 adapter 会把图片编码为请求内容；本地仍需能够下载或读取对应数据集和媒体文件。高分辨率、多图、OCR 与文档任务可能显著增加请求体、预处理时间和显存压力。
- 脚本会为 `MMMU-Pro` 选择 `vision` 数据格式，为 `MMBenchEN-DEV-v1.1` 选择 `en` subset，并为 `RefCOCO(avg)` 使用 `bbox_rec`；这些自动参数不等于已经解决模型卡 revision 和最终聚合口径。
- `ZEROBench`、`SimpleVQA` 等 `vlm_judge` 项正式评测需要独立 judge。可在运行前设置 `EVALSCOPE_JUDGE_MODEL` 和 `EVALSCOPE_JUDGE_API_URL`，不要把真实 token 写入仓库或日志。
- 本地 EvalScope 1.6.1 没有清单中视频、Visual Agent 和多数 3D/医疗任务的精确 adapter；它们会保留为 `BLOCKED_EXTERNAL`，需要逐项准备官方 harness、媒体、许可证和 scorer 后再接入。
