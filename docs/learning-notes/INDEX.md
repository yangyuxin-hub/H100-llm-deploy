# 大模型推理知识体系总目录

本目录是 `docs/learning-notes/` 下全部 12 篇笔记的总索引，按"知识体系 → 笔记章节 → 项目实战对应"三层组织，用于快速定位任何知识点。

---

## 一、知识体系全景

```text
大模型推理
├── 推理流程（01, 02）
│   ├── token化 → embedding → 多层计算 → 概率分布 → 循环生成
│   ├── Prefill（读题，算力密集） / Decode（写答案，访存密集）
│   └── OpenAI 兼容 API（/v1/models, /v1/chat/completions）
│
├── 显存管理（05, 07, 09）
│   ├── KV Cache（避免重算历史 token）
│   │   ├── 精确公式：2 × layers × seq × kv_heads × head_dim × dtype × batch
│   │   ├── GQA（减少 KV Cache 大小）
│   │   └── KV Cache 量化（fp8，显存减半）
│   ├── PagedAttention（分页管理，无碎片）
│   │   ├── block + block table + scheduler
│   │   └── Prefix Caching（前缀共享）
│   └── 权重显存（量化决定）
│       ├── FP8（1 byte/param，H100 原生）
│       └── NVFP4（0.5 byte/param，需反量化）
│
├── 计算加速（07, 08）
│   ├── H100 Tensor Core（FP8 算力 2× BF16）
│   ├── Attention Backend（FlashAttention v3、FlashInfer、Triton）
│   ├── CUDA Graph（消除 kernel launch 开销）
│   └── torch.compile/inductor（算子融合，NVFP4 有 NaN bug）
│
├── 多卡并行（06, 09）
│   ├── TP（张量并行，权重切分，每层 all-reduce）
│   ├── DP（数据并行，多副本）
│   ├── PP（流水线并行，按层切）
│   ├── EP（专家并行，MoE 用）
│   └── 通信（NVLink vs PCIe、NCCL、CUDA_VISIBLE_DEVICES）
│
├── 模型结构（01, 05, 08, 09）
│   ├── 标准 Transformer（attention + MLP）
│   ├── GQA（Grouped-Query Attention）
│   ├── GDN（Gated DeltaNet，Qwen3.6 混合架构）
│   ├── MoE（Mixture of Experts，总参数大激活参数小）
│   └── MTP（Multi-Token Prediction，投机解码）
│
├── 采样与生成（10）
│   ├── 采样参数（temperature, top-k, top-p, min-p）
│   ├── 思考模型（reasoning + content 分离）
│   ├── 结构化输出（grammar/JSON/regex 约束解码）
│   └── Function Calling（tool-call-parser）
│
├── 推理框架（02, 11）
│   ├── vLLM 架构（Scheduler / Worker / KV Cache Manager）
│   ├── 调度参数（max-num-seqs, max-num-batched-tokens, gpu-memory-utilization）
│   ├── Continuous Batching + Chunked Prefill
│   ├── Preemption（显存不足换出）
│   └── benchmark 工具与指标
│
└── 运维与稳定性（03, 04, 12）
    ├── Docker 部署（镜像、GPU 隔离、模型挂载）
    ├── 监控（/metrics, Prometheus, Grafana, dcgm-exporter）
    ├── 稳定性测试（长上下文、长时间并发、OOM 恢复）
    └── 排坑记录（torch.compile NaN、GDN bug、tool-parser 选型）
```

---

## 二、笔记详细目录

### 01-llm-inference-basics.md — 大模型推理基础

- 推理流程：文本 → token → token id → embedding → 多层计算 → 概率分布 → 循环生成
- Prefill 阶段（读题，影响 TTFT，适合并行）
- Decode 阶段（写答案，影响 tokens/s，每步依赖上一步）
- KV Cache 概念（缓存历史 token 的 K/V，减少重复计算）
- 项目请求路径：curl → vLLM API → tokenizer → Qwen3.6 → TP=2 → 生成 → detokenizer → 返回

### 02-deployment-concepts.md — 部署相关核心概念

- vLLM 职责（加载权重、多卡切分、管理 KV Cache、调度请求、执行 prefill/decode）
- 双模型并行部署（TP=2 × 2，CUDA_VISIBLE_DEVICES 隔离）
- TP=2 含义（一个模型用 2 张 GPU，`--tensor-parallel-size 2`）
- FP8（8-bit 浮点，H100 支持，省显存）
- 上下文长度（256K，KV Cache 显存代价）
- OpenAI 兼容 API（/v1/models, /v1/chat/completions, served-model-name）
- MTP / Speculative Decoding（一次预测多 token，只加速 decode）
- Function Calling（`--enable-auto-tool-choice --tool-call-parser`）
- One API 网关（统一入口、计费、路由）
- Text-only 模式（`--language-model-only` 跳过 vision encoder）

### 03-qwen3nvfp4-optimization.md — Qwen3.6 NVFP4 优化记录（历史）

- 问题 1：torch.compile/inductor 对 NVFP4+GDN 模型产生 NaN logits（输出全是 "!"）
- 问题 2：FlashInfer GDN prefill kernel 在 Hopper 上 NaN
- 二分实验定位根因（9 组实验，排除 attention backend / KV 量化 / max-len）
- 最终解法：`TORCH_COMPILE_DISABLE=1` + `FULL_DECODE_ONLY` CUDA graph
- cu129 原生镜像（无需 cuda-compat 层，启动快 36%）
- 性能数据（MTP2 28 tok/s → MTP3 32 tok/s → +CUDA graph 162 tok/s）
- 理论吞吐估算（roofline：decode 访存瓶颈，prefill 算力瓶颈）
- 实测效率：FP8 30%，NVFP4 16%（Marlin 反量化吃收益）

### 04-dual-model-parallel-deployment.md — 双模型并行部署学习笔记

- GPU 隔离：`CUDA_VISIBLE_DEVICES` + `--gpus all`
- Function Calling 踩坑：hermes ❌ / qwen3_xml ⚠️ / qwen3_coder ✅
- MTP 投机解码原理（draft 是模型自带 MTP 层，不用额外模型）
- 思考模型（Qwen3.6 reasoning 字段，关闭用 `chat_template_kwargs`）
- 官方采样参数表（思考通用/编码/非思考三档）
- One API 网关配置（渠道、token、路由）
- opencode 接入（provider、limit.context 对齐）
- 官方配置对齐检查清单（max-len / kv-dtype / parser / 采样 / max_tokens / MTP）
- 显存计算（Qwen 71.6GB/卡，Agents 75.3GB/卡）

### 05-kv-cache-pagedattention-long-context.md — KV Cache、PagedAttention 与长上下文

- KV Cache 复盘（为什么需要、显存代价）
- **精确公式**：`2 × num_layers × seq_len × num_kv_heads × head_dim × dtype_size × batch_size`
- 项目实例计算（Qwen3.6：256K 单请求 KV Cache = 32 GiB）
- GQA（MHA/MQA/GQA 对比，Qwen3.6 的 40 query head / 8 kv head）
- GQA 对 TP 的影响（kv_heads 要被 TP size 整除）
- PagedAttention 原理（block + block table + scheduler，借鉴 OS 分页）
- PagedAttention 收益（无内部碎片、无外部碎片、可共享）
- block 大小取舍（默认 16）
- Continuous Batching（动态插入/移出请求，配合 PagedAttention）
- 关键调度参数（max-num-seqs, max-num-batched-tokens, gpu-memory-utilization）
- 注意力 O(n²) 复杂度（prefill 平方增长）
- FlashAttention tiling（显存 O(n²)→O(n)，不落地完整注意力矩阵）
- FlashAttention vs PagedAttention（计算 vs 管理，vLLM 结合两者）
- decode 是访存瓶颈（每步读整个 KV Cache）
- RoPE 位置编码（旋转矩阵，相对位置）
- 长上下文外推（NTK-aware / YaRN / Dynamic NTK）
- 外推不是免费的（精度下降、显存、prefill 慢）
- Prefix Caching（相同前缀复用 block，copy-on-write）

### 06-tensor-parallelism-communication.md — 张量并行、通信与多卡部署

- 四种并行对比（DP/TP/PP/EP 各自思路与适用场景）
- 当前项目选 TP 的原因（单卡装不下 27B+256K）
- **列并行**（Column Parallel）：权重按列切，输入完整，输出部分
- **行并行**（Row Parallel）：权重按行切，输入切片，all-reduce 求和
- Transformer 层的混合切分（QKV 列并行 + O 投影行并行，每层 2 次 all-reduce）
- GQA 与 TP 切分（head 数要整除 TP size）
- All-Reduce 通信（Ring/Tree/NCCL，custom all-reduce 与 driver 兼容）
- 通信量估算（每层 2 次，64 层 128 次）
- **NVLink vs PCIe**（900 GB/s vs 64 GB/s，差 10 倍）
- 查 NVLink 拓扑（`nvidia-smi topo -m`，NV2/PIX/NODE/SYS）
- CUDA_VISIBLE_DEVICES 原理（物理 GPU → 逻辑 GPU 重新映射）
- Docker 里的 GPU 隔离（`--gpus all` + `CUDA_VISIBLE_DEVICES`）
- TP 对性能影响（降单请求延迟，不一定提升吞吐）
- NCCL 通信库（communicator、NCCL_DEBUG 调试）
- 项目盲点：未检查 H100 的 NVLink 拓扑

### 07-quantization-fp8-h100.md — 量化、FP8 与 H100 Tensor Core

- 浮点数结构（符号 + 指数 + 尾数）
- 格式对比（FP32/FP16/BF16/FP8 E4M3/FP8 E5M2/INT8/FP4）
- BF16 为什么是大模型主流（动态范围同 FP32，训练稳定）
- FP8 两种格式（E4M3 推理用，E5M2 反向用）
- 量化按对象分（Weight-only W4A16 / Weight+Activation W8A8）
- 量化按粒度分（per-tensor/per-channel/per-token/per-group）
- 对称 vs 非对称量化
- **H100 Tensor Core**（矩阵乘专用单元，FP8 算力 3958 TFLOPS = 2× BF16）
- H100 无原生 FP4 计算单元（Blackwell 才有）
- Transformer Engine（NVIDIA FP8 计算库）
- Qwen3.6-27B-FP8 量化方式（per-channel weight + per-token activation）
- FP8 显存对比（54GB BF16 → 27GB FP8）
- FP8 KV Cache（1 byte/token，显存减半，256K 可行的关键）
- FP8 精度问题（softmax 用 FP32，attention score 用 FP16）
- NVFP4（E2M1，0.5 byte，per-group scale）
- Marlin 反量化（NVFP4 → BF16 计算，H100 上的开销来源）
- NVFP4 实测 vs 理论（理论 2× FP8，实测只快 10%，反量化吃收益）
- ModelOpt 混合量化（FP8 + NVFP4 混合，校准）
- 权重量化 vs KV Cache 量化（独立，可单独开）
- fp8 KV Cache 对长上下文精度影响
- 项目显存账（fp8 KV cache 把并发数翻倍）
- 为什么最终选 FP8 而不是 NVFP4（原生计算、无 NaN、256K 可行）

### 08-attention-backend-cuda-graph.md — Attention Backend 与 CUDA Graph

- Attention Backend 概念（vLLM 的可插拔 kernel 接口）
- 选择依据（GPU 架构、精度、模型结构、序列长度、PagedAttention 支持）
- **FlashAttention**（tiling，O(n²)→O(n)，v1/v2/v3 演进，H100 用 v3）
- FlashAttention v3 的 H100 优化（TMA、Warp Group、FP8）
- **FlashInfer**（推理专用库，但 GDN 有 NaN bug）
- **Triton**（OpenAI GPU 编程语言，JIT 编译，兼容性好）
- XFormers（Meta，已过时）
- backend 选型决策树（H100 → FLASH_ATTN；特殊模型 → triton GDN）
- **GDN（Gated DeltaNet）**（Qwen3.6 混合架构的线性注意力）
- GDN 为什么容易 NaN（`exp(g)` 溢出，FlashInfer gating space bug）
- GDN 与 attention backend 的关系（单独 backend 接口）
- **CUDA Graph 原理**（录制 kernel 序列 → 一次提交，消除 launch 开销）
- kernel launch 开销（5-10 μs/kernel，decode 每步几百 kernel）
- CUDA Graph 捕获要求（固定 shape，动态 shape 难 capture）
- vLLM 的 CUDA graph 模式（NONE / FULL_DECODE_ONLY / FULL_AND_PIECEWISE）
- decode-only CUDA Graph（不依赖 torch.compile，绕 NaN bug）
- CUDA Graph 代价（显存 2.52 GiB、启动时间、KV cache 降 5.7%）
- 实测效果（32 tok/s → 162 tok/s，5 倍提升）
- **torch.compile / inductor**（JIT 编译器，算子融合）
- inductor 工作流程（PyTorch → FX Graph → Triton kernel）
- 为什么 inductor 改变数值行为（融合重排计算顺序，累加精度不同）
- NVFP4 NaN 根因（inductor 改变 Marlin/GDN kernel 数值行为）
- 解耦方案（`TORCH_COMPILE_DISABLE=1` 禁 compile + `FULL_DECODE_ONLY` 保留 graph）
- mm-encoder-attn-backend（vision encoder 的 attention，当前跳过）
- 项目三层 backend 配置（标准 attention / GDN / vision 各自配）

### 09-moe-expert-parallelism.md — MoE 与专家并行

- 稠密模型瓶颈（参数越多计算量线性增长）
- MoE 思路（N 个专家，每 token 选 K 个）
- 稠密 vs MoE 对比（总参数/激活参数/计算量/显存/能力）
- **Router 路由器**（W_router 矩阵，top-K 选择，加权求和）
- load balancing（训练时加 loss，推理时仍可能不均）
- 项目 MoE 模型（Agents-A1：256 专家/8 激活；DeepSeek-V4：256 专家/6 激活）
- MoE 显存特点（总参数常驻，计算只和激活参数有关）
- 与稠密模型对比（Agents-A1 权重大 33%，decode 计算量只有一半）
- **EP 专家并行**（不同专家整块放不同卡，all-to-all 通信）
- EP 工作流程（router 计算 → all-to-all 发送 → 各卡算 → all-to-all 回传）
- EP vs TP 对比（通信方式、适合场景）
- EP + TP 混合（大规模部署）
- 当前项目用 TP 不用 EP 的原因（vLLM EP 较新、TP=2 够用、all-to-all 复杂）
- MoE decode 优势（计算密度低，单步快）
- MoE + MTP（decode 快，MTP 加速更明显）
- MoE prefill 瓶颈（router + 调度开销）
- DeepSeek-V4 的 FP4 专家（`expert_dtype: fp4`，自研非 NVFP4）
- DSpark（DeepSeek 推测解码模块，待测试）
- MoE 工程问题（负载不均、专家交换、router 精度、KV Cache 不受影响）

### 10-sampling-structured-output.md — 采样策略与结构化输出

- 模型输出概率分布（logits → softmax → probs）
- 贪心解码（argmax，确定性但易重复）
- 随机采样（按概率选，需截断）
- **temperature**（缩放 logits，→0 贪心，>1 更随机）
- **top-k**（保留概率最高 k 个）
- **top-p**（nucleus，保留累加到 p 的最小集合，自适应）
- **min-p**（保留 ≥ min_p × max_prob 的 token）
- presence_penalty / frequency_penalty（减少重复）
- 思考模型为什么用高温度（探索性推理，试错）
- 官方采样参数表（思考通用 1.0 / 编码 0.6 / 非思考 0.7）
- 编码为什么用 0.6（语法要确定，逻辑要探索）
- 思考模型输出结构（reasoning + content 分离）
- max_tokens 要给够（思考消耗几百~几千 token，复杂任务 81920）
- 关闭思考的方法（`chat_template_kwargs: enable_thinking: false`）
- **结构化输出**（约束解码，强制格式）
- 约束解码原理（根据格式计算合法 token，屏蔽非法）
- 约束类型（JSON Schema / Regex / Grammar / Choice）
- vLLM guided_decoding（outlines / lm-format-enforcer / xgrammar）
- **Function Calling 底层**（模型输出 tool_calls 文本 → parser 解析）
- Function Calling 流程（客户端发工具列表 → 模型生成 → parser 解析 → 执行 → 回传）
- tool-call-parser 作用（不同模型格式不同，选错不解析）
- 项目踩坑（hermes ❌ / qwen3_xml ⚠️ / qwen3_coder ✅）
- reasoning-parser 作用（分离 reasoning/content）
- 采样开销（softmax O(vocab)，top-k 排序）
- 结构化输出开销（状态机维护）
- MTP 与采样交互（低温度接受率高）

### 11-vllm-internals-tuning.md — vLLM 内部架构与性能调优

- **三层架构**（Scheduler / KV Cache Manager / Worker）
- Scheduler 职责（决定每步算哪些请求，组装 batch）
- Scheduler 输入输出（waiting/running 队列 → batch + block table）
- 调度策略（优先 decode，剩余显存 prefill，不足 preempt）
- KV Cache Manager（free_blocks 池、block_table、allocate/free/can_allocate）
- Worker（每个 TP rank 一个，执行 forward + NCCL 通信）
- **max-num-seqs**（batch 最大请求数，当前 16，是吞吐瓶颈）
- **max-num-batched-tokens**（单次 forward token 上限，影响 chunked prefill）
- **gpu-memory-utilization**（显存使用比例，当前 0.90）
- 三参数相互作用（utilization → KV 空间 → 请求数 → max-num-seqs 限制）
- Prefill vs Decode 调度差异（计算密集 vs 访存密集）
- **Chunked Prefill**（长 prompt 切块 + decode 混合，降 TTFT 抖动）
- **Preemption**（显存不足换出请求到 CPU，要避免）
- 避免抢占的方法（降 max-num-seqs / max-model-len，开 fp8 kv-cache）
- Prefix Caching 实现（block 哈希比较，copy-on-write）
- 性能调优方法论（瓶颈定位 → 调优顺序）
- benchmark 工具（vllm bench serve）
- 关键指标（TTFT/TPOT/ITL/output tok/s/接受率）
- 指标间关系（TPOT ≈ 1/单流 decode，吞吐 ≈ 并发/TPOT）
- request-rate=inf 含义（饱和吞吐测试）
- 项目瓶颈分析（并发 16 未饱和，max-num-seqs 卡住）
- 推荐调优实验（max-num-seqs 32、prefix caching、chunked prefill）

### 12-monitoring-stability.md — 监控与稳定性

- 黑盒监控局限（`curl /v1/models` 只看活没活）
- 白盒监控目标（队列、显存、吞吐、preempt、延迟）
- **vLLM /metrics 端点**（Prometheus 格式，默认开启）
- 关键指标（running/waiting/swapped/gpu_cache_usage_perc/TTFT/TPOT/E2E）
- 水位指标 vs 性能指标 vs 异常指标
- 项目快速检查脚本（grep swapped / gpu_cache / running|waiting）
- **Prometheus + Grafana** 架构与配置
- 告警规则（KVCacheNearlyFull / RequestsBackingUp / PreemptionOccurred / TPOTP99High）
- Grafana 面板建议（总览/显存/延迟/请求特征）
- GPU 层监控（nvidia-smi dmon，温度/SM/显存/功耗）
- **dcgm-exporter**（NVIDIA 官方 GPU 监控导出器）
- 稳定性测试维度（长上下文/长时间并发/OOM 恢复/优雅重启/热节流）
- 长上下文压测方法（256K 输入 + 多并发）
- 长时间并发测试（12h+，看吞吐退化/显存增长）
- OOM 恢复测试（超量请求，看 preempt vs 崩溃）
- 优雅重启测试（docker stop/start，看在途请求处理）
- 常见稳定性问题（显存泄漏/热节流/请求超时堆积/容器异常退出）
- 日志管理（轮转、关键关键词）
- 项目监控现状与缺口
- 最小化监控脚本（不用 Prometheus 也能做）
- 项目待做的稳定性测试项
- 双模型并行特殊监控点（两端口/4 张卡/互不影响验证）

---

## 三、按主题交叉索引

### KV Cache 相关
- 概念引入：01（基础）、02（部署配置）
- 精确公式 + GQA：05
- 量化（fp8）：07
- PagedAttention 管理：05
- Manager 实现：11
- 监控指标：12

### 张量并行 / 多卡相关
- TP=2 用法：02、04
- TP 原理 + 切分细节：06
- EP（MoE 用）：09
- CUDA_VISIBLE_DEVICES：04、06
- 通信（NCCL/NVLink）：06
- 项目双模型并行：04

### 量化相关
- FP8 概念：02
- FP8 实测（Qwen3.6-27B-FP8）：03、04
- NVFP4 实测 + 踩坑：03
- 量化理论 + H100 Tensor Core：07
- KV Cache 量化：05、07
- ModelOpt：03、07
- Marlin 反量化：03、07

### Attention / 计算加速相关
- Prefill/Decode 概念：01
- O(n²) 复杂度 + FlashAttention：05
- Attention Backend 选型：08
- CUDA Graph + torch.compile：03、08
- GDN bug：03、08
- MTP 投机解码：02、04

### 模型结构相关
- 推理流程：01
- GQA：05
- GDN（Qwen3.6 混合架构）：03、08
- MoE：09
- 思考模型：04、10
- 多模态（vision encoder）：02、08

### 采样 / 生成相关
- 采样参数表：04
- 采样理论：10
- 结构化输出 / Function Calling：02、04、10
- tool-call-parser 踩坑：04、10
- reasoning-parser：04、10

### vLLM 框架相关
- 基础职责：02
- 架构（Scheduler/Worker/Manager）：11
- 调度参数：11
- benchmark 工具：03、11
- /metrics 监控：12
- PagedAttention / Continuous Batching：05、11
- Prefix Caching / Chunked Prefill：05、11

### 排坑记录
- torch.compile NaN（NVFP4）：03、08
- FlashInfer GDN NaN：03、08
- tool-call-parser 选型：04、10
- cuda-compat 层：03
- driver 550 兼容：03
- Mamba cache blocks 不足：03

### 性能数据
- NVFP4 各配置对比：03
- FP8 在线 benchmark：03（引用）、11
- 理论 roofline 估算：03
- TP=2 vs TP=4 取舍：06
- CUDA graph 5 倍提升：08
- 并发 16 未饱和分析：11

---

## 四、项目实战对应关系

### 当前部署架构
```text
4× H100 80GB (driver 550.144.03, CUDA 12.9)
├── GPU 0,1 → Qwen3.6-27B-FP8 (TP=2, 端口 8000)
│   ├── 权重 FP8, KV Cache fp8, 256K 上下文
│   ├── MTP qwen3_next_mtp (2 tokens)
│   ├── --language-model-only (跳过 vision)
│   └── tool-call-parser qwen3_coder
│
└── GPU 2,3 → Agents-A1-FP8 (TP=2, 端口 8001)
    ├── MoE 256专家/8激活, FP8, KV Cache fp8, 256K
    ├── MTP qwen3_next_mtp (2 tokens)
    └── tool-call-parser qwen3_coder

One API 网关 (10.30.75.58:18082) → 两个渠道
opencode → 两个 provider (qwen 8000, agents 8001)
```

### 知识点 → 笔记 → 项目配置 对应

| 项目配置 | 知识点 | 笔记编号 |
|---|---|---|
| `--tensor-parallel-size 2` | TP 切分、列/行并行 | 06 |
| `CUDA_VISIBLE_DEVICES=0,1` | GPU 隔离、重新映射 | 04, 06 |
| `--kv-cache-dtype fp8` | KV Cache 量化、显存减半 | 05, 07 |
| `--max-model-len 262144` | 长上下文、RoPE 外推、O(n²) | 05 |
| `--gpu-memory-utilization 0.90` | 显存分配、KV Cache 空间 | 11 |
| `--speculative-config qwen3_next_mtp` | MTP 投机解码 | 02, 04 |
| `--language-model-only` | 多模态、vision encoder | 02, 08 |
| `--enable-auto-tool-choice` | Function Calling | 02, 10 |
| `--tool-call-parser qwen3_coder` | 工具调用解析 | 04, 10 |
| `--reasoning-parser qwen3` | 思考模型、reasoning 分离 | 04, 10 |
| FP8 权重 | 量化、H100 Tensor Core | 07 |
| Agents-A1 MoE | MoE 路由、激活参数 | 09 |
| `max-num-seqs=16` | 调度上限、吞吐瓶颈 | 11 |
| `/metrics` | 监控、Prometheus | 12 |
| NVFP4 `TORCH_COMPILE_DISABLE=1` | torch.compile NaN、CUDA graph | 03, 08 |
| NVFP4 `--gdn-prefill-backend triton` | GDN bug、FlashInfer | 03, 08 |

---

## 五、阅读建议

### 按学习阶段

**入门（先读）**：01 → 02 → 04
- 理解推理流程、部署概念、当前项目架构

**机制深入（其次）**：05 → 06 → 07 → 08
- KV Cache、并行、量化、计算加速四大机制

**模型与生成**：09 → 10
- MoE 模型、采样策略、结构化输出

**框架与运维**：11 → 12
- vLLM 内部、调优、监控、稳定性

**历史排坑（参考）**：03
- NVFP4 项目的踩坑记录，含 roofline 分析

### 按问题查找

| 问题 | 去哪看 |
|---|---|
| 为什么 256K 上下文显存够用？ | 05（KV Cache 公式）、07（fp8 量化） |
| 为什么用 TP=2 不用 TP=4？ | 06（TP 取舍）、04（双模型并行） |
| 为什么 NVFP4 会输出 "!"？ | 03（NaN 根因）、08（torch.compile） |
| 为什么 CUDA graph 提速 5 倍？ | 08（launch 开销） |
| 为什么并发 16 未饱和？ | 11（max-num-seqs 限制） |
| 为什么 MoE 模型 decode 快？ | 09（激活参数小） |
| 为什么思考模型用高温度？ | 10（探索性推理） |
| 为什么 tool-call-parser 选 qwen3_coder？ | 04（踩坑）、10（parser 机制） |
| 怎么监控服务健康？ | 12（/metrics, Prometheus） |
| 怎么调优吞吐？ | 11（max-num-seqs, prefix caching） |

---

## 六、后续待补充

| 主题 | 内容 | 优先级 |
|---|---|---|
| MTP 投机解码深入 | 数学原理、接受率分析、EAGLE/Medusa 对比 | 中 |
| SGLang 对比 | RadixAttention、与 vLLM A/B 测试 | 中 |
| TensorRT-LLM | engine 构建、FP8/FP4、NVIDIA Dynamo | 低 |
| 生产部署 | 负载均衡、多副本、灰度发布、容量规划 | 低 |
| DeepSeek-V4 部署 | FP4 expert、DSpark、1M 上下文 | 待模型测试后 |
