# 监控与稳定性

## 一句话理解

生产部署不能只靠 `curl /v1/models` 判断健康，要用 Prometheus 抓 vLLM 的 `/metrics` 端点监控请求队列、显存使用、吞吐等指标；稳定性测试要覆盖长上下文压测、长时间并发、OOM 恢复、优雅重启等场景，确保服务在生产负载下不崩溃、不退化。

---

## 一、为什么需要监控

### 1. 黑盒监控的局限

当前项目的检查方式：

```bash
curl http://10.16.11.24:8000/v1/models   # 返回 200 就算健康
```

这只能证明"服务还活着"，不能发现：

- 请求排队但处理不动（pending 堆积）。
- KV Cache 快满了（即将 preempt）。
- 吞吐下降（GPU 降频、热节流）。
- 长尾延迟上升（调度抖动）。

### 2. 白盒监控的目标

通过 vLLM 内部指标，能回答：

- 当前有多少请求在跑？多少在等？
- KV Cache 用了多少？还剩多少？
- 每秒生成多少 token？趋势如何？
- 有没有 preempt？频率多少？
- GPU 利用率多少？显存多少？

---

## 二、vLLM 的 /metrics 端点

### 1. 默认开启

vLLM 的 OpenAI 兼容服务默认在 `/metrics` 暴露 Prometheus 格式指标：

```bash
curl http://10.16.11.24:8000/metrics
```

### 2. 关键指标

| 指标 | 含义 | 关注点 |
|---|---|---|
| `vllm:num_requests_running` | 正在 decode 的请求数 | 应 ≤ max-num-seqs |
| `vllm:num_requests_waiting` | 等待 prefill 的请求数 | 持续增长说明吞吐不足 |
| `vllm:num_requests_swapped` | 被换到 CPU 的请求数 | 应为 0，非 0 说明显存不足 |
| `vllm:gpu_cache_usage_perc` | KV Cache 使用比例 | 接近 1.0 要警惕 |
| `vllm:cpu_cache_usage_perc` | CPU swap cache 使用比例 | 应为 0 |
| `vllm:time_to_first_token_seconds` | TTFT 分布 | P99 是否在涨 |
| `vllm:time_per_output_token_seconds` | TPOT 分布 | 是否稳定 |
| `vllm:e2e_request_latency_seconds` | 端到端延迟 | 用户感知 |
| `vllm:request_inference_time_seconds` | 单请求推理时间 | - |
| `vllm:request_prompt_tokens` | 请求 prompt 长度 | 输入特征 |
| `vllm:request_generated_tokens` | 请求生成 token 数 | 输出特征 |

### 3. 指标分类

**水位指标**（看是否到上限）：

- `num_requests_running` vs `max-num-seqs`
- `gpu_cache_usage_perc` → 接近 1.0 危险
- `num_requests_waiting` → 持续增长危险

**性能指标**（看趋势）：

- `time_to_first_token_seconds` 的 P50/P99
- `time_per_output_token_seconds` 的 P50/P99
- `e2e_request_latency_seconds` 的 P50/P99

**异常指标**（应该是 0）：

- `num_requests_swapped` > 0 → preempt 发生
- `cpu_cache_usage_perc` > 0 → swap 在用

### 4. 在当前项目的应用

两个模型都有 `/metrics`：

```bash
curl http://10.16.11.24:8000/metrics  # Qwen
curl http://10.16.11.24:8001/metrics  # Agents
```

快速检查脚本：

```bash
# 检查是否有 preempt
curl -s http://10.16.11.24:8000/metrics | grep swapped

# 检查 KV Cache 使用率
curl -s http://10.16.11.24:8000/metrics | grep gpu_cache_usage

# 检查队列堆积
curl -s http://10.16.11.24:8000/metrics | grep -E "running|waiting"
```

---

## 三、Prometheus + Grafana 部署

### 1. 架构

```text
vLLM /metrics → Prometheus（抓取+存储）→ Grafana（可视化）
```

### 2. Prometheus 配置

`prometheus.yml`：

```yaml
scrape_configs:
  - job_name: 'vllm-qwen'
    static_configs:
      - targets: ['10.16.11.24:8000']
    metrics_path: /metrics
    scrape_interval: 15s

  - job_name: 'vllm-agents'
    static_configs:
      - targets: ['10.16.11.24:8001']
    metrics_path: /metrics
    scrape_interval: 15s
```

### 3. 关键告警规则

```yaml
groups:
  - name: vllm-alerts
    rules:
      # KV Cache 快满
      - alert: KVCacheNearlyFull
        expr: vllm_gpu_cache_usage_perc > 0.95
        for: 1m
        annotations:
          summary: "KV Cache 使用率 >95%"

      # 请求堆积
      - alert: RequestsBackingUp
        expr: vllm_num_requests_waiting > 10
        for: 2m
        annotations:
          summary: "等待队列 >10 持续 2 分钟"

      # 发生 preempt
      - alert: PreemptionOccurred
        expr: vllm_num_requests_swapped > 0
        for: 30s
        annotations:
          summary: "发生 preempt，显存不足"

      # TPOT 恶化
      - alert: TPOTP99High
        expr: histogram_quantile(0.99, rate(vllm_time_per_output_token_seconds_bucket[5m])) > 0.05
        for: 5m
        annotations:
          summary: "TPOT P99 >50ms 持续 5 分钟"
```

### 4. Grafana 面板建议

- **总览**：running/waiting/swapped 请求数、吞吐、TTFT/TPOT。
- **显存**：gpu_cache_usage_perc、cpu_cache_usage_perc。
- **延迟分布**：TTFT、TPOT、E2E 的 P50/P95/P99。
- **请求特征**：prompt_tokens、generated_tokens 分布。

---

## 四、GPU 层监控

### 1. nvidia-smi

```bash
# 实时监控
nvidia-smi dmon -s pucvmet -d 1

# 关键指标:
# pwr: 功耗
# temp: 温度
# sm: SM 利用率
# mem: 显存利用率
# fb: 显存占用
```

### 2. 关注点

| 指标 | 异常 | 可能原因 |
|---|---|---|
| 温度 >85°C | 热节流，性能下降 | 散热问题 |
| SM 利用率 <50% | GPU 空闲 | 调度问题、batch 太小 |
| 显存接近 100% | OOM 风险 | KV Cache 配置激进 |
| 功耗波动大 | 频繁升降频 | 负载不均 |

### 3. dcgm-exporter

NVIDIA 官方的 GPU 监控导出器，比 nvidia-smi 更适合 Prometheus：

```bash
# 部署 dcgm-exporter 容器
docker run -d --gpus all -p 9400:9400 nvcr.io/nvidia/dcgm-exporter:latest
```

暴露的指标包括：

- `DCGM_FI_DEV_GPU_UTIL`：GPU 利用率
- `DCGM_FI_DEV_MEM_COPY_UTIL`：显存带宽利用率
- `DCGM_FI_DEV_GPU_TEMP`：温度
- `DCGM_FI_DEV_POWER_USAGE`：功耗

---

## 五、稳定性测试

### 1. 测试维度

| 维度 | 目的 | 方法 |
|---|---|---|
| 长上下文压测 | 256K 能否稳定 | 满长度输入 + 输出 |
| 长时间并发 | 持续负载下是否退化 | 持续 12-24 小时并发 |
| OOM 恢复 | 显存耗尽能否恢复 | 超量请求 |
| 优雅重启 | 重启不丢数据 | docker stop/start |
| 热节流 | 高温下性能 | 持续高负载 |

### 2. 长上下文压测

```bash
# 测试 256K 上下文
vllm bench serve \
  --backend vllm \
  --model qwen3.6-27b-fp8 \
  --dataset-name random \
  --input-len 262144 \
  --output-len 512 \
  --max-concurrency 4 \
  --request-rate inf
```

关注：

- 是否 OOM。
- TTFT 是否可接受（256K prefill 很慢）。
- 能否稳定完成多个请求。

### 3. 长时间并发测试

```bash
# 持续并发 12 小时
vllm bench serve \
  --backend vllm \
  --model qwen3.6-27b-fp8 \
  --dataset-name random \
  --input-len 500 \
  --output-len 512 \
  --request-rate 5 \
  --duration 43200
```

关注：

- 吞吐是否随时间下降（内存泄漏、缓存膨胀）。
- 延迟是否恶化。
- 显存是否持续增长。
- 是否有请求失败。

### 4. OOM 恢复测试

故意发超量请求，观察：

- 是否 preempt（正常）。
- 是否 OOM 崩溃（异常）。
- 压力解除后是否恢复。

```bash
# 瞬间发 100 个并发请求（超过 max-num-seqs）
for i in $(seq 1 100); do
  curl -s http://10.16.11.24:8000/v1/completions \
    -d '{"model":"qwen3.6-27b-fp8","prompt":"test","max_tokens":100}' &
done
wait
```

正常行为：

- 部分请求排队等待。
- 可能 preempt 换出。
- 最终全部完成。
- 无崩溃。

### 5. 优雅重启

```bash
# 停止容器
docker stop qwen3.6-27b-fp8

# 检查请求是否被拒绝（不是挂起）
curl http://10.16.11.24:8000/v1/models  # 应连接失败

# 重启
bash scripts/start_qwen_fp8_docker.sh

# 验证恢复
curl http://10.16.11.24:8000/v1/models
```

关注：

- 停止时是否优雅处理在途请求（vLLM 默认会尝试完成）。
- 重启后是否能正常服务。
- 启动时间是否可接受（当前约 210s）。

---

## 六、常见稳定性问题

### 1. 显存泄漏

现象：显存占用随时间持续增长，最终 OOM。

排查：

```bash
# 定期记录显存
watch -n 60 'nvidia-smi --query-gpu=memory.used --format=csv >> /tmp/gpu_mem.log'

# 看趋势是否单调增长
```

可能原因：

- vLLM bug（KV Cache 未正确回收）。
- PyTorch CUDA 缓存不释放（通常无害，会复用）。
- 框架 buffer 累积。

### 2. 热节流

现象：长时间高负载后吞吐下降，GPU 温度高。

排查：

```bash
nvidia-smi -q -d TEMPERATURE
```

H100 工作温度上限约 80°C，超过会降频。

解决：

- 改善散热。
- 降低 gpu-memory-utilization（减少计算密度）。
- 降低 max-num-seqs。

### 3. 请求超时堆积

现象：`num_requests_waiting` 持续增长，用户感知延迟越来越大。

排查：

- 吞吐是否低于请求到达速率。
- 是否有长 prompt 请求阻塞调度。

解决：

- 调高 max-num-seqs。
- 限制单请求 max_tokens。
- 加请求超时（vLLM 支持 `--timeout-keep-alive` 等）。

### 4. 容器异常退出

现象：容器 exit，服务不可用。

排查：

```bash
docker logs <container> --tail 100
docker inspect <container> --format '{{.State.ExitCode}} {{.State.Error}}'
```

常见原因：

- OOM（显存不足）。
- CUDA error（驱动问题、硬件问题）。
- vLLM 内部 bug。

解决：

- 看 exit code 和日志定位。
- OOM 调参（降 max-num-seqs 或 max-model-len）。
- CUDA error 检查驱动和硬件。

---

## 七、日志管理

### 1. 当前项目的日志

```text
logs/
  qwen3.6-27b-fp8.docker.log
  agents-a1-fp8.docker.log
```

### 2. 日志轮转

长时间运行日志会很大，要配轮转：

```bash
# Docker 日志轮转
docker run --log-opt max-size=100m --log-opt max-file=5 ...
```

### 3. 关键日志关键词

```bash
# 查错误
docker logs <container> 2>&1 | grep -iE "error|exception|traceback"

# 查 OOM
docker logs <container> 2>&1 | grep -iE "oom|out of memory"

# 查 preempt
docker logs <container> 2>&1 | grep -i preempt

# 查 NaN
docker logs <container> 2>&1 | grep -i nan

# 查吞吐统计
docker logs <container> 2>&1 | grep "Engine 000" | tail -15
```

---

## 八、放到当前项目里看

### 1. 当前监控现状

| 维度 | 现状 | 缺口 |
|---|---|---|
| 健康检查 | `curl /v1/models` | 只能看活没活 |
| 指标暴露 | `/metrics` 可用 | 未被抓取存储 |
| 可视化 | 无 | 无 Grafana |
| 告警 | 无 | 无主动告警 |
| GPU 监控 | `nvidia-smi` 手动 | 无持续记录 |

### 2. 最小化监控方案

不部署 Prometheus/Grafana，用简单脚本也能做基础监控：

```bash
# 每 60 秒记录关键指标
while true; do
  echo "=== $(date) ===" >> /tmp/vllm_monitor.log
  curl -s http://10.16.11.24:8000/metrics | grep -E "running|waiting|swapped|gpu_cache" >> /tmp/vllm_monitor.log
  nvidia-smi --query-gpu=index,memory.used,temperature.gpu,utilization.gpu --format=csv,noheader >> /tmp/vllm_monitor.log
  sleep 60
done
```

### 3. 当前项目的稳定性测试项

对应 PROJECT_LOG.md 的"下一步任务"：

| 测试项 | 状态 | 方法 |
|---|---|---|
| 长上下文压力测试（256K） | 待做 | 256K 输入 + 多并发 |
| 长时间并发（12h+） | 待做 | 持续中等负载 |
| OOM 恢复 | 待做 | 超量请求 |
| 优雅重启 | 待做 | docker stop/start |
| 双模型互相影响 | 待做 | 两个模型同时高负载 |

### 4. 双模型并行的特殊监控点

两个模型各占 2 卡，要同时监控：

```bash
# 两个模型的指标
curl -s http://10.16.11.24:8000/metrics | grep running
curl -s http://10.16.11.24:8001/metrics | grep running

# 4 张卡的显存
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# GPU 0,1: Qwen（应各约 71GB）
# GPU 2,3: Agents（应各约 75GB）
```

关注：

- 两个模型同时高负载时，是否互相影响（不应该，因为 GPU 隔离）。
- 两个模型的显存是否稳定（不泄漏）。
- 两个模型的吞吐是否独立。

---

## 总结

| 概念 | 作用 | 项目里的体现 |
|---|---|---|
| /metrics | vLLM 暴露的 Prometheus 指标 | 两个端口都有，未被抓取 |
| num_requests_running | 当前 decode 请求数 | 应 ≤ max-num-seqs |
| num_requests_waiting | 等待队列 | 持续增长说明瓶颈 |
| num_requests_swapped | preempt 次数 | 应为 0 |
| gpu_cache_usage_perc | KV Cache 使用率 | 接近 1.0 危险 |
| Prometheus + Grafana | 监控存储+可视化 | 未部署，可扩展 |
| dcgm-exporter | GPU 层指标导出 | 未部署 |
| 稳定性测试 | 长时间/长上下文/OOM | 待做 |
| 日志轮转 | 防止日志爆盘 | 当前未配 |
