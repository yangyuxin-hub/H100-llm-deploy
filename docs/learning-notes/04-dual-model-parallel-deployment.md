# 双模型并行部署学习笔记

## 背景

2026-07-07 从单模型互斥部署切换到 Qwen3.6-27B-FP8 + Agents-A1-FP8 双模型并行部署。本文档记录这个过程中学到的关键知识点。

## 1. GPU 隔离:CUDA_VISIBLE_DEVICES

### 问题
4 张 H100,要同时跑两个模型,每个用 2 张卡。Docker 的 `--gpus all` 会把所有 GPU 都给容器,两个容器会冲突。

### 解决
用 `CUDA_VISIBLE_DEVICES` 环境变量限制容器内可见的 GPU:

```bash
# Qwen 容器:只用 GPU 0,1
docker run --gpus all --env CUDA_VISIBLE_DEVICES=0,1 ...

# Agents 容器:只用 GPU 2,3
docker run --gpus all --env CUDA_VISIBLE_DEVICES=2,3 ...
```

### 原理
- `--gpus all` 让容器能访问宿主机所有 GPU
- `CUDA_VISIBLE_DEVICES=0,1` 在容器内过滤,只暴露前两张
- 容器内 vLLM 看到的 GPU 编号会重新映射为 0,1(即使宿主机是 2,3)
- `--tensor-parallel-size 2` 对应容器内的 GPU 数

## 2. Function Calling 支持

### 问题
opencode 等 Agent 工具需要 Function Calling(让模型调用工具),vLLM 默认关闭。

### 解决
vLLM 启动加两个参数:
```bash
--enable-auto-tool-choice          # 开启工具调用
--tool-call-parser qwen3_coder     # 指定解析格式
```

### tool-call-parser 选择(踩坑)

不同模型用不同的 parser,选错了模型会输出工具调用文本但不被解析:

| Parser | 模型 | 结果 |
|---|---|---|
| `hermes` | Agents-A1-FP8 | ❌ 不解析,输出文本 |
| `qwen3_xml` | Agents-A1-FP8 | ⚠️ 能工作但非官方 |
| `qwen3_coder` | Agents-A1-FP8 | ✅ 官方推荐 |
| `qwen3_coder` | Qwen3.6-27B-FP8 | ✅ 官方推荐 |

**教训**:优先查模型官方文档推荐的 parser,不要猜。

### 查看 vLLM 支持的 parser

```bash
docker exec <container> python3 -c "
from vllm.tool_parsers import ToolParserManager
print(ToolParserManager.list_registered())
"
```

## 3. MTP 投机解码(Multi-Token Prediction)

### 原理
- 正常 decode:每次 forward 生成 1 个 token
- MTP:每次 forward 生成 N 个候选 token,然后验证,加速 decode
- 类似投机解码,但 draft model 是模型自带的 MTP 层(不用额外模型)

### 启用方法
```bash
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

### 如何判断模型支持 MTP
查模型的 `config.json`:
```json
"mtp_num_hidden_layers": 1    // 有 MTP 层就支持
```

### 注意
- MTP 只加速 decode 阶段,prefill 不加速
- MoE 模型 decode 计算密度低,MTP 加速效果更明显
- 启用后启动时间会增加(需要编译额外图)

## 4. 思考模型(Thinking Model)

### Qwen3.6 思考模式
- 默认开启思考,输出在 `reasoning` 字段,正式回答在 `content`
- 思考过程可能消耗几百~几千 token
- `max_tokens` 要给够(建议 ≥1000,复杂任务 81920)

### 关闭思考的方法(Qwen3.6)
```python
# Qwen3.6 不支持 /no_think(那是 Qwen3 的语法)
# 正确方式:
resp = client.chat.completions.create(
    model="qwen3.6-27b-fp8",
    messages=[...],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)
```

### 采样参数(官方推荐)

| 模式 | temperature | top_p | top_k | presence_penalty | 适用场景 |
|---|---|---|---|---|---|
| 思考(通用) | 1.0 | 0.95 | 20 | 0.0 | 日常对话、推理 |
| 思考(编码) | 0.6 | 0.95 | 20 | 0.0 | 代码生成(opencode) |
| 非思考 | 0.7 | 0.8 | 20 | 1.5 | 快速回答 |

## 5. One API 网关

### 作用
- 统一入口:多个模型通过一个端口访问
- 计费统计:记录每次调用的 token 数
- Token 管理:给不同应用发不同 token
- 渠道路由:根据 model 名自动转发到对应后端

### 架构
```
应用 → One API (10.30.75.58:18082) → vLLM (10.16.11.24:8000 或 8001)
```

### 配置
- 每个模型配一个「渠道」
- 渠道类型=OpenAI,Base URL 指向 vLLM 端点
- Token 在「令牌」页面创建,格式 `sk-xxx`

### 注意
- 登录密码 ≠ API token,token 需要在「令牌」页面单独创建
- Token key 在 API 响应中被掩码,只能去网页复制完整 key

## 6. opencode 接入

### 配置文件
`~/.config/opencode/opencode.json`

### 关键配置
```json
{
  "provider": {
    "<provider_id>": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://10.16.11.24:8000/v1",
        "apiKey": "any-token"
      },
      "models": {
        "<model_id>": {
          "limit": {
            "context": 262144,    // 要与 vLLM --max-model-len 一致
            "output": 81920       // 要与 vLLM max_tokens 一致
          }
        }
      }
    }
  }
}
```

### 一个 provider 只能有一个 baseURL
两个模型在不同端口,要配两个 provider(qwen、agents)。

### limit.context 的作用
- 告诉 opencode 客户端模型的能力上限
- 用于计算状态栏剩余 token
- 不影响实际推理(实际限制由 vLLM --max-model-len 决定)
- 要与服务端对齐,否则状态栏显示错误

## 7. 官方配置对齐

### 教训
部署模型前**一定要看官方文档**,不要凭经验猜参数。

### 本次对齐的配置

| 参数 | 原配置 | 官方推荐 | 影响 |
|---|---|---|---|
| Agents max-model-len | 128K | 256K | 上下文减半 |
| Agents kv-cache-dtype | 未设 | fp8 | 显存浪费 |
| Agents tool-call-parser | hermes | qwen3_coder | Function Calling 不工作 |
| Qwen max_tokens | 8192 | 81920 | 长回复被截断 |
| Qwen language-model-only | 未加 | 可选 | 显存浪费(纯文本场景) |

### 检查清单
- [ ] max-model-len 是否与模型原生支持一致
- [ ] kv-cache-dtype 是否优化(fp8)
- [ ] tool-call-parser 是否与模型匹配
- [ ] reasoning-parser 是否正确
- [ ] 采样参数是否符合官方推荐
- [ ] max_tokens 是否足够(思考模型要更大)
- [ ] 是否启用 MTP(如果模型支持)

## 8. 显存计算

### 两个模型的显存占用

| 模型 | 权重 | KV cache (fp8) | 总占用(2卡) |
|---|---|---|---|
| Qwen3.6-27B-FP8 | ~27GB(FP8) | 动态 | 71.6GB / 卡 |
| Agents-A1-FP8 | ~36GB(FP8 MoE) | 动态 | 75.3GB / 卡 |

### 优化手段
- `--kv-cache-dtype fp8`:KV cache 量化,省一半显存
- `--language-model-only`:跳过 vision encoder,省显存给 KV cache
- `--gpu-memory-utilization 0.90`:控制显存使用比例

## 总结

双模型并行部署的核心知识点:
1. **GPU 隔离**:CUDA_VISIBLE_DEVICES
2. **Function Calling**:--enable-auto-tool-choice + 正确的 parser
3. **MTP 加速**:--speculative-config(模型要支持)
4. **思考模型**:max_tokens 要给足,关闭思考用 chat_template_kwargs
5. **官方配置**:一定要看文档,不要猜
6. **opencode 接入**:limit.context 要与服务端对齐
