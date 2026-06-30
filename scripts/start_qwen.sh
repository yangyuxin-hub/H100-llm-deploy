#!/usr/bin/env bash
# ============================================================================
# 启动 Qwen3.6-27B-FP8 vLLM 服务
# 互斥部署:启动前会检查是否已有其他模型在运行
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config/serving.env"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*" >&2; }

MODEL_NAME="${QWEN_SERVED_NAME}"
PID_FILE="${RUNTIME_DIR}/${MODEL_NAME}.pid"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}.log"

mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}"

# 1. 检查 vllm 是否安装
if ! command -v vllm &>/dev/null; then
    err "vllm 未安装。请先执行: uv pip install vllm --torch-backend=auto (需要 vllm>=0.19.0)"
    exit 1
fi

# 2. 检查本模型是否已在运行
if [[ -f "${PID_FILE}" ]]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        err "${MODEL_NAME} 已在运行 (PID=${PID})。如需重启请先执行: bash scripts/stop.sh"
        exit 1
    else
        warn "发现残留 PID 文件但进程已死,清理中"
        rm -f "${PID_FILE}"
    fi
fi

# 3. 互斥检查:是否有其他模型在运行
for other_pid_file in "${RUNTIME_DIR}"/*.pid; do
    [[ -f "${other_pid_file}" ]] || continue
    other_name=$(basename "${other_pid_file}" .pid)
    [[ "${other_name}" == "download" ]] && continue   # 忽略下载进程
    [[ "${other_name}" == "${MODEL_NAME}" ]] && continue
    other_pid=$(cat "${other_pid_file}" 2>/dev/null || echo "")
    if [[ -n "${other_pid}" ]] && kill -0 "${other_pid}" 2>/dev/null; then
        err "互斥部署:检测到 ${other_name} 正在运行 (PID=${other_pid})。请先执行: bash scripts/stop.sh"
        exit 1
    fi
done

# 4. 检查模型权重
if [[ ! -d "${MODEL_PATH_QWEN}" ]]; then
    err "模型目录不存在: ${MODEL_PATH_QWEN}"
    err "请先执行: bash scripts/download_models.sh qwen"
    exit 1
fi
if [[ ! -f "${MODEL_PATH_QWEN}/config.json" ]]; then
    err "config.json 不存在,模型可能未下载完成: ${MODEL_PATH_QWEN}"
    exit 1
fi
# 支持 layers-N.safetensors / model-*.safetensors 两种命名
shard_count=$(ls "${MODEL_PATH_QWEN}"/layers-*.safetensors "${MODEL_PATH_QWEN}"/model-*.safetensors 2>/dev/null | wc -l)
if [[ "${shard_count}" -eq 0 ]]; then
    shard_count=$(ls "${MODEL_PATH_QWEN}"/*.safetensors 2>/dev/null | wc -l)
fi
if [[ "${shard_count}" -eq 0 ]]; then
    err "未找到 safetensors 权重文件,模型可能未下载完成"
    exit 1
fi
# 校验总大小是否达到最小阈值(GB)
total_size_kb=$(du -sk "${MODEL_PATH_QWEN}" 2>/dev/null | awk '{print $1}')
total_size_gb=$((total_size_kb / 1024 / 1024))
if [[ ${total_size_gb} -lt ${QWEN_MIN_SIZE_GB} ]]; then
    err "模型目录总大小 ${total_size_gb}GB < 预期最小 ${QWEN_MIN_SIZE_GB}GB,模型可能未下载完成"
    err "请执行: bash scripts/download_models.sh qwen (支持断点续传)"
    exit 1
fi
log "模型权重: ${MODEL_PATH_QWEN} (${shard_count} 个 safetensors, ${total_size_gb}GB)"

# 5. 检查 GPU 数量
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
REQUIRED_GPU_COUNT=$((TENSOR_PARALLEL_SIZE * ${DATA_PARALLEL_SIZE:-1}))
if [[ "${GPU_COUNT}" -lt "${REQUIRED_GPU_COUNT}" ]]; then
    err "GPU 数量不足:需要 ${REQUIRED_GPU_COUNT} (TP=${TENSOR_PARALLEL_SIZE},DP=${DATA_PARALLEL_SIZE:-1}),实际 ${GPU_COUNT}"
    exit 1
fi
log "检测到 ${GPU_COUNT} 张 GPU,使用 TP=${TENSOR_PARALLEL_SIZE},DP=${DATA_PARALLEL_SIZE:-1}"

# 6. 启动 vLLM
log "启动 vLLM 服务: ${MODEL_NAME}"
log "  端口: ${SERVE_PORT}"
log "  最大上下文: ${QWEN_MAX_MODEL_LEN}"
log "  TP: ${TENSOR_PARALLEL_SIZE}, DP: ${DATA_PARALLEL_SIZE:-1}"
if [[ "${QWEN_ENABLE_MTP:-0}" == "1" ]]; then
    log "  MTP: enabled (${QWEN_SPECULATIVE_CONFIG})"
else
    log "  MTP: disabled"
fi
log "  日志: ${LOG_FILE}"

VLLM_ARGS=(
    serve "${MODEL_PATH_QWEN}"
    --served-model-name "${MODEL_NAME}"
    --port "${SERVE_PORT}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --max-model-len "${QWEN_MAX_MODEL_LEN}"
    --gpu-memory-utilization "${QWEN_GPU_MEM_UTIL}"
    --reasoning-parser "${QWEN_REASONING_PARSER}"
    --enable-auto-tool-choice
    --tool-call-parser "${QWEN_TOOL_PARSER}"
    --trust-remote-code
)

if [[ "${DATA_PARALLEL_SIZE:-1}" != "1" ]]; then
    VLLM_ARGS+=(--data-parallel-size "${DATA_PARALLEL_SIZE}")
fi

if [[ "${QWEN_ENABLE_MTP:-0}" == "1" ]]; then
    VLLM_ARGS+=(--speculative-config "${QWEN_SPECULATIVE_CONFIG}")
fi

nohup vllm "${VLLM_ARGS[@]}" > "${LOG_FILE}" 2>&1 &

VLLM_PID=$!
echo "${VLLM_PID}" > "${PID_FILE}"
log "vLLM 进程已启动 (PID=${VLLM_PID})"

# 7. 健康检查轮询
log "等待服务就绪 (最长 ${HEALTH_CHECK_TIMEOUT} 秒)..."
ELAPSED=0
while [[ ${ELAPSED} -lt ${HEALTH_CHECK_TIMEOUT} ]]; do
    # 检查进程是否还活着
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        err "vLLM 进程已退出,请查看日志: ${LOG_FILE}"
        tail -30 "${LOG_FILE}" 2>/dev/null || true
        rm -f "${PID_FILE}"
        exit 1
    fi

    # 尝试健康检查
    if curl -sf "${HEALTH_CHECK_URL}" >/dev/null 2>&1; then
        log "服务已就绪!健康检查通过 (${ELAPSED} 秒)"
        log "API 端点: ${HEALTH_CHECK_URL}"
        log "查看日志: tail -f ${LOG_FILE}"
        exit 0
    fi

    # 显示进度
    if [[ $((ELAPSED % 30)) -eq 0 ]] && [[ ${ELAPSED} -gt 0 ]]; then
        log "仍在启动中... (${ELAPSED}s 已过)"
        tail -1 "${LOG_FILE}" 2>/dev/null | sed 's/^/    | /' || true
    fi

    sleep "${HEALTH_CHECK_INTERVAL}"
    ELAPSED=$((ELAPSED + HEALTH_CHECK_INTERVAL))
done

err "服务启动超时 (${HEALTH_CHECK_TIMEOUT} 秒)。请查看日志: ${LOG_FILE}"
exit 1
