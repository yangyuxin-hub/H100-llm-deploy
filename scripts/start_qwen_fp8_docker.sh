#!/usr/bin/env bash
# ============================================================================
# 使用 vLLM 官方 Docker 镜像启动 Qwen3.6-27B-FP8 服务
# 双模型并行模式: TP=2, GPU 0,1, 端口 8000
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config/serving.env"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*" >&2; }

MODEL_NAME="${QWEN_FP8_SERVED_NAME}"
CONTAINER_NAME="${QWEN_FP8_DOCKER_CONTAINER_NAME}"
SERVE_PORT="${QWEN_FP8_SERVE_PORT}"
PID_FILE="${RUNTIME_DIR}/${MODEL_NAME}.pid"
CONTAINER_FILE="${RUNTIME_DIR}/${MODEL_NAME}.container"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}.docker.log"
HEALTH_URL="http://127.0.0.1:${SERVE_PORT}/v1/models"

mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}"

if ! command -v docker &>/dev/null; then
    err "docker 不可用。请先安装 Docker 和 nvidia-container-toolkit"
    exit 1
fi

if [[ ! -f "${QWEN_FP8_MODEL_PATH}/config.json" ]]; then
    err "config.json 不存在,模型可能未下载完成: ${QWEN_FP8_MODEL_PATH}"
    exit 1
fi

if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    container_state=$(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "unknown")
    if [[ "${container_state}" == "running" ]]; then
        err "Docker 容器已在运行: ${CONTAINER_NAME}。如需重启请先执行: bash scripts/stop.sh ${MODEL_NAME}"
        exit 1
    fi
    warn "清理已退出的 Docker 容器: ${CONTAINER_NAME}"
    docker rm "${CONTAINER_NAME}" >/dev/null
fi

# 缓存目录
VLLM_CACHE_DIR="${MODEL_ROOT}/vllm_cache/vllm"
TRITON_CACHE_DIR="${MODEL_ROOT}/vllm_cache/triton"
mkdir -p "${VLLM_CACHE_DIR}" "${TRITON_CACHE_DIR}"

log "使用 vLLM 官方 Docker 镜像启动: ${VLLM_DOCKER_IMAGE}"
log "  容器名: ${CONTAINER_NAME}"
log "  模型: ${QWEN_FP8_MODEL_PATH}"
log "  端口: ${SERVE_PORT}"
log "  GPU: ${QWEN_FP8_CUDA_VISIBLE_DEVICES} (TP=${QWEN_FP8_TENSOR_PARALLEL_SIZE})"
log "  最大上下文: ${QWEN_FP8_MAX_MODEL_LEN}"
log "  日志: docker logs -f ${CONTAINER_NAME}"

VLLM_ARGS=(
    --model "${QWEN_FP8_MODEL_PATH}"
    --served-model-name "${MODEL_NAME}"
    --host 0.0.0.0
    --port "${SERVE_PORT}"
    --tensor-parallel-size "${QWEN_FP8_TENSOR_PARALLEL_SIZE}"
    --max-model-len "${QWEN_FP8_MAX_MODEL_LEN}"
    --gpu-memory-utilization "${QWEN_FP8_GPU_MEM_UTIL}"
    --max-num-seqs "${QWEN_FP8_MAX_NUM_SEQS}"
    --max-num-batched-tokens "${QWEN_FP8_MAX_NUM_BATCHED_TOKENS}"
    --enable-chunked-prefill
    --async-scheduling
    --enable-prefix-caching
    --trust-remote-code
    --reasoning-parser "${QWEN_FP8_REASONING_PARSER}"
    --enable-auto-tool-choice
    --tool-call-parser "${QWEN_FP8_TOOL_PARSER}"
    --kv-cache-dtype "${QWEN_FP8_KV_CACHE_DTYPE}"
    --attention-backend "${QWEN_FP8_ATTENTION_BACKEND}"
    --speculative-config "${QWEN_FP8_SPECULATIVE_CONFIG}"
    --override-generation-config "${QWEN_FP8_OVERRIDE_GENERATION_CONFIG}"
)

# 跳过 vision encoder,释放显存给 KV cache(纯文本场景)
if [[ "${QWEN_FP8_LANGUAGE_MODEL_ONLY:-0}" == "1" ]]; then
    VLLM_ARGS+=(--language-model-only)
fi

if [[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]]; then
    VLLM_ARGS+=(--disable-custom-all-reduce)
fi

CONTAINER_ID=$(docker run -d \
    --name "${CONTAINER_NAME}" \
    --runtime nvidia \
    --gpus all \
    --ipc=host \
    -p "${SERVE_PORT}:${SERVE_PORT}" \
    -v "${MODEL_ROOT}:${MODEL_ROOT}:ro" \
    -v "${VLLM_CACHE_DIR}:/root/.cache/vllm:rw" \
    -v "${TRITON_CACHE_DIR}:/root/.cache/triton:rw" \
    --env "CUDA_VISIBLE_DEVICES=${QWEN_FP8_CUDA_VISIBLE_DEVICES}" \
    --env "VLLM_ENABLE_CUDA_COMPATIBILITY=${VLLM_DOCKER_ENABLE_CUDA_COMPATIBILITY}" \
    --env "VLLM_DEEP_GEMM_WARMUP=${VLLM_DEEP_GEMM_WARMUP:-skip}" \
    --env "VLLM_USE_FLASHINFER_SAMPLER=1" \
    "${VLLM_DOCKER_IMAGE}" \
    "${VLLM_ARGS[@]}")

echo "${CONTAINER_ID}" > "${CONTAINER_FILE}"
CONTAINER_PID=$(docker inspect -f '{{.State.Pid}}' "${CONTAINER_ID}")
echo "${CONTAINER_PID}" > "${PID_FILE}"
{
    echo "container=${CONTAINER_NAME}"
    echo "container_id=${CONTAINER_ID}"
    echo "image=${VLLM_DOCKER_IMAGE}"
    echo "model=${QWEN_FP8_MODEL_PATH}"
    echo "gpu=${QWEN_FP8_CUDA_VISIBLE_DEVICES}"
    echo "port=${SERVE_PORT}"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "logs=docker logs -f ${CONTAINER_NAME}"
} > "${LOG_FILE}"
log "Docker 容器已启动 (container=${CONTAINER_NAME}, PID=${CONTAINER_PID})"

log "等待服务就绪 (最长 ${HEALTH_CHECK_TIMEOUT} 秒)..."
ELAPSED=0
while [[ ${ELAPSED} -lt ${HEALTH_CHECK_TIMEOUT} ]]; do
    container_state=$(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "missing")
    if [[ "${container_state}" != "running" ]]; then
        err "Docker 容器已退出或不存在: ${CONTAINER_NAME}"
        docker logs --tail 80 "${CONTAINER_NAME}" 2>/dev/null || true
        rm -f "${PID_FILE}" "${CONTAINER_FILE}"
        exit 1
    fi

    if curl -sf "${HEALTH_URL}" >/dev/null 2>&1; then
        log "服务已就绪!健康检查通过 (${ELAPSED} 秒)"
        log "API 端点: ${HEALTH_URL}"
        log "查看日志: docker logs -f ${CONTAINER_NAME}"
        exit 0
    fi

    if [[ $((ELAPSED % 30)) -eq 0 ]] && [[ ${ELAPSED} -gt 0 ]]; then
        log "仍在启动中... (${ELAPSED}s 已过)"
        docker logs --tail 1 "${CONTAINER_NAME}" 2>/dev/null | sed 's/^/    | /' || true
    fi

    sleep "${HEALTH_CHECK_INTERVAL}"
    ELAPSED=$((ELAPSED + HEALTH_CHECK_INTERVAL))
done

err "服务启动超时 (${HEALTH_CHECK_TIMEOUT} 秒)。请查看: docker logs -f ${CONTAINER_NAME}"
exit 1
