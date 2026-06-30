#!/usr/bin/env bash
# ============================================================================
# 启动 DeepSeek-V4-Flash-DSpark vLLM 服务
# 互斥部署:启动前会检查是否已有其他模型在运行
# DSpark 投机采样通过 config.json 自动启用,无需 --speculative-config
# expert_dtype:"fp4" 字段通过 --hf-overrides 移除,避免干扰 vLLM
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config/serving.env"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*" >&2; }

MODEL_NAME="${DS_SERVED_NAME}"
PID_FILE="${RUNTIME_DIR}/${MODEL_NAME}.pid"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}.log"

mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}"

# 1. 检查 vllm
if ! command -v vllm &>/dev/null; then
    err "vllm 未安装。请先执行: uv pip install vllm --torch-backend=auto (需要 vllm>=0.19.0)"
    exit 1
fi

# 2. 检查本模型是否在运行
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

# 3. 互斥检查
for other_pid_file in "${RUNTIME_DIR}"/*.pid; do
    [[ -f "${other_pid_file}" ]] || continue
    other_name=$(basename "${other_pid_file}" .pid)
    [[ "${other_name}" == "download" ]] && continue
    [[ "${other_name}" == "${MODEL_NAME}" ]] && continue
    other_pid=$(cat "${other_pid_file}" 2>/dev/null || echo "")
    if [[ -n "${other_pid}" ]] && kill -0 "${other_pid}" 2>/dev/null; then
        err "互斥部署:检测到 ${other_name} 正在运行 (PID=${other_pid})。请先执行: bash scripts/stop.sh"
        exit 1
    fi
done

# 4. 检查模型权重
if [[ ! -d "${MODEL_PATH_DS}" ]]; then
    err "模型目录不存在: ${MODEL_PATH_DS}"
    err "请先执行: bash scripts/download_models.sh deepseek"
    exit 1
fi
if [[ ! -f "${MODEL_PATH_DS}/config.json" ]]; then
    err "config.json 不存在,模型可能未下载完成: ${MODEL_PATH_DS}"
    exit 1
fi
shard_count=$(ls "${MODEL_PATH_DS}"/model-*.safetensors "${MODEL_PATH_DS}"/layers-*.safetensors 2>/dev/null | wc -l)
if [[ "${shard_count}" -lt "${DS_SHARDS_EXPECTED}" ]]; then
    err "safetensors 文件数 ${shard_count} 少于预期 ${DS_SHARDS_EXPECTED},模型可能未下载完成"
    err "请执行: bash scripts/download_models.sh deepseek (支持断点续传)"
    exit 1
fi
log "模型权重: ${MODEL_PATH_DS} (${shard_count}/${DS_SHARDS_EXPECTED} 个 safetensors)"

# 5. 检查 GPU
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "${GPU_COUNT}" -lt "${TENSOR_PARALLEL_SIZE}" ]]; then
    err "GPU 数量不足:需要 ${TENSOR_PARALLEL_SIZE},实际 ${GPU_COUNT}"
    exit 1
fi
log "检测到 ${GPU_COUNT} 张 GPU,使用 TP=${TENSOR_PARALLEL_SIZE}"

# 6. 启动 vLLM
log "启动 vLLM 服务: ${MODEL_NAME}"
log "  端口: ${SERVE_PORT}"
log "  最大上下文: ${DS_MAX_MODEL_LEN} (默认保守值,需要 1M 改 config/serving.env)"
log "  HF overrides: ${DS_HF_OVERRIDES}"
log "  DSpark: 通过 config.json 自动启用"
log "  日志: ${LOG_FILE}"

nohup vllm serve "${MODEL_PATH_DS}" \
    --served-model-name "${MODEL_NAME}" \
    --port "${SERVE_PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${DS_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${DS_GPU_MEM_UTIL}" \
    --hf-overrides "${DS_HF_OVERRIDES}" \
    --trust-remote-code \
    > "${LOG_FILE}" 2>&1 &

VLLM_PID=$!
echo "${VLLM_PID}" > "${PID_FILE}"
log "vLLM 进程已启动 (PID=${VLLM_PID})"

# 7. 健康检查
log "等待服务就绪 (最长 ${HEALTH_CHECK_TIMEOUT} 秒)..."
ELAPSED=0
while [[ ${ELAPSED} -lt ${HEALTH_CHECK_TIMEOUT} ]]; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        err "vLLM 进程已退出,请查看日志: ${LOG_FILE}"
        tail -30 "${LOG_FILE}" 2>/dev/null || true
        rm -f "${PID_FILE}"
        exit 1
    fi

    if curl -sf "${HEALTH_CHECK_URL}" >/dev/null 2>&1; then
        log "服务已就绪!健康检查通过 (${ELAPSED} 秒)"
        log "API 端点: ${HEALTH_CHECK_URL}"
        log "查看日志: tail -f ${LOG_FILE}"
        log "提示: 推理时建议 temperature=1.0, top_p=1.0 (README 推荐)"
        exit 0
    fi

    if [[ $((ELAPSED % 30)) -eq 0 ]] && [[ ${ELAPSED} -gt 0 ]]; then
        log "仍在启动中... (${ELAPSED}s 已过)"
        tail -1 "${LOG_FILE}" 2>/dev/null | sed 's/^/    | /' || true
    fi

    sleep "${HEALTH_CHECK_INTERVAL}"
    ELAPSED=$((ELAPSED + HEALTH_CHECK_INTERVAL))
done

err "服务启动超时 (${HEALTH_CHECK_TIMEOUT} 秒)。请查看日志: ${LOG_FILE}"
exit 1
