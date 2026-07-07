#!/usr/bin/env bash
# ============================================================================
# 下载 Qwen3.6-27B-FP8 和 Agents-A1-FP8 模型权重
# 数据源: ModelScope (国内速度快)
# 用法:
#   bash scripts/download_models.sh           # 下载两个模型
#   bash scripts/download_models.sh qwen      # 仅下载 Qwen3.6-27B-FP8
#   bash scripts/download_models.sh agents    # 仅下载 Agents-A1-FP8
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config/serving.env"

# 颜色输出
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
err() { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*" >&2; }

# 检查 modelscope CLI
if ! command -v modelscope &>/dev/null; then
    err "modelscope CLI 未安装。请先执行: pip install -U modelscope"
    exit 1
fi

mkdir -p "${MODEL_ROOT}"

# 下载函数
# 参数: $1=modelscope_id  $2=local_path  $3=model_name
download_model() {
    local ms_id="$1"
    local local_path="$2"
    local name="$3"

    log "开始下载 ${name}"
    log "  ModelScope ID : ${ms_id}"
    log "  本地路径     : ${local_path}"

    if [[ -d "${local_path}" && $(ls -A "${local_path}" 2>/dev/null | wc -l) -gt 0 ]]; then
        log "  目录已存在且非空,modelscope 会自动断点续传"
    fi

    # modelscope download 支持断点续传
    modelscope download \
        --model "${ms_id}" \
        --local_dir "${local_path}"

    log "${name} 下载完成: ${local_path}"
}

# 根据参数选择下载目标
TARGET="${1:-all}"
case "${TARGET}" in
    all)
        log "=== 将下载两个模型(Qwen3.6-27B-FP8 + Agents-A1-FP8) ==="
        download_model "${QWEN_FP8_MODELSCOPE_ID}" "${QWEN_FP8_MODEL_PATH}" "Qwen3.6-27B-FP8"
        echo ""
        download_model "${AGENTS_MODELSCOPE_ID}" "${AGENTS_MODEL_PATH}" "Agents-A1-FP8"
        ;;
    qwen|qwen-fp8|qwen_fp8)
        download_model "${QWEN_FP8_MODELSCOPE_ID}" "${QWEN_FP8_MODEL_PATH}" "Qwen3.6-27B-FP8"
        ;;
    agents|a1)
        download_model "${AGENTS_MODELSCOPE_ID}" "${AGENTS_MODEL_PATH}" "Agents-A1-FP8"
        ;;
    *)
        err "未知参数: ${TARGET}。可选: all | qwen | agents"
        exit 1
        ;;
esac

# 下载完整性校验
echo ""
log "=== 下载完整性校验 ==="
verify_weights() {
    local path="$1" name="$2"
    if [[ ! -d "${path}" ]]; then
        err "${name} 目录不存在: ${path}"
        return 1
    fi
    # 同时支持 model-*.safetensors / layers-*.safetensors / 单文件 *.safetensors
    local actual
    actual=$(ls "${path}"/model-*.safetensors "${path}"/layers-*.safetensors 2>/dev/null | wc -l)
    if [[ "${actual}" -eq 0 ]]; then
        actual=$(ls "${path}"/*.safetensors 2>/dev/null | wc -l)
    fi
    local config_ok="NO"
    [[ -f "${path}/config.json" ]] && config_ok="YES"
    local size
    size=$(du -sh "${path}" 2>/dev/null | awk '{print $1}')
    log "${name}: safetensors 文件数 = ${actual}, config.json = ${config_ok}, 总大小 = ${size}"
}

verify_weights "${QWEN_FP8_MODEL_PATH}" "Qwen3.6-27B-FP8"
verify_weights "${AGENTS_MODEL_PATH}"   "Agents-A1-FP8"

echo ""
log "全部完成。可执行 scripts/start_qwen_fp8_docker.sh 或 scripts/start_agents_docker.sh 启动服务"
