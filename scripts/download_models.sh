#!/usr/bin/env bash
# ============================================================================
# 下载 Qwen3.6-27B-FP8 和 DeepSeek-V4-Flash-DSpark 模型权重
# 数据源: ModelScope (国内速度快)
# 用法:
#   bash scripts/download_models.sh           # 下载两个模型
#   bash scripts/download_models.sh qwen     # 仅下载 Qwen
#   bash scripts/download_models.sh deepseek # 仅下载 DeepSeek
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
        log "=== 将下载两个模型(Qwen ~28GB + DeepSeek ~154GB) ==="
        download_model "${QWEN_MODELSCOPE_ID}" "${MODEL_PATH_QWEN}" "Qwen3.6-27B-FP8"
        echo ""
        download_model "${DS_MODELSCOPE_ID}" "${MODEL_PATH_DS}" "DeepSeek-V4-Flash-DSpark"
        ;;
    qwen)
        download_model "${QWEN_MODELSCOPE_ID}" "${MODEL_PATH_QWEN}" "Qwen3.6-27B-FP8"
        ;;
    deepseek|ds)
        download_model "${DS_MODELSCOPE_ID}" "${MODEL_PATH_DS}" "DeepSeek-V4-Flash-DSpark"
        ;;
    *)
        err "未知参数: ${TARGET}。可选: all | qwen | deepseek"
        exit 1
        ;;
esac

# 下载完整性校验
echo ""
log "=== 下载完整性校验 ==="
verify_shards() {
    local path="$1" expected="$2" name="$3"
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
    log "${name}: safetensors 文件数 = ${actual} (预期 ≥ ${expected}), config.json = ${config_ok}, 总大小 = ${size}"
}

verify_shards "${MODEL_PATH_QWEN}" 1 "Qwen3.6-27B-FP8"
verify_shards "${MODEL_PATH_DS}" "${DS_SHARDS_EXPECTED}" "DeepSeek-V4-Flash-DSpark"

echo ""
log "全部完成。可执行 scripts/start_qwen.sh 或 scripts/start_deepseek.sh 启动服务"
