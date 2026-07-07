#!/usr/bin/env bash
# ============================================================================
# 停止当前运行的 vLLM 服务
# 用法:
#   bash scripts/stop.sh              # 停止所有运行中的服务
#   bash scripts/stop.sh qwen-fp8     # 仅停止 Qwen3.6-27B-FP8
#   bash scripts/stop.sh agents       # 仅停止 Agents-A1-FP8
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config/serving.env"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*" >&2; }

TARGET="${1:-all}"
STOPPED_COUNT=0

stop_by_pidfile() {
    local pid_file="$1"
    local name="$2"
    local pid
    local container_file="${RUNTIME_DIR}/${name}.container"
    local container_id

    if [[ -f "${container_file}" ]] && command -v docker &>/dev/null; then
        container_id=$(cat "${container_file}" 2>/dev/null || echo "")
        if [[ -n "${container_id}" ]] && docker inspect "${container_id}" >/dev/null 2>&1; then
            log "${name}: 停止 Docker 容器 (${container_id:0:12})..."
            docker stop "${container_id}" >/dev/null 2>&1 || true
            docker rm "${container_id}" >/dev/null 2>&1 || true
            rm -f "${container_file}" "${pid_file}"
            STOPPED_COUNT=$((STOPPED_COUNT + 1))
            return 0
        fi
        warn "${name}: Docker 容器不存在,清理容器记录"
        rm -f "${container_file}"
    fi

    pid=$(cat "${pid_file}" 2>/dev/null || echo "")
    if [[ -z "${pid}" ]]; then
        warn "${name}: PID 文件为空,删除"
        rm -f "${pid_file}"
        return 0
    fi

    if ! kill -0 "${pid}" 2>/dev/null; then
        warn "${name}: 进程 (PID=${pid}) 已不存在,清理 PID 文件"
        rm -f "${pid_file}"
        return 0
    fi

    log "${name}: 发送 SIGTERM (PID=${pid})..."
    kill "${pid}" 2>/dev/null || true

    # 等待最多 30 秒优雅退出
    for i in $(seq 1 30); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            log "${name}: 已退出 (${i}s)"
            rm -f "${pid_file}"
            STOPPED_COUNT=$((STOPPED_COUNT + 1))
            return 0
        fi
        sleep 1
    done

    # 强制 kill
    err "${name}: 优雅退出超时,发送 SIGKILL"
    kill -9 "${pid}" 2>/dev/null || true
    sleep 2
    if kill -0 "${pid}" 2>/dev/null; then
        err "${name}: SIGKILL 后进程仍存在 (PID=${pid}),可能需要手动处理"
    else
        log "${name}: 已强制退出"
        STOPPED_COUNT=$((STOPPED_COUNT + 1))
    fi
    rm -f "${pid_file}"
}

# 遍历 PID 文件
FOUND_ANY=false
for pid_file in "${RUNTIME_DIR}"/*.pid; do
    [[ -f "${pid_file}" ]] || continue
    name=$(basename "${pid_file}" .pid)

    # 跳过下载进程的 PID(除非显式指定)
    if [[ "${name}" == "download" ]]; then
        continue
    fi

    # 按目标过滤
    if [[ "${TARGET}" != "all" ]]; then
        case "${TARGET}" in
            qwen-fp8|qwen_fp8) [[ "${name}" == "${QWEN_FP8_SERVED_NAME}" ]] || continue ;;
            agents|a1)         [[ "${name}" == "${AGENTS_SERVED_NAME}" ]] || continue ;;
            *)                 [[ "${name}" == "${TARGET}" ]] || continue ;;
        esac
    fi

    FOUND_ANY=true
    stop_by_pidfile "${pid_file}" "${name}"
done

if [[ "${FOUND_ANY}" == "false" ]]; then
    log "没有找到运行中的 vLLM 服务"
fi

if [[ ${STOPPED_COUNT} -gt 0 ]]; then
    log "已停止 ${STOPPED_COUNT} 个服务"
    # 提示 GPU 状态
    if command -v nvidia-smi &>/dev/null; then
        log "当前 GPU 显存占用(确认已释放):"
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/    /' || true
    fi
fi
