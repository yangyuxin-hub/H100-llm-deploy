#!/usr/bin/env bash
# ============================================================================
# 查看当前部署状态:运行中的服务、GPU 占用、健康检查
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config/serving.env"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

echo -e "${BLUE}==================== LLM 部署状态 ====================${NC}"
echo -e "${BLUE}时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo ""

# ---------- 1. 检查运行中的服务 ----------
echo -e "${BLUE}【1】运行中的服务${NC}"
RUNNING_COUNT=0
for pid_file in "${RUNTIME_DIR}"/*.pid; do
    [[ -f "${pid_file}" ]] || continue
    name=$(basename "${pid_file}" .pid)
    pid=$(cat "${pid_file}" 2>/dev/null || echo "")

    if [[ "${name}" == "download" ]]; then
        # 下载进程单独显示
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            echo -e "  ${YELLOW}↓ 下载任务${NC} (PID=${pid}) - 运行中"
        else
            rm -f "${pid_file}"
        fi
        continue
    fi

    if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
        echo -e "  ${RED}✗ ${name}${NC} (PID=${pid:-N/A}) - 进程已死,清理 PID 文件"
        rm -f "${pid_file}"
        continue
    fi

    # 健康检查:根据 served-name 选择对应端点
    case "${name}" in
        "${QWEN_FP8_SERVED_NAME}") HEALTH_URL="${HEALTH_CHECK_URL_QWEN_FP8}" ;;
        "${AGENTS_SERVED_NAME}")   HEALTH_URL="${HEALTH_CHECK_URL_AGENTS}"   ;;
        *)                         HEALTH_URL="http://127.0.0.1:8000/v1/models" ;;
    esac

    HEALTH="未知"
    if curl -sf "${HEALTH_URL}" >/dev/null 2>&1; then
        HEALTH=$(echo -e "${GREEN}就绪${NC}")
    else
        HEALTH=$(echo -e "${YELLOW}启动中${NC}")
    fi

    # 进程信息
    MEM=$(ps -o rss= -p "${pid}" 2>/dev/null | awk '{printf "%.1f GB", $1/1024/1024}')
    ELAPSED=$(ps -o etime= -p "${pid}" 2>/dev/null | tr -d ' ')

    echo -e "  ${GREEN}✓ ${name}${NC} (PID=${pid}, 内存=${MEM}, 运行时长=${ELAPSED})"
    container_file="${RUNTIME_DIR}/${name}.container"
    if [[ -f "${container_file}" ]] && command -v docker &>/dev/null; then
        container_id=$(cat "${container_file}" 2>/dev/null || echo "")
        container_name=$(docker inspect -f '{{.Name}}' "${container_id}" 2>/dev/null | sed 's#^/##' || echo "")
        container_state=$(docker inspect -f '{{.State.Status}}' "${container_id}" 2>/dev/null || echo "unknown")
        if [[ -n "${container_name}" ]]; then
            echo -e "    健康: ${HEALTH}  Docker: ${container_name} (${container_state})"
            echo -e "    日志: docker logs -f ${container_name}"
        else
            echo -e "    健康: ${HEALTH}  日志: ${LOG_DIR}/${name}.docker.log"
        fi
    else
        echo -e "    健康: ${HEALTH}  日志: ${LOG_DIR}/${name}.docker.log"
    fi
    RUNNING_COUNT=$((RUNNING_COUNT + 1))
done

if [[ ${RUNNING_COUNT} -eq 0 ]]; then
    echo -e "  ${YELLOW}(无运行中的服务)${NC}"
    QWEN_FP8_OK="✗"; AGENTS_OK="✗"
    [[ -f "${QWEN_FP8_MODEL_PATH}/config.json" ]] && QWEN_FP8_OK="✓"
    [[ -f "${AGENTS_MODEL_PATH}/config.json" ]] && AGENTS_OK="✓"
    if command -v docker &>/dev/null; then
        echo -e "  可启动: bash scripts/start_qwen_fp8_docker.sh  [权重 ${QWEN_FP8_OK}]"
        echo -e "  可启动: bash scripts/start_agents_docker.sh   [权重 ${AGENTS_OK}]"
    fi
fi
echo ""

# ---------- 2. GPU 状态 ----------
echo -e "${BLUE}【2】GPU 状态${NC}"
if ! command -v nvidia-smi &>/dev/null; then
    echo -e "  ${YELLOW}nvidia-smi 不可用${NC}"
else
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    echo -e "  GPU 数量: ${GPU_COUNT}"
    echo ""
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader 2>/dev/null | \
        awk -F', ' '{
            used=$3; total=$4; util=$5; temp=$6; idx=$1; name=$2
            pct=used/total*100
            if (pct > 5) color="\033[32m"; else color="\033[0m"
            printf "  GPU %s: %s | 显存 %s/%s (%.0f%%) | 利用率 %s | 温度 %s°C\n", \
                idx, name, used, total, pct, util, temp
        }'
fi
echo ""

# ---------- 3. 模型权重状态 ----------
echo -e "${BLUE}【3】模型权重状态${NC}"
check_weights() {
    local path="$1" name="$2"
    if [[ ! -d "${path}" ]]; then
        echo -e "  ${RED}✗ ${name}${NC}: 未下载 (${path})"
        return
    fi
    # 支持 layers-N.safetensors / model-*.safetensors / 单文件 *.safetensors
    local actual
    actual=$(ls "${path}"/model-*.safetensors "${path}"/layers-*.safetensors 2>/dev/null | wc -l)
    if [[ "${actual}" -eq 0 ]]; then
        actual=$(ls "${path}"/*.safetensors 2>/dev/null | wc -l)
    fi
    local config_ok="✗"
    [[ -f "${path}/config.json" ]] && config_ok="✓"
    local size
    size=$(du -sh "${path}" 2>/dev/null | awk '{print $1}')
    if [[ "${actual}" -gt 0 ]]; then
        echo -e "  ${GREEN}✓ ${name}${NC}: ${actual} safetensors, config=${config_ok}, 大小=${size}"
    else
        echo -e "  ${YELLOW}↓ ${name}${NC}: 未找到 safetensors, config=${config_ok}, 大小=${size} (下载中或不完整)"
    fi
}
check_weights "${QWEN_FP8_MODEL_PATH}" "Qwen3.6-27B-FP8"
check_weights "${AGENTS_MODEL_PATH}"   "Agents-A1-FP8"
echo ""

# ---------- 4. 下载日志(如果下载中) ----------
if [[ -f "${RUNTIME_DIR}/download.pid" ]]; then
    DL_PID=$(cat "${RUNTIME_DIR}/download.pid" 2>/dev/null || echo "")
    if [[ -n "${DL_PID}" ]] && kill -0 "${DL_PID}" 2>/dev/null; then
        echo -e "${BLUE}【4】下载进度 (最近 5 行)${NC}"
        tail -5 "${LOG_DIR}/download.log" 2>/dev/null | sed 's/^/    /' || true
        echo ""
    fi
fi

echo -e "${BLUE}====================================================${NC}"
