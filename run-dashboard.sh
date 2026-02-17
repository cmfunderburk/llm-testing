#!/bin/bash
#
# LLM Learning Lab - Dashboard Launcher
#
# Starts the API server (or attaches to an existing one) and keeps
# the frontend alive on an available localhost port.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

API_HOST="127.0.0.1"
API_PORT=8000
API_RELOAD=0
FRONTEND_HOST="127.0.0.1"
FRONTEND_PORT_BASE=5173
FRONTEND_PORT_MAX=5273
FRONTEND_PORT="$FRONTEND_PORT_BASE"
API_BASE_URL="http://${API_HOST}:${API_PORT}"
API_PID=""
FRONTEND_PID=""
API_MANAGED=0
SHUTTING_DOWN=0

usage() {
    cat <<EOF
Usage: ./run-dashboard.sh [options]

Options:
  --dev                    Enable API auto-reload (not recommended for long training runs)
  --api-port <port>        API port (default: 8000)
  --frontend-port <port>   Starting frontend port to try (default: 5173)
  --help                   Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev|--reload)
            API_RELOAD=1
            shift
            ;;
        --api-port)
            if [[ $# -lt 2 ]]; then
                echo -e "${RED}Missing value for --api-port${NC}"
                exit 1
            fi
            API_PORT="$2"
            shift 2
            ;;
        --frontend-port)
            if [[ $# -lt 2 ]]; then
                echo -e "${RED}Missing value for --frontend-port${NC}"
                exit 1
            fi
            FRONTEND_PORT_BASE="$2"
            FRONTEND_PORT="$FRONTEND_PORT_BASE"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

API_BASE_URL="http://${API_HOST}:${API_PORT}"

is_port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "( sport = :${port} )" 2>/dev/null | awk 'NR > 1 { found = 1 } END { exit found ? 0 : 1 }'
        return $?
    fi
    if command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"${port}" -sTCP:LISTEN -Pn >/dev/null 2>&1
        return $?
    fi
    return 1
}

find_available_port() {
    local start="$1"
    local end="$2"
    local port
    for ((port=start; port<=end; port++)); do
        if ! is_port_in_use "$port"; then
            echo "$port"
            return 0
        fi
    done
    return 1
}

is_api_healthy() {
    curl -fsS "${API_BASE_URL}/api/health" >/dev/null 2>&1
}

is_training_active() {
    local pretraining_status fine_tuning_status
    pretraining_status="$(curl -fsS "${API_BASE_URL}/api/pretraining/status" 2>/dev/null || true)"
    fine_tuning_status="$(curl -fsS "${API_BASE_URL}/api/fine-tuning/status" 2>/dev/null || true)"

    if echo "$pretraining_status" | grep -Eq '"state":"(running|loading|paused)"'; then
        return 0
    fi
    if echo "$fine_tuning_status" | grep -Eq '"state":"(running|loading|paused)"'; then
        return 0
    fi
    return 1
}

request_checkpoint_save() {
    curl -fsS -X POST "${API_BASE_URL}/api/pretraining/checkpoint/save-now" >/dev/null 2>&1 || true
    curl -fsS -X POST "${API_BASE_URL}/api/fine-tuning/checkpoint/save-now" >/dev/null 2>&1 || true
}

print_urls() {
    echo
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo -e "${GREEN}Dashboard running!${NC}"
    echo -e "  Frontend: ${BLUE}http://${FRONTEND_HOST}:${FRONTEND_PORT}${NC}"
    echo -e "  API docs: ${BLUE}${API_BASE_URL}/docs${NC}"
    if [[ "$API_MANAGED" -eq 0 ]]; then
        echo -e "  API mode: ${YELLOW}attached to existing API process${NC}"
    fi
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo
}

start_api() {
    local reload_flag=()
    if [[ "$API_RELOAD" -eq 1 ]]; then
        reload_flag=(--reload)
    fi

    echo -e "${GREEN}Starting API server...${NC}"
    # shellcheck source=/dev/null
    source .venv/bin/activate
    uvicorn api.main:app --host "$API_HOST" --port "$API_PORT" "${reload_flag[@]}" &
    API_PID=$!
    API_MANAGED=1

    echo -e "${YELLOW}Waiting for API to start...${NC}"
    for _ in {1..60}; do
        if is_api_healthy; then
            echo -e "${GREEN}API ready!${NC}"
            return 0
        fi
        if ! kill -0 "$API_PID" 2>/dev/null; then
            echo -e "${RED}API process exited during startup.${NC}"
            return 1
        fi
        sleep 0.5
    done

    echo -e "${RED}API did not become ready in time.${NC}"
    return 1
}

start_or_attach_api() {
    if is_api_healthy; then
        echo -e "${GREEN}Found running API at ${API_BASE_URL}; attaching.${NC}"
        API_MANAGED=0
        return 0
    fi

    if is_port_in_use "$API_PORT"; then
        echo -e "${RED}Port ${API_PORT} is in use by a non-LLM process.${NC}"
        echo -e "${RED}Use --api-port to pick another port, or stop the conflicting service.${NC}"
        return 1
    fi

    start_api
}

start_frontend() {
    FRONTEND_PORT="$(find_available_port "$FRONTEND_PORT_BASE" "$FRONTEND_PORT_MAX")" || {
        echo -e "${RED}No available frontend port found in range ${FRONTEND_PORT_BASE}-${FRONTEND_PORT_MAX}.${NC}"
        return 1
    }

    echo -e "${GREEN}Starting frontend on http://${FRONTEND_HOST}:${FRONTEND_PORT}...${NC}"
    (
        cd dashboard
        npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT" --strictPort
    ) &
    FRONTEND_PID=$!

    for _ in {1..40}; do
        if curl -fsS "http://${FRONTEND_HOST}:${FRONTEND_PORT}" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
            echo -e "${RED}Frontend process exited during startup.${NC}"
            return 1
        fi
        sleep 0.25
    done

    echo -e "${RED}Frontend did not become ready in time.${NC}"
    return 1
}

cleanup() {
    local exit_code="${1:-$?}"
    if [[ "$SHUTTING_DOWN" -eq 1 ]]; then
        return 0
    fi
    SHUTTING_DOWN=1
    trap - EXIT SIGINT SIGTERM

    echo
    echo -e "${YELLOW}Shutting down...${NC}"

    if [[ -n "${FRONTEND_PID}" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi

    if [[ "$API_MANAGED" -eq 1 ]] && [[ -n "${API_PID}" ]] && kill -0 "$API_PID" 2>/dev/null; then
        if is_training_active; then
            echo -e "${YELLOW}Active training detected. Requesting immediate checkpoint save...${NC}"
            request_checkpoint_save
            sleep 3
        fi
        kill "$API_PID" 2>/dev/null || true
    fi

    if [[ -n "${FRONTEND_PID}" ]]; then
        wait "$FRONTEND_PID" 2>/dev/null || true
    fi
    if [[ "$API_MANAGED" -eq 1 ]] && [[ -n "${API_PID}" ]]; then
        wait "$API_PID" 2>/dev/null || true
    fi

    exit "$exit_code"
}

trap 'cleanup $?' SIGINT SIGTERM EXIT

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     LLM Learning Lab Dashboard         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo

# Check for virtual environment
if [[ ! -d ".venv" ]]; then
    echo -e "${RED}Error: Virtual environment not found at .venv${NC}"
    echo "Create one with: uv sync"
    exit 1
fi

# Check for node_modules in dashboard
if [[ ! -d "dashboard/node_modules" ]]; then
    echo -e "${YELLOW}Installing dashboard dependencies...${NC}"
    (
        cd dashboard
        npm install
    )
fi

start_or_attach_api
start_frontend
print_urls

while true; do
    if [[ "$API_MANAGED" -eq 1 ]] && [[ -n "${API_PID}" ]] && ! kill -0 "$API_PID" 2>/dev/null; then
        echo -e "${RED}API server exited. Shutting down launcher.${NC}"
        cleanup 1
    fi

    if [[ -n "${FRONTEND_PID}" ]] && ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "${YELLOW}Frontend exited. Restarting on an available port...${NC}"
        start_frontend
        print_urls
    fi

    sleep 2
done
