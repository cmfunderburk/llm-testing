#!/bin/bash
#
# LLM Learning Lab - Dashboard Launcher
#
# Starts both the API server and frontend dashboard.
# Usage: ./run-dashboard.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     LLM Learning Lab Dashboard         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found at .venv${NC}"
    echo "Create one with: uv sync"
    exit 1
fi

# Check for node_modules in dashboard
if [ ! -d "dashboard/node_modules" ]; then
    echo -e "${YELLOW}Installing dashboard dependencies...${NC}"
    cd dashboard && npm install && cd ..
fi

# Function to cleanup background processes on exit
cleanup() {
    echo
    echo -e "${YELLOW}Shutting down...${NC}"
    kill $API_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start API server
echo -e "${GREEN}Starting API server...${NC}"
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000 &
API_PID=$!

# Wait for API to be ready
echo -e "${YELLOW}Waiting for API to start...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}API ready!${NC}"
        break
    fi
    sleep 0.5
done

# Start frontend
echo -e "${GREEN}Starting frontend...${NC}"
cd dashboard && npm run dev &
FRONTEND_PID=$!
cd ..

echo
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}Dashboard running!${NC}"
echo -e "  Frontend: ${BLUE}http://localhost:5173${NC}"
echo -e "  API docs: ${BLUE}http://localhost:8000/docs${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo

# Wait for either process to exit
wait $API_PID $FRONTEND_PID
