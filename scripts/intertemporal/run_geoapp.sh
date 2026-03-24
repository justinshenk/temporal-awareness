#!/bin/bash
# Run GeoViz Explorer - starts backend, frontend, and opens browser
#
# Usage:
#   ./scripts/intertemporal/run_geoapp.sh
#   ./scripts/intertemporal/run_geoapp.sh --data-dir out/geo_test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/src/intertemporal/geoapp/frontend"

# Default data directory
DATA_DIR="${1:-out/geometry}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   GeoViz Explorer - Starting...${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Kill any existing processes from previous runs
echo "Cleaning up any existing processes..."
pkill -f "run_geoapp.py" 2>/dev/null || true
pkill -f "vite.*geoapp" 2>/dev/null || true
# Also kill by port
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
sleep 1

# Cleanup function to kill background processes on exit
cleanup() {
    echo
    echo -e "${BLUE}Shutting down...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    pkill -f "run_geoapp.py" 2>/dev/null || true
    pkill -f "vite.*geoapp" 2>/dev/null || true
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Start backend server
echo -e "${GREEN}Starting backend server...${NC}"
cd "$PROJECT_ROOT"
uv run python scripts/intertemporal/run_geoapp.py --data-dir "$DATA_DIR" --dev &
BACKEND_PID=$!

# Wait for backend to be ready (check health endpoint)
echo "Waiting for backend to be ready..."
BACKEND_READY=false
for i in {1..60}; do
    if curl -s http://localhost:8000/api/config > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}Backend is ready!${NC}"
        BACKEND_READY=true
        break
    fi
    # Check if backend process died
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo ""
        echo -e "\033[0;31mError: Backend process died unexpectedly${NC}"
        exit 1
    fi
    sleep 0.5
    echo -n "."
done

if [ "$BACKEND_READY" = false ]; then
    echo ""
    echo -e "\033[0;31mError: Backend failed to start within 30 seconds${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Small buffer after backend ready
sleep 1

# Start frontend dev server
echo -e "${GREEN}Starting frontend dev server...${NC}"
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start and verify
echo "Waiting for frontend to be ready..."
FRONTEND_READY=false
for i in {1..30}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo -e "${GREEN}Frontend is ready!${NC}"
        FRONTEND_READY=true
        break
    fi
    sleep 0.5
    echo -n "."
done
echo ""

# Open browser
echo -e "${GREEN}Opening browser...${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "http://localhost:3000"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open "http://localhost:3000" 2>/dev/null || echo "Please open http://localhost:3000 in your browser"
fi

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}GeoViz Explorer is running!${NC}"
echo -e "${BLUE}========================================${NC}"
echo
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo
echo "Press Ctrl+C to stop all servers"
echo

# Wait for any process to exit
wait
