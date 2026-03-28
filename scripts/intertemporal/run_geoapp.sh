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

# Create log file with timestamp
LOG_DIR="$PROJECT_ROOT/temp"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/geoapp_${TIMESTAMP}.log"
LATEST_LOG="$LOG_DIR/geoapp_latest.log"

# Start logging - tee to both console and file
exec > >(tee -a "$LOG_FILE") 2>&1

# Create symlink to latest log
ln -sf "$LOG_FILE" "$LATEST_LOG"
echo "[LOG] Logging to: $LOG_FILE"
echo "[LOG] Latest log symlink: $LATEST_LOG"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   GeoViz Explorer - Starting...${NC}"
echo -e "${BLUE}========================================${NC}"
echo "[$(date +%H:%M:%S)] Session started"
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
    echo "[$(date +%H:%M:%S)] Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    pkill -f "run_geoapp.py" 2>/dev/null || true
    pkill -f "vite.*geoapp" 2>/dev/null || true
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    echo "[$(date +%H:%M:%S)] Log saved to: $LOG_FILE"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Check if analysis exists, run compute_geometry_analysis.py if not
ANALYSIS_DIR="$PROJECT_ROOT/$DATA_DIR/analysis"
EMBEDDINGS_DIR="$ANALYSIS_DIR/embeddings/pca"
LEGACY_CACHE_DIR="$PROJECT_ROOT/$DATA_DIR/cache/pca"

if [ ! -d "$EMBEDDINGS_DIR" ] && [ ! -d "$LEGACY_CACHE_DIR" ]; then
    echo -e "${BLUE}----------------------------------------${NC}"
    echo -e "${BLUE}Pre-computed embeddings not found.${NC}"
    echo -e "${BLUE}Running compute_geometry_analysis.py...${NC}"
    echo -e "${BLUE}----------------------------------------${NC}"
    echo
    cd "$PROJECT_ROOT"
    uv run python scripts/intertemporal/compute_geometry_analysis.py --data-dir "$DATA_DIR"
    echo
    echo -e "${GREEN}Analysis complete!${NC}"
    echo
fi

# Start backend server - load-only mode (no runtime computation)
echo -e "${GREEN}Starting backend server...${NC}"
cd "$PROJECT_ROOT"
uv run python scripts/intertemporal/run_geoapp.py --data-dir "$DATA_DIR" &
BACKEND_PID=$!

# Wait for backend to be ready (check health endpoint)
# Server preloads ALL 816 embeddings into memory at startup, which takes 5-7 minutes
echo "Waiting for backend to preload embeddings (this may take 5-7 minutes)..."
BACKEND_READY=false
for i in {1..600}; do  # 600 * 1s = 10 minutes timeout
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
    sleep 1
    # Show progress every 10 seconds
    if (( i % 10 == 0 )); then
        echo -n " ${i}s"
    else
        echo -n "."
    fi
done

if [ "$BACKEND_READY" = false ]; then
    echo ""
    echo -e "\033[0;31mError: Backend failed to start within 10 minutes${NC}"
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
