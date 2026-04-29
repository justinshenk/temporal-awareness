#!/bin/bash
# Run GeoViz Explorer - starts backend, frontend, and opens browser
#
# Usage:
#   ./scripts/intertemporal/run_geoapp.sh           # Load ALL datasets from out/geo/
#   ./scripts/intertemporal/run_geoapp.sh geometry  # Load only geometry dataset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/src/intertemporal/geoapp/frontend"

# Base directory for datasets
BASE_DIR="out/geo"

# Dataset argument (optional - if not provided, load all datasets)
DATASET_ARG="$1"

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
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   GeoViz Explorer - Starting...${NC}"
echo -e "${BLUE}========================================${NC}"
echo "[$(date +%H:%M:%S)] Session started"
echo

# Kill any existing servers on ports 8000 and 3000
echo "Killing any existing servers on ports 8000 and 3000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
pkill -f "run_geoapp.py" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
sleep 2
# Verify ports are free
if lsof -i:8000 >/dev/null 2>&1 || lsof -i:3000 >/dev/null 2>&1; then
    echo -e "${RED}Warning: Ports 8000 or 3000 still in use. Forcing...${NC}"
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi
echo "Ports cleared."

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

# Discover or validate datasets
if [ -n "$DATASET_ARG" ]; then
    # Single dataset specified
    DATASETS=("$DATASET_ARG")
    echo "Loading single dataset: $DATASET_ARG"
else
    # Discover all datasets
    echo "Discovering datasets in $BASE_DIR/..."
    DATASETS=()
    for dir in "$PROJECT_ROOT/$BASE_DIR"/*/; do
        if [ -d "${dir}data/samples" ] && [ -d "${dir}analysis/embeddings" ]; then
            name=$(basename "$dir")
            DATASETS+=("$name")
        fi
    done
    if [ ${#DATASETS[@]} -eq 0 ]; then
        echo -e "${RED}No valid datasets found in $BASE_DIR/${NC}"
        exit 1
    fi
    echo "Found ${#DATASETS[@]} dataset(s): ${DATASETS[*]}"
fi
echo


# Build the dataset arguments for Python script
DATASET_ARGS=""
if [ -n "$DATASET_ARG" ]; then
    DATASET_ARGS="$DATASET_ARG"
fi

# Start backend server - load-only mode (no runtime computation)
echo -e "${GREEN}Starting backend server...${NC}"
cd "$PROJECT_ROOT"
uv run python scripts/intertemporal/run_geoapp.py $DATASET_ARGS &
BACKEND_PID=$!

# Wait for backend to be ready (check health endpoint)
echo "Waiting for backend to preload data..."
BACKEND_READY=false
for i in {1..600}; do  # 600 * 1s = 10 minutes timeout
    if curl -s http://localhost:8000/api/datasets > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}Backend is ready!${NC}"
        BACKEND_READY=true
        break
    fi
    # Check if backend process died
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo ""
        echo -e "${RED}Error: Backend process died unexpectedly${NC}"
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
    echo -e "${RED}Error: Backend failed to start within 10 minutes${NC}"
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

# Open browser for ALL datasets
echo -e "${GREEN}Opening browser for all ${#DATASETS[@]} datasets...${NC}"
for dataset in "${DATASETS[@]}"; do
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "http://localhost:3000/${dataset}"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open "http://localhost:3000/${dataset}" 2>/dev/null || echo "Please open http://localhost:3000/${dataset} in your browser"
    fi
    sleep 0.5  # Small delay between opening windows
done

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}GeoViz Explorer is running!${NC}"
echo -e "${BLUE}========================================${NC}"
echo
echo "  Datasets loaded: ${DATASETS[*]}"
echo
echo "  URLs:"
for dataset in "${DATASETS[@]}"; do
    echo "    - http://localhost:3000/${dataset}"
done
echo
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo
echo "Press Ctrl+C to stop all servers"
echo

# Wait for any process to exit
wait
