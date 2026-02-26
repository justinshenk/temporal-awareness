#!/bin/bash
# Run SAE Feature Stability experiment on GPU
#
# Usage:
#   # Full experiment (all 4 layers, ~2-4 hours on A100)
#   bash scripts/experiments/run_sae_stability_gpu.sh
#
#   # Quick sanity check (~15 min on A100)
#   bash scripts/experiments/run_sae_stability_gpu.sh --quick
#
# Requirements:
#   pip install transformer-lens sae-lens torch
#   GPU with >=24GB VRAM (A100 recommended, V100 works with smaller batch)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    DEVICE="cuda"
    BATCH_SIZE=32
else
    echo "No GPU detected, using CPU (will be slow)"
    DEVICE="cpu"
    BATCH_SIZE=8
fi

# Parse arguments
EXTRA_ARGS=""
if [[ "${1:-}" == "--quick" ]]; then
    EXTRA_ARGS="--quick"
    echo "Running in QUICK mode (single layer)"
fi

# Create output directory
RESULTS_DIR="results/sae_feature_stability"
mkdir -p "$RESULTS_DIR"

echo ""
echo "================================================================"
echo "SAE Feature Stability Under Distribution Shift"
echo "================================================================"
echo "Device:     $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo "Output:     $RESULTS_DIR"
echo "Started:    $(date)"
echo "================================================================"
echo ""

# Run experiment
python scripts/experiments/sae_feature_stability.py \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$RESULTS_DIR" \
    $EXTRA_ARGS \
    2>&1 | tee "$RESULTS_DIR/experiment_log_$(date +%Y%m%d_%H%M%S).txt"

echo ""
echo "================================================================"
echo "Experiment complete at $(date)"
echo "Results in: $RESULTS_DIR"
echo "================================================================"
