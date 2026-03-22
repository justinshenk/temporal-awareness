#!/bin/bash
#SBATCH --job-name=behavioral-metrics
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/behavioral_%x_%j.out
#SBATCH --error=logs/behavioral_%x_%j.err

# Behavioral Metrics for Patience Degradation — Phase 1
#
# Usage:
#   # Single model
#   sbatch --export=MODEL=gemma-2-2b scripts/experiments/train_behavioral_metrics.sh
#
#   # All models (submit 5 jobs)
#   for model in gemma-2-2b gpt2 pythia-70m Qwen2.5-3B-Instruct Llama-3.1-8B-Instruct; do
#     sbatch --export=MODEL=$model scripts/experiments/train_behavioral_metrics.sh
#   done
#
#   # Quick validation
#   sbatch --export=MODEL=gpt2,QUICK=1 scripts/experiments/train_behavioral_metrics.sh

set -euo pipefail

# Defaults
MODEL=${MODEL:-gemma-2-2b}
QUICK=${QUICK:-0}
STAKES=${STAKES:-""}  # empty = all stakes
WANDB_PROJECT=${WANDB_PROJECT:-patience-degradation}

echo "============================================"
echo "Behavioral Metrics Experiment"
echo "Model: $MODEL"
echo "Quick: $QUICK"
echo "Stakes: ${STAKES:-all}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "============================================"

# Setup
cd "$SLURM_SUBMIT_DIR" || cd "$(dirname "$0")/../.."
mkdir -p logs results/behavioral_metrics

# Activate environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Build command
CMD="python scripts/experiments/behavioral_metrics.py"
CMD="$CMD --model $MODEL"
CMD="$CMD --device cuda"
CMD="$CMD --batch-size 8"
CMD="$CMD --wandb-project $WANDB_PROJECT"

if [ "$QUICK" -eq 1 ]; then
    CMD="$CMD --quick"
fi

if [ -n "$STAKES" ]; then
    CMD="$CMD --stakes $STAKES"
fi

echo "Running: $CMD"
eval $CMD

echo "Done: $(date)"
