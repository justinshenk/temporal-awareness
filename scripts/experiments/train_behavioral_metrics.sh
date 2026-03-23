#!/bin/bash
#SBATCH --job-name=behavioral-metrics
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/behavioral_%x_%j.out
#SBATCH --error=logs/behavioral_%x_%j.err

# Behavioral Metrics for Patience Degradation -- Phase 1
#
# Usage:
#   sbatch --export=MODEL=gemma-2-2b scripts/experiments/train_behavioral_metrics.sh
#   sbatch --export=MODEL=gpt2,QUICK=1 scripts/experiments/train_behavioral_metrics.sh
#   bash scripts/experiments/submit_behavioral_all.sh

set -euo pipefail

# Load Python 3.12 and activate sae-env
module load python/3.12.1
source ~/sae-env/bin/activate

# HuggingFace token for gated models (gemma, llama)
# Reads from ~/.huggingface_token file — create with:
#   echo "YOUR_HF_TOKEN" > ~/.huggingface_token
if [ -f "$HOME/.huggingface_token" ]; then
    export HF_TOKEN="$(cat $HOME/.huggingface_token)"
fi

# Force unbuffered output so logs stream in real time
export PYTHONUNBUFFERED=1

# Cache HF models to shared storage (avoids re-download per node)
export HF_HOME="$HOME/.cache/huggingface"

# Defaults
MODEL=${MODEL:-gemma-2-2b}
QUICK=${QUICK:-0}
MAX_EXAMPLES=${MAX_EXAMPLES:-200}
STAKES=${STAKES:-""}
WANDB_PROJECT=${WANDB_PROJECT:-patience-degradation}

echo "============================================"
echo "Behavioral Metrics Experiment"
echo "Model: $MODEL"
echo "Quick: $QUICK"
echo "Max examples: $MAX_EXAMPLES"
echo "Stakes: ${STAKES:-all}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "============================================"

# Setup
cd "$SLURM_SUBMIT_DIR" || cd "$(dirname "$0")/../.."
mkdir -p logs results/behavioral_metrics

# Build command
CMD="python3 scripts/experiments/behavioral_metrics.py"
CMD="$CMD --model $MODEL"
CMD="$CMD --device cuda"
CMD="$CMD --batch-size 8"
CMD="$CMD --max-examples $MAX_EXAMPLES"
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
