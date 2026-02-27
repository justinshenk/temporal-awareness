#!/usr/bin/bash
#SBATCH --time=06:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_BRD:TESLA
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB

module load python/3.12.1

MODE="${1:-full}"
EXTRA_ARGS="${2:-}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="results/sae_feature_stability"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "SAE Feature Stability Experiment"
echo "Mode: $MODE"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Started: $(date)"
echo "=========================================="

# W&B config — logs to justinshenk-time team workspace
WANDB_PROJECT="sae-feature-stability"
WANDB_ENTITY="justinshenk-time"
WANDB_RUN_NAME="sae-stability-${MODE}-$(date +%Y%m%d_%H%M%S)"

srun python3 scripts/experiments/sae_feature_stability.py \
    --device cuda \
    --batch-size 32 \
    --output-dir "$RESULTS_DIR" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-run-name "$WANDB_RUN_NAME" \
    $EXTRA_ARGS

echo "=========================================="
echo "Finished: $(date)"
echo "Results: $RESULTS_DIR"
echo "W&B: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "=========================================="
