#!/usr/bin/bash
#SBATCH --time=06:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_BRD:TESLA
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB

module load python/3.12.1
module load py-pyarrow/18.1.0_py312
source ~/sae-env/bin/activate
export HF_HOME=~/.cache/huggingface

MODE="${1:-full}"
EXTRA_ARGS="${2:-}"

cd "${SLURM_SUBMIT_DIR:-$HOME/temporal-awareness}"

RESULTS_DIR="results/sequential_tracking"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Sequential Activation Tracking (RQ1)"
echo "Mode: $MODE"
echo "PWD: $(pwd)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Started: $(date)"
echo "=========================================="

WANDB_PROJECT="sequential-tracking"
WANDB_ENTITY="justinshenk-time"
WANDB_RUN_NAME="seq-tracking-${MODE}-$(date +%Y%m%d_%H%M%S)"

if [ "$MODE" = "quick" ]; then
    EXTRA_ARGS="--quick $EXTRA_ARGS"
fi

srun python3 scripts/experiments/sequential_activation_tracking.py \
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
