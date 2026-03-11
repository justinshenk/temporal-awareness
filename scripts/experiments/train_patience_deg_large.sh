#!/usr/bin/bash
#SBATCH --time=08:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:48GB
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB

# For instruction-tuned models (Qwen2.5-3B-Instruct, Llama-3.1-8B-Instruct)
# Requires >=48GB VRAM (L40S) for Llama-3.1-8B

module load python/3.12.1
module load py-pyarrow/18.1.0_py312
source ~/sae-env/bin/activate
export HF_HOME=$SCRATCH/.cache/huggingface
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")

MODEL="${1:?Usage: sbatch train_patience_deg_large.sh MODEL_NAME [quick|full]}"
MODE="${2:-full}"
EXTRA_ARGS="${3:-}"

cd "${SLURM_SUBMIT_DIR:-$HOME/temporal-awareness}"

RESULTS_DIR="results/patience_degradation"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Patience & Compliance Degradation (RQ3)"
echo "Model: $MODEL"
echo "Mode: $MODE"
echo "PWD: $(pwd)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Started: $(date)"
echo "=========================================="

WANDB_PROJECT="patience-degradation"
WANDB_ENTITY="justinshenk-time"
# Sanitize model name for W&B (replace / with -)
MODEL_SLUG=$(echo "$MODEL" | tr '/' '-')
WANDB_RUN_NAME="patience-deg-${MODEL_SLUG}-${MODE}-$(date +%Y%m%d_%H%M%S)"

if [ "$MODE" = "quick" ]; then
    EXTRA_ARGS="--quick $EXTRA_ARGS"
fi

srun python3 scripts/experiments/patience_degradation.py \
    --model "$MODEL" \
    --device cuda \
    --batch-size 8 \
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
