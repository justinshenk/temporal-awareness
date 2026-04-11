#!/usr/bin/bash
#SBATCH --job-name=phase3-cross-model
#SBATCH --time=08:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=80GB
#SBATCH -C GPU_MEM:32GB
#SBATCH --output=logs/phase3_cross_model_%j_%x.out
#SBATCH --error=logs/phase3_cross_model_%j_%x.err

# Phase 3: Cross-Model Direction Transfer
# ---------------------------------------------------
# Tests whether the degradation direction is universal across architectures:
#   - Pairwise direction cosine similarity matrix
#   - Cross-model probe transfer (train on A, test on B)
#   - CKA for dimension-agnostic comparison (DeepSeek d=3584 vs others d=4096)
#   - Base vs Instruct Llama comparison (RLHF creates degradation?)
#   - Direction injection transfer testing
#
# IMPORTANT: This experiment loads ALL models sequentially (pairwise comparison).
# For single-model direction extraction, use --model. For full cross-model
# analysis, use ALL (requires sequential model loading, ~8hr total).
#
# Usage:
#   sbatch scripts/experiments/submit_phase3_cross_model.sh
#   sbatch scripts/experiments/submit_phase3_cross_model.sh ALL
#   sbatch scripts/experiments/submit_phase3_cross_model.sh Llama-3.1-8B-Instruct quick
#
# GPU: V100 32GB for 8B models. For full cross-model (ALL), use L40S:
#   sbatch -C GPU_MEM:48GB --time=12:00:00 scripts/experiments/submit_phase3_cross_model.sh ALL

# ── Environment setup ────────────────────────────────────────
module load python/3.12.1
module load py-pyarrow/18.1.0_py312

if [ -f /home/groups/barbarae/molofsky/ml-env/bin/activate ]; then
    source /home/groups/barbarae/molofsky/ml-env/bin/activate
elif [ -f ~/sae-env/bin/activate ]; then
    source ~/sae-env/bin/activate
else
    echo "WARNING: Could not find virtualenv, using system python"
fi

export HF_HOME=$SCRATCH/.cache/huggingface
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Parse arguments ──────────────────────────────────────────
MODEL="${1:-ALL}"
MODE="${2:-full}"

cd "${SLURM_SUBMIT_DIR:-$HOME/temporal-awareness}"
mkdir -p logs

# ── Build command ────────────────────────────────────────────
COMMON_ARGS="--device cuda --wandb-project patience-degradation"

if [ "$MODEL" = "ALL" ]; then
    RUN_ARGS="--all-models $COMMON_ARGS"
else
    RUN_ARGS="--model $MODEL $COMMON_ARGS"
fi

if [ "$MODE" = "quick" ]; then
    RUN_ARGS="$RUN_ARGS --quick"
fi

# ── Print job info ───────────────────────────────────────────
echo "=========================================="
echo "Phase 3: Cross-Model Direction Transfer"
echo "Model:   $MODEL"
echo "Mode:    $MODE"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="

# ── Run experiment ───────────────────────────────────────────
srun python3 scripts/experiments/phase3_cross_model_transfer.py \
    $RUN_ARGS

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results:   results/phase3_cross_model_transfer/"
echo "=========================================="

exit $EXIT_CODE
