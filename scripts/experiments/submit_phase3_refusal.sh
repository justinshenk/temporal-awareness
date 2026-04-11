#!/usr/bin/bash
#SBATCH --job-name=phase3-refusal
#SBATCH --time=06:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=80GB
#SBATCH -C GPU_MEM:32GB
#SBATCH --output=logs/phase3_refusal_%j_%x.out
#SBATCH --error=logs/phase3_refusal_%j_%x.err

# Phase 3, Experiment 1: Refusal vs Degradation Direction Comparison
# -------------------------------------------------------------------
# Extracts refusal directions (AdvBench + HarmBench) and degradation
# directions, computes cosine similarity, saves .npy direction vectors
# for downstream causal patching (Experiment 3).
#
# Usage:
#   sbatch scripts/experiments/submit_phase3_refusal.sh
#   sbatch scripts/experiments/submit_phase3_refusal.sh Qwen3-8B
#   sbatch scripts/experiments/submit_phase3_refusal.sh ALL
#   sbatch scripts/experiments/submit_phase3_refusal.sh Llama-3.1-8B-Instruct quick
#
# GPU: ~16GB model + activations. V100 32GB sufficient for all models
# except Qwen3-30B-A3B which needs L40S 48GB:
#   sbatch -C GPU_MEM:48GB scripts/experiments/submit_phase3_refusal.sh Qwen3-30B-A3B

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
MODEL="${1:-Llama-3.1-8B-Instruct}"
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
echo "Phase 3, Exp 1: Refusal Direction Comparison"
echo "Model:   $MODEL"
echo "Mode:    $MODE"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="

# ── Run experiment ───────────────────────────────────────────
srun python3 scripts/experiments/phase3_refusal_direction.py \
    $RUN_ARGS

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results:   results/phase3_refusal_direction/"
echo "=========================================="

exit $EXIT_CODE
