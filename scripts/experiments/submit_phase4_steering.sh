#!/usr/bin/bash
#SBATCH --job-name=phase4-steering
#SBATCH --time=10:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=80GB
#SBATCH -C GPU_MEM:32GB
#SBATCH --output=logs/phase4_steering_%j_%x.out
#SBATCH --error=logs/phase4_steering_%j_%x.err

# Phase 4: Intervention Experiments — Activation Steering, Context Refresh,
# Prompt Restructuring
# ---------------------------------------------------
# Tests three intervention strategies for mitigating degradation:
#   1. Continuous activation steering (Turner et al. 2023)
#   2. Context refresh (truncate + summarize at probe threshold)
#   3. Prompt restructuring (emphasis, reframing, system reset)
# Plus controls: random direction steering, sycophancy direction steering.
#
# IMPORTANT: Run Phase 3 Exp 1 (refusal direction) first — this script loads
# the saved degradation and refusal direction .npy files.
#
# Usage:
#   sbatch scripts/experiments/submit_phase4_steering.sh
#   sbatch scripts/experiments/submit_phase4_steering.sh Qwen3-8B
#   sbatch scripts/experiments/submit_phase4_steering.sh ALL
#   sbatch scripts/experiments/submit_phase4_steering.sh Llama-3.1-8B-Instruct quick
#
# This experiment is generation-heavy (full rep sequence per strategy)
# so it takes longer than Phase 3 experiments. Allow 4–5hr per model.
#
# GPU: V100 32GB for 8B models, L40S 48GB for Qwen3-30B-A3B:
#   sbatch -C GPU_MEM:48GB --time=14:00:00 scripts/experiments/submit_phase4_steering.sh Qwen3-30B-A3B

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
echo "Phase 4: Intervention Experiments"
echo "Model:   $MODEL"
echo "Mode:    $MODE"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="

# ── Run experiment ───────────────────────────────────────────
srun python3 scripts/experiments/phase4_intervention_steering.py \
    $RUN_ARGS

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results:   results/phase4_intervention/"
echo "=========================================="

exit $EXIT_CODE
