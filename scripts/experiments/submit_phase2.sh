#!/usr/bin/bash
#SBATCH --job-name=phase2-cross-domain
#SBATCH --time=12:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=80GB
#SBATCH --output=logs/phase2_%j_%x.out
#SBATCH --error=logs/phase2_%j_%x.err

# Phase 2/3: Cross-Domain & Cross-Stake Activation Analysis
# -----------------------------------------------------------
# Runs the new activation_api pipeline (replaces TransformerLens).
#
# Usage:
#   # Single model (default: Llama-3.1-8B-Instruct)
#   sbatch scripts/experiments/submit_phase2.sh
#
#   # Specific model
#   sbatch scripts/experiments/submit_phase2.sh Qwen3-8B
#
#   # Quick validation run
#   sbatch scripts/experiments/submit_phase2.sh Llama-3.1-8B-Instruct quick
#
#   # All 4 models (sequential, needs ~12h on L40S)
#   sbatch --time=24:00:00 --mem=96GB scripts/experiments/submit_phase2.sh ALL
#
# GPU requirements:
#   - 7B/8B models (Llama, Qwen3-8B, DeepSeek): V100 32GB or L40S 48GB
#   - Qwen3-30B-A3B (MoE, ~3B active): V100 32GB works but L40S preferred
#   - Activation streaming to CPU keeps GPU RAM bounded
#
# Uncomment one of the following for GPU memory constraints:
##SBATCH -C GPU_MEM:48GB    # L40S — recommended for all models
##SBATCH -C GPU_MEM:32GB    # V100 — fine for 7B/8B

# ── Environment setup ────────────────────────────────────────
module load python/3.12.1
module load py-pyarrow/18.1.0_py312
source ~/sae-env/bin/activate

export HF_HOME=$SCRATCH/.cache/huggingface
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Parse arguments ──────────────────────────────────────────
MODEL="${1:-Llama-3.1-8B-Instruct}"
MODE="${2:-full}"

cd "${SLURM_SUBMIT_DIR:-$HOME/temporal-awareness}"
mkdir -p logs

# ── W&B configuration ────────────────────────────────────────
WANDB_PROJECT="patience-degradation"
WANDB_ENTITY="justinshenk-time"
MODEL_SLUG=$(echo "$MODEL" | tr '/' '-')
WANDB_RUN_NAME="phase2-${MODEL_SLUG}-${MODE}-$(date +%Y%m%d_%H%M%S)"

# ── Build command ────────────────────────────────────────────
COMMON_ARGS="--device cuda --wandb-project $WANDB_PROJECT --save-activations"

if [ "$MODEL" = "ALL" ]; then
    RUN_ARGS="--all-models $COMMON_ARGS"
    WANDB_RUN_NAME="phase2-all-models-${MODE}-$(date +%Y%m%d_%H%M%S)"
else
    RUN_ARGS="--model $MODEL $COMMON_ARGS"
fi

if [ "$MODE" = "quick" ]; then
    RUN_ARGS="$RUN_ARGS --quick"
fi

# ── Print job info ───────────────────────────────────────────
echo "=========================================="
echo "Phase 2: Cross-Domain Activation Analysis"
echo "Model:   $MODEL"
echo "Mode:    $MODE"
echo "PWD:     $(pwd)"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "CUDA:    $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>/dev/null)"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="

# ── Run experiment ───────────────────────────────────────────
srun python3 scripts/experiments/phase2_cross_domain.py \
    $RUN_ARGS

EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results:   results/phase2_cross_domain/"
echo "W&B:       https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "=========================================="

exit $EXIT_CODE
