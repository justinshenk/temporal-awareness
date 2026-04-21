#!/usr/bin/bash
# ── Split Phase 5 Safety Evaluations into per-eval SLURM jobs ──
#
# For large models (Qwen3-30B-A3B) where the full safety eval exceeds
# wall time limits. Submits 4 separate jobs, one per evaluation.
#
# Usage:
#   bash scripts/experiments/submit_phase5_safety_split.sh Qwen3-30B-A3B
#   bash scripts/experiments/submit_phase5_safety_split.sh Ouro-2.6B
#
# Or submit a single eval:
#   sbatch scripts/experiments/submit_phase5_safety_split.sh Qwen3-30B-A3B alignment_stability

#SBATCH --job-name=p5-split
#SBATCH --time=08:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=80GB
#SBATCH --output=logs/phase5_safety_split_%j.out
#SBATCH --error=logs/phase5_safety_split_%j.err

MODEL="${1:-Qwen3-30B-A3B}"
EVAL="${2:-}"

# ── Environment setup ────────────────────────────────────────
module load python/3.12.1
module load py-pyarrow/18.1.0_py312

OURO_ENV_DIR="${OURO_ENV_DIR:-/home/groups/barbarae/molofsky/ouro-env}"
if [[ "$MODEL" == "Ouro-2.6B" ]] && [ -f "$OURO_ENV_DIR/bin/activate" ]; then
    echo "Using Ouro-compatible venv (transformers==4.54.1)"
    source "$OURO_ENV_DIR/bin/activate"
elif [ -f /home/groups/barbarae/molofsky/ml-env/bin/activate ]; then
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
export WANDB_MODE=online

cd "${SLURM_SUBMIT_DIR:-$HOME/temporal-awareness}"
mkdir -p logs

echo "=========================================="
echo "Phase 5: Safety Split — $MODEL"
echo "Eval:    ${EVAL:-all (submitting 4 separate jobs)}"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="

# If called with a specific eval, run it directly
if [ -n "$EVAL" ]; then
    srun python3 scripts/experiments/phase5_safety_evaluations.py \
        --model "$MODEL" --device cuda --wandb-project patience-degradation \
        --eval "$EVAL"
    echo "Finished $EVAL: $(date), exit code: $?"
    exit $?
fi

# Otherwise, submit 4 separate jobs (called from login node, not via sbatch)
EVALS=("alignment_stability" "realtime_monitoring" "intervention_efficacy")
# NOTE: deployment_robustness already partially done for Qwen3-30B-A3B (injection done, fidelity not)
# Include it to get the remaining sub-evaluations
EVALS+=("deployment_robustness")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Determine GPU constraint based on model
if [[ "$MODEL" == "Qwen3-30B-A3B" ]]; then
    GPU_CONSTRAINT="-C GPU_MEM:80GB"
    TIME="08:00:00"
else
    GPU_CONSTRAINT="-C GPU_MEM:32GB"
    TIME="04:00:00"
fi

echo "Submitting 4 split jobs for $MODEL..."
for EVAL_NAME in "${EVALS[@]}"; do
    JOB_ID=$(sbatch \
        --job-name="p5-${EVAL_NAME:0:6}-${MODEL:0:8}" \
        --time=$TIME \
        -p gpu -G 1 $GPU_CONSTRAINT \
        --cpus-per-gpu=4 --gpus-per-node=1 --mem=80GB \
        --output="logs/phase5_${EVAL_NAME}_${MODEL}_%j.out" \
        --error="logs/phase5_${EVAL_NAME}_${MODEL}_%j.err" \
        "$SCRIPT_DIR/submit_phase5_safety_split.sh" "$MODEL" "$EVAL_NAME" \
        2>&1 | grep -oP '\d+')
    echo "  $EVAL_NAME → job $JOB_ID ($TIME, $GPU_CONSTRAINT)"
done
echo "Done. Check: squeue -u \$USER"
