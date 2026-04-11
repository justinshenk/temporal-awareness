#!/usr/bin/bash
# ── Per-model SLURM submission for Phase 5: Safety Evaluations ──
#
# Runs all 4 safety evaluations: deployment robustness, real-time monitoring,
# alignment stability, and intervention efficacy.
#
# Generation-heavy experiment (prompt injection + safety boundary testing +
# steering + latency measurement), so allocate generous time.
#
# Usage:
#   sbatch scripts/experiments/submit_phase5_safety.sh Llama-3.1-8B-Instruct
#   sbatch scripts/experiments/submit_phase5_safety.sh Qwen3-8B
#   sbatch scripts/experiments/submit_phase5_safety.sh Qwen3-30B-A3B
#   sbatch scripts/experiments/submit_phase5_safety.sh DeepSeek-R1-Distill-Qwen-7B
#   sbatch scripts/experiments/submit_phase5_safety.sh Llama-3.1-8B

#SBATCH --job-name=p5-safety
#SBATCH --time=08:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=80GB
#SBATCH --output=logs/phase5_safety_%j.out
#SBATCH --error=logs/phase5_safety_%j.err

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
export WANDB_MODE=online

# ── Parse arguments ──────────────────────────────────────────
MODEL="${1:-Llama-3.1-8B-Instruct}"

cd "${SLURM_SUBMIT_DIR:-$HOME/temporal-awareness}"
mkdir -p logs

# ── Build command ────────────────────────────────────────────
RUN_ARGS="--model $MODEL --device cuda --wandb-project patience-degradation"

# ── Print job info ───────────────────────────────────────────
echo "=========================================="
echo "Phase 5: Safety-Critical Evaluations"
echo "Model:   $MODEL"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="

srun python3 scripts/experiments/phase5_safety_evaluations.py $RUN_ARGS

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results:   results/phase5_safety_evaluations/"
echo "=========================================="

exit $EXIT_CODE
