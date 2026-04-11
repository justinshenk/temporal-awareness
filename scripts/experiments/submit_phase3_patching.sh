#!/usr/bin/bash
# ── Per-model SLURM submission for Phase 3 Exp 3: Causal Patching ──
# IMPORTANT: Run refusal experiment first — this loads saved .npy directions.
#
# Usage (submit one job per model):
#   sbatch scripts/experiments/submit_phase3_patching.sh Llama-3.1-8B-Instruct
#   sbatch scripts/experiments/submit_phase3_patching.sh Qwen3-8B
#   sbatch scripts/experiments/submit_phase3_patching.sh Qwen3-30B-A3B
#   sbatch scripts/experiments/submit_phase3_patching.sh DeepSeek-R1-Distill-Qwen-7B
#   sbatch scripts/experiments/submit_phase3_patching.sh Llama-3.1-8B

#SBATCH --job-name=p3-patching
#SBATCH --time=04:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=80GB
#SBATCH --output=logs/phase3_patching_%j.out
#SBATCH --error=logs/phase3_patching_%j.err

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
RUN_ARGS="--model $MODEL --device cuda --wandb-project patience-degradation"

if [ "$MODE" = "quick" ]; then
    RUN_ARGS="$RUN_ARGS --quick"
fi

# ── Print job info ───────────────────────────────────────────
echo "=========================================="
echo "Phase 3, Exp 3: Causal Activation Patching"
echo "Model:   $MODEL"
echo "Mode:    $MODE"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="

srun python3 scripts/experiments/phase3_causal_patching.py $RUN_ARGS

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results:   results/phase3_causal_patching/"
echo "=========================================="

exit $EXIT_CODE
