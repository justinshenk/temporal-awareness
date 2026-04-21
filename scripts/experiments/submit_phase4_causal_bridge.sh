#!/usr/bin/bash
# ── Per-model SLURM submission for Phase 4: Causal Bridge ──
#
# This is the most compute-intensive experiment: runs probing, generation
# (sycophancy eval), and steering at multiple strengths × layers.
# Allocate extra time vs other experiments.
#
# Usage:
#   sbatch scripts/experiments/submit_phase4_causal_bridge.sh Llama-3.1-8B-Instruct
#   sbatch scripts/experiments/submit_phase4_causal_bridge.sh Qwen3-8B
#   sbatch scripts/experiments/submit_phase4_causal_bridge.sh Qwen3-30B-A3B
#   sbatch scripts/experiments/submit_phase4_causal_bridge.sh DeepSeek-R1-Distill-Qwen-7B
#   sbatch scripts/experiments/submit_phase4_causal_bridge.sh Ouro-2.6B
#   sbatch scripts/experiments/submit_phase4_causal_bridge.sh Llama-3.1-8B
#   sbatch scripts/experiments/submit_phase4_causal_bridge.sh Qwen3-4B-Instruct-2507

#SBATCH --job-name=p4-bridge
#SBATCH --time=06:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=80GB
#SBATCH --output=logs/phase4_causal_bridge_%j.out
#SBATCH --error=logs/phase4_causal_bridge_%j.err

# ── Parse arguments (before env setup so we can switch venvs) ─
MODEL="${1:-Llama-3.1-8B-Instruct}"

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

# ── Build command ────────────────────────────────────────────
RUN_ARGS="--model $MODEL --device cuda --wandb-project patience-degradation"

# ── Print job info ───────────────────────────────────────────
echo "=========================================="
echo "Phase 4: Probe-to-Behavior Causal Bridge"
echo "Model:   $MODEL"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="

srun python3 scripts/experiments/phase4_causal_bridge.py $RUN_ARGS

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results:   results/phase4_causal_bridge/"
echo "=========================================="

exit $EXIT_CODE
