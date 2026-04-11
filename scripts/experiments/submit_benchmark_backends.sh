#!/usr/bin/bash
# ── Per-model SLURM submission for Backend Benchmarking ──
# Compares TransformerLens vs PyTorch activation extraction.
# Requires more memory than typical experiments since both backends
# are loaded sequentially.
#
# Usage (submit one job per model):
#   sbatch scripts/experiments/submit_benchmark_backends.sh Llama-3.1-8B-Instruct
#   sbatch scripts/experiments/submit_benchmark_backends.sh Qwen3-8B
#   sbatch scripts/experiments/submit_benchmark_backends.sh Qwen3-30B-A3B
#   sbatch scripts/experiments/submit_benchmark_backends.sh DeepSeek-R1-Distill-Qwen-7B
#   sbatch scripts/experiments/submit_benchmark_backends.sh Ouro-2.6B
#   sbatch scripts/experiments/submit_benchmark_backends.sh Llama-3.1-8B

#SBATCH --job-name=bench-backends
#SBATCH --time=04:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:32GB
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=80GB
#SBATCH --output=logs/benchmark_backends_%j.out
#SBATCH --error=logs/benchmark_backends_%j.err

# ── Parse arguments (before env setup so we can switch venvs) ─
MODEL="${1:-Llama-3.1-8B-Instruct}"
MODE="${2:-full}"

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

cd "${SLURM_SUBMIT_DIR:-$HOME/temporal-awareness}"
mkdir -p logs

# ── Build command ────────────────────────────────────────────
RUN_ARGS="--model $MODEL --device cuda --wandb-project patience-degradation"

if [ "$MODE" = "quick" ]; then
    RUN_ARGS="$RUN_ARGS --quick"
fi

# ── Print job info ───────────────────────────────────────────
echo "=========================================="
echo "Backend Benchmark: TransformerLens vs PyTorch"
echo "Model:   $MODEL"
echo "Mode:    $MODE"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="

srun python3 scripts/experiments/benchmark_extraction_backends.py $RUN_ARGS

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results:   results/benchmark_backends/"
echo "=========================================="

exit $EXIT_CODE
