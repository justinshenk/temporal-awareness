#!/bin/bash
# Submit behavioral metrics jobs for all 5 models on Sherlock
#
# Usage:
#   bash scripts/experiments/submit_behavioral_all.sh          # full runs (500 examples, reps 1-100)
#   bash scripts/experiments/submit_behavioral_all.sh --quick   # quick validation (20 examples)

set -euo pipefail

QUICK=0
if [ "${1:-}" = "--quick" ]; then
    QUICK=1
    echo "=== QUICK VALIDATION MODE ==="
fi

mkdir -p logs

MODELS=(
    "gpt2"
    "pythia-70m"
    "gemma-2-2b"
    "Qwen2.5-3B-Instruct"
    "Llama-3.1-8B-Instruct"
)

# Host memory per model
declare -A MEM_MAP
MEM_MAP[gpt2]="32G"
MEM_MAP[pythia-70m]="32G"
MEM_MAP[gemma-2-2b]="32G"
MEM_MAP[Qwen2.5-3B-Instruct]="48G"
MEM_MAP[Llama-3.1-8B-Instruct]="64G"

# Time limits (extended for 500 examples × 10 rep levels including 30/50/100)
declare -A TIME_MAP
TIME_MAP[gpt2]="12:00:00"
TIME_MAP[pythia-70m]="8:00:00"
TIME_MAP[gemma-2-2b]="18:00:00"
TIME_MAP[Qwen2.5-3B-Instruct]="24:00:00"
TIME_MAP[Llama-3.1-8B-Instruct]="24:00:00"

# GPU constraints per model (ensure enough VRAM)
# gpt2/pythia: ~1-2GB VRAM, any GPU works
# gemma-2-2b: ~5GB fp32 weights + activations at rep 100, need 24GB
# Qwen 3B: ~7GB fp16, need 24GB for long seqs at rep 100
# Llama 8B: ~16GB fp16 weights + activations for long seqs at rep 100, need 40GB+
declare -A GPU_CONSTRAINT
GPU_CONSTRAINT[gpt2]=""
GPU_CONSTRAINT[pythia-70m]=""
GPU_CONSTRAINT[gemma-2-2b]="GPU_MEM:24GB"
GPU_CONSTRAINT[Qwen2.5-3B-Instruct]="GPU_MEM:24GB"
GPU_CONSTRAINT[Llama-3.1-8B-Instruct]="GPU_MEM:40GB"

if [ "$QUICK" -eq 1 ]; then
    for m in "${MODELS[@]}"; do
        TIME_MAP[$m]="1:00:00"
    done
fi

echo "Submitting behavioral metrics jobs for ${#MODELS[@]} models..."
echo ""

for model in "${MODELS[@]}"; do
    mem=${MEM_MAP[$model]}
    time=${TIME_MAP[$model]}
    constraint=${GPU_CONSTRAINT[$model]}

    # Build sbatch command
    SBATCH_CMD="sbatch"
    SBATCH_CMD="$SBATCH_CMD --job-name=behav-${model}"
    SBATCH_CMD="$SBATCH_CMD --mem=$mem"
    SBATCH_CMD="$SBATCH_CMD --time=$time"
    SBATCH_CMD="$SBATCH_CMD --export=MODEL=$model,QUICK=$QUICK"

    # Add GPU constraint only for models that need it
    if [ -n "$constraint" ]; then
        SBATCH_CMD="$SBATCH_CMD -C $constraint"
    fi

    SBATCH_CMD="$SBATCH_CMD scripts/experiments/train_behavioral_metrics.sh"

    job_id=$(eval $SBATCH_CMD | awk '{print $4}')

    if [ -n "$constraint" ]; then
        echo "  $model: job $job_id (mem=$mem, time=$time, gpu=$constraint)"
    else
        echo "  $model: job $job_id (mem=$mem, time=$time, gpu=any)"
    fi
done

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in: logs/behavioral_*.out"
echo "Results in: results/behavioral_metrics/"
