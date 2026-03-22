#!/bin/bash
# Submit behavioral metrics jobs for all 5 models on Sherlock
#
# Usage:
#   bash scripts/experiments/submit_behavioral_all.sh          # full runs
#   bash scripts/experiments/submit_behavioral_all.sh --quick   # quick validation

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

# GPU memory requirements
declare -A MEM_MAP
MEM_MAP[gpt2]="16G"
MEM_MAP[pythia-70m]="16G"
MEM_MAP[gemma-2-2b]="16G"
MEM_MAP[Qwen2.5-3B-Instruct]="24G"
MEM_MAP[Llama-3.1-8B-Instruct]="32G"

declare -A TIME_MAP
TIME_MAP[gpt2]="4:00:00"
TIME_MAP[pythia-70m]="2:00:00"
TIME_MAP[gemma-2-2b]="8:00:00"
TIME_MAP[Qwen2.5-3B-Instruct]="12:00:00"
TIME_MAP[Llama-3.1-8B-Instruct]="16:00:00"

if [ "$QUICK" -eq 1 ]; then
    # Override times for quick runs
    for m in "${MODELS[@]}"; do
        TIME_MAP[$m]="1:00:00"
    done
fi

echo "Submitting behavioral metrics jobs for ${#MODELS[@]} models..."
echo ""

for model in "${MODELS[@]}"; do
    mem=${MEM_MAP[$model]}
    time=${TIME_MAP[$model]}

    job_id=$(sbatch \
        --job-name="behav-${model}" \
        --mem="$mem" \
        --time="$time" \
        --export=MODEL="$model",QUICK="$QUICK" \
        scripts/experiments/train_behavioral_metrics.sh \
        | awk '{print $4}')

    echo "  $model: job $job_id (mem=$mem, time=$time)"
done

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in: logs/behavioral_*.out"
