#!/usr/bin/bash
# Submit all three experiments to Sherlock.
# Usage: bash scripts/experiments/submit_all_experiments.sh [quick|full]
#
# Submits: sae_feature_stability, sequential_tracking, patience_degradation

set -euo pipefail

MODE="${1:-quick}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "============================================="
echo "Submitting all experiments (mode: $MODE)"
echo "Project root: $PROJECT_ROOT"
echo "============================================="

# 1. SAE Feature Stability (gemma, gpt2, pythia)
for MODEL in gemma-2-2b gpt2 pythia-70m; do
    echo ""
    echo "--- SAE Feature Stability: $MODEL ---"
    if [ "$MODE" = "quick" ]; then
        sbatch --time=01:00:00 scripts/experiments/train_sae_stability.sh "$MODEL" "$MODE"
    else
        sbatch scripts/experiments/train_sae_stability.sh "$MODEL" "$MODE"
    fi
done

# 2. Sequential Activation Tracking (gemma, gpt2, pythia)
for MODEL in gemma-2-2b gpt2 pythia-70m; do
    echo ""
    echo "--- Sequential Activation Tracking: $MODEL ---"
    if [ "$MODE" = "quick" ]; then
        sbatch --time=02:00:00 scripts/experiments/train_seq_tracking.sh "$MODEL" "$MODE"
    else
        sbatch scripts/experiments/train_seq_tracking.sh "$MODEL" "$MODE"
    fi
done

# 3. Patience Degradation (gemma, gpt2, pythia — with SAEs)
for MODEL in gemma-2-2b gpt2 pythia-70m; do
    echo ""
    echo "--- Patience Degradation: $MODEL ---"
    if [ "$MODE" = "quick" ]; then
        sbatch --time=02:00:00 scripts/experiments/train_patience_deg.sh "$MODEL" "$MODE"
    else
        sbatch scripts/experiments/train_patience_deg.sh "$MODEL" "$MODE"
    fi
done

# 4. Patience Degradation — Qwen2.5-3B-Instruct (activation-only)
echo ""
echo "--- Patience Degradation: Qwen2.5-3B-Instruct ---"
if [ "$MODE" = "quick" ]; then
    sbatch --time=03:00:00 scripts/experiments/train_patience_deg_large.sh "Qwen/Qwen2.5-3B-Instruct" "$MODE"
else
    sbatch scripts/experiments/train_patience_deg_large.sh "Qwen/Qwen2.5-3B-Instruct" "$MODE"
fi

# 5. Patience Degradation — Llama-3.1-8B-Instruct (activation-only)
echo ""
echo "--- Patience Degradation: Llama-3.1-8B-Instruct ---"
if [ "$MODE" = "quick" ]; then
    sbatch --time=04:00:00 scripts/experiments/train_patience_deg_large.sh "meta-llama/Llama-3.1-8B-Instruct" "$MODE"
else
    sbatch scripts/experiments/train_patience_deg_large.sh "meta-llama/Llama-3.1-8B-Instruct" "$MODE"
fi

echo ""
echo "============================================="
echo "All jobs submitted. Check with: squeue -u \$USER"
echo "============================================="
