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

# 1. SAE Feature Stability
echo ""
echo "--- SAE Feature Stability ---"
if [ "$MODE" = "quick" ]; then
    sbatch --time=01:00:00 scripts/experiments/train_sae_stability.sh "$MODE" "--quick"
else
    sbatch scripts/experiments/train_sae_stability.sh "$MODE"
fi

# 2. Sequential Activation Tracking
echo ""
echo "--- Sequential Activation Tracking ---"
if [ "$MODE" = "quick" ]; then
    sbatch --time=02:00:00 scripts/experiments/train_seq_tracking.sh "$MODE"
else
    sbatch scripts/experiments/train_seq_tracking.sh "$MODE"
fi

# 3. Patience Degradation
echo ""
echo "--- Patience & Compliance Degradation ---"
if [ "$MODE" = "quick" ]; then
    sbatch --time=02:00:00 scripts/experiments/train_patience_deg.sh "$MODE"
else
    sbatch scripts/experiments/train_patience_deg.sh "$MODE"
fi

echo ""
echo "============================================="
echo "All jobs submitted. Check with: squeue -u \$USER"
echo "============================================="
