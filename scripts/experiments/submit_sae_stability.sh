#!/usr/bin/bash
# Submit SAE feature stability experiment to Sherlock SLURM
#
# Usage:
#   bash scripts/experiments/submit_sae_stability.sh              # full run
#   bash scripts/experiments/submit_sae_stability.sh quick         # single layer sanity check

MODE="${1:-full}"

if [ "$MODE" = "quick" ]; then
    JOB_NAME="sae_stability_quick"
    TIME="01:00:00"
    EXTRA_ARGS="--quick"
else
    JOB_NAME="sae_stability_full"
    TIME="06:00:00"
    EXTRA_ARGS=""
fi

OUT_FILE="${JOB_NAME}.%j.out"
ERR_FILE="${JOB_NAME}.%j.err"

# Submit from project root so SLURM_SUBMIT_DIR is set correctly
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

sbatch -J "${JOB_NAME}" -o "${OUT_FILE}" -e "${ERR_FILE}" \
    --time="${TIME}" \
    scripts/experiments/train_sae_stability.sh "$MODE" "$EXTRA_ARGS"
