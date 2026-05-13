#!/bin/bash
# fixup_a6000.sh — clean reruns after both bug fixes, plus the Gemma-2-9b
#                  tier that never started before the original overnight got tangled.
#
# Phases (sequential, resume-safe):
#   1. qa_neutral reruns on the 3 models we have (~2 hr)
#      — uses paired CV fix (StratifiedGroupKFold grouped by question text)
#   2. code domain on small workshop-anchor models (~2 hr)
#      — uses workshop's untyped DATASET_500 (no annotation leak)
#   3. Qwen3-1.7B × trivia (the one Tier-4 job that got cut short)
#   4. Gemma-2-9b × {rhyme, qa_neutral} (~7-8 hr)
#      — size-scaling on the discriminator pair, the missing Tier 5
#
# Launch:
#   cd /workspace/temporal-awareness
#   git pull origin psycoplankton/emnlp-staircase-v2     # ensures dfa6414+ is local
#   chmod +x scripts/lookahead/experiments/fixup_a6000.sh
#   nohup bash scripts/lookahead/experiments/fixup_a6000.sh > fixup.log 2>&1 &
#   disown
#
# Estimated wall time: ~12 hours.

set -uo pipefail
cd /workspace/temporal-awareness
mkdir -p logs results/v2

# Sanity-check HF auth
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. Gemma will fail without it."
fi

# Common runner args
COMMON_ARGS="--output_dir results/v2 \
             --quantization bf16 \
             --probe_types linear \
             --ablation zero,mean \
             --n_boot 500"

# Delete the qa_neutral JSONs we KNOW are bad (CV bug, fixed in c8fa7f3)
echo "[$(date '+%H:%M:%S')] Removing 3 broken qa_neutral JSONs (CV-bug rerun)..."
for slug in google__gemma-2-2b google__gemma-2-2b-it Qwen__Qwen3-1.7B-Base; do
    rm -fv results/v2/${slug}__qa_neutral__staircase.json
done

# ──────────────────────────────────────────────────────────────────────
# Job runner: idempotent, never fatal on a single failure.
# Uses DOUBLE-underscore slug to match Python's model_slug() exactly,
# so log files and JSON outputs line up.
# ──────────────────────────────────────────────────────────────────────
run_job() {
    local model=$1
    local domain=$2
    local layer_mode=${3:-maar_range}
    # NOTE: Python's model_slug uses "/"→"__" (double); match it here.
    local slug=$(echo "$model" | sed 's|/|__|g')
    local out="results/v2/${slug}__${domain}__staircase.json"
    local log="logs/${slug}__${domain}_fixup.log"

    if [ -f "$out" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP (exists): $model × $domain"
        return 0
    fi

    echo ""
    echo "=========================================================="
    echo "[$(date '+%H:%M:%S')] START: $model × $domain  ($layer_mode)"
    echo "=========================================================="

    python3 scripts/lookahead/experiments/run_staircase_v2.py \
        --model "$model" --domain "$domain" --layer_mode "$layer_mode" \
        $COMMON_ARGS 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}

    if [ "$rc" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ DONE:  $model × $domain"
    else
        echo "[$(date '+%H:%M:%S')] ✗ FAIL (rc=$rc):  $model × $domain  — moving on"
    fi
    return 0
}

T0=$(date +%s)
echo "==================================================="
echo "Fixup run started at $(date)"
echo "==================================================="

# ─────────────────────────────────────────────────────────────────────
# PHASE 1: qa_neutral reruns with paired CV (~2 hr)
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 1: qa_neutral reruns (paired StratifiedGroupKFold)"
run_job google/gemma-2-2b      qa_neutral
run_job google/gemma-2-2b-it   qa_neutral
run_job Qwen/Qwen3-1.7B-Base   qa_neutral

# ─────────────────────────────────────────────────────────────────────
# PHASE 2: code domain on small models — the workshop anchor (~2 hr)
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 2: code domain on workshop-scale models (untyped DATASET_500)"
run_job EleutherAI/pythia-410m-deduped code workshop_6
run_job EleutherAI/pythia-1b-deduped   code workshop_6
run_job EleutherAI/pythia-1.4b-deduped code workshop_6
run_job EleutherAI/pythia-2.8b-deduped code workshop_6
run_job gpt2                            code workshop_6
run_job gpt2-medium                     code workshop_6
run_job gpt2-xl                         code workshop_6

# ─────────────────────────────────────────────────────────────────────
# PHASE 3: Missing Qwen3-1.7B trivia (Tier 4 was cut short)
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 3: Missing Qwen3 trivia"
run_job Qwen/Qwen3-1.7B-Base trivia

# ─────────────────────────────────────────────────────────────────────
# PHASE 4: Gemma-2-9b size scaling on the discriminator pair (~7-8 hr)
# Uses paired CV automatically for qa_neutral via the new fix.
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 4: Gemma-2-9b size scaling (the missing Tier 5)"
run_job google/gemma-2-9b rhyme
run_job google/gemma-2-9b qa_neutral
# Bonus if time:
run_job google/gemma-2-9b trivia
run_job google/gemma-2-9b qa_suggestive

# ─────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "==================================================="
echo "[$(date '+%H:%M:%S')] Aggregating results"
echo "==================================================="
python3 scripts/lookahead/experiments/analyze_staircase_v2.py \
    --results_dir results/v2 --output_dir results/v2

T1=$(date +%s)
ELAPSED=$((T1 - T0))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))

echo ""
echo "==================================================="
echo "Fixup run complete at $(date)"
echo "Total wall: ${HOURS}h ${MINS}m"
echo "==================================================="
ls results/v2/*__staircase.json 2>/dev/null | wc -l | xargs -I {} echo "  {} JSON results in results/v2/"
ls -la results/v2/*.csv results/v2/*.md 2>/dev/null
echo ""
[ -f results/v2/SUMMARY.md ] && head -25 results/v2/SUMMARY.md
