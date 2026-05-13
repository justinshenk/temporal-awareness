#!/bin/bash
# overnight_a6000.sh — comprehensive overnight runs for the 48 GB A6000 instance.
#
# Designed for ~10-12 hours wall time. Five tiers, priority-ordered.
# Each job is independently resumable (skipped if output JSON exists).
# A single job failure does NOT kill the script — we move on.
#
# Expected coverage if it all finishes:
#   - Gemma-2-2b: ALL 4 domains (within-model discriminator proof)
#   - Gemma-2-9b: rhyme + qa_neutral (size scaling)
#   - Gemma-2-2b-it: rhyme + qa_neutral (base vs instruct comparison)
#   - Qwen3 small: 3 domains (architecture diversity)
#   - Pythia-2.8b: code + trivia (negative anchors)
#   - Pythia-1b: code (workshop reproduction)
#
# At the end: aggregates everything written into MASTER_TABLE.csv + SUMMARY.md
#
# Just run:  nohup bash overnight_a6000.sh > overnight.log 2>&1 &
#            disown
# Then  tail -f overnight.log  to watch.

set -uo pipefail
cd /workspace/temporal-awareness
mkdir -p logs results/v2

# Sanity-check HF auth (gated models will silently fail without this)
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. Gated models (Gemma, Llama) will fail."
    echo "         Re-export it: export HF_TOKEN=hf_..."
fi

# Common args for every run_staircase_v2.py invocation
COMMON_ARGS="--output_dir results/v2 \
             --quantization bf16 \
             --probe_types linear \
             --ablation zero,mean \
             --n_boot 500"

# ──────────────────────────────────────────────────────────────────────
# Job runner — skips finished jobs, logs each one separately, never fatal
# ──────────────────────────────────────────────────────────────────────
run_job() {
    local model=$1
    local domain=$2
    local layer_mode=${3:-maar_range}
    local slug=$(echo "$model" | tr / _)
    local out="results/v2/${slug}__${domain}__staircase.json"
    local log="logs/${slug}__${domain}.log"

    if [ -f "$out" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP (exists): $model × $domain"
        return 0
    fi

    echo ""
    echo "=========================================================="
    echo "[$(date '+%H:%M:%S')] START: $model × $domain  ($layer_mode)"
    echo "=========================================================="

    python3 scripts/lookahead/experiments/run_staircase_v2.py \
        --model "$model" \
        --domain "$domain" \
        --layer_mode "$layer_mode" \
        $COMMON_ARGS 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}

    if [ "$rc" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ DONE:  $model × $domain"
    else
        echo "[$(date '+%H:%M:%S')] ✗ FAIL (rc=$rc):  $model × $domain  — moving on"
    fi
    return 0  # never propagate failure
}

T0=$(date +%s)
echo "==================================================="
echo "Overnight run started at $(date)"
echo "==================================================="

# ─────────────────────────────────────────────────────────────────────
# TIER 1 — Within-model discriminator proof (~2 hr)
# Gemma-2-2b on the 3 domains we don't have yet. We already have rhyme.
# This finishes the most important within-model figure for the paper.
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "### TIER 1: Gemma-2-2b across all domains (within-model proof)"
run_job google/gemma-2-2b trivia
run_job google/gemma-2-2b qa_suggestive
run_job google/gemma-2-2b qa_neutral

# ─────────────────────────────────────────────────────────────────────
# TIER 2 — Base vs Instruct on the same architecture (~1.5 hr)
# Tests if instruction-tuning changes the staircase signature.
# Same gating as Gemma-2-2b, so should auth without re-approval.
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "### TIER 2: Gemma-2-2b-it (base→instruct comparison)"
run_job google/gemma-2-2b-it rhyme
run_job google/gemma-2-2b-it qa_neutral

# ─────────────────────────────────────────────────────────────────────
# TIER 3 — Negative anchors on non-gated workshop models (~45 min)
# Pythia for code (workshop reproduction) and trivia (negative control).
# These are small (1-3 GB) and fast (~10-20 min each).
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "### TIER 3: Negative anchors (Pythia, workshop_6 layer sampling)"
run_job EleutherAI/pythia-2.8b-deduped code   workshop_6
run_job EleutherAI/pythia-2.8b-deduped trivia workshop_6
run_job EleutherAI/pythia-1.4b-deduped code   workshop_6
run_job EleutherAI/pythia-1b-deduped   code   workshop_6

# ─────────────────────────────────────────────────────────────────────
# TIER 4 — Architecture diversity: Qwen3 small (~2 hr)
# Qwen3 is not gated, validates the framework outside Gemma family.
# Qwen3-1.7B-Base ≈ 3.5 GB in bf16 — fast.
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "### TIER 4: Architecture diversity (Qwen3 small)"
run_job Qwen/Qwen3-1.7B-Base rhyme
run_job Qwen/Qwen3-1.7B-Base qa_neutral
run_job Qwen/Qwen3-1.7B-Base trivia

# ─────────────────────────────────────────────────────────────────────
# TIER 5 — Size scaling: Gemma-2-9b discriminator pair (~5 hr)
# The big one. 9B model ≈ 18 GB in bf16 — fits comfortably on 48 GB.
# This is the most expensive section but the highest-value remaining
# experiment because it shows the framework scales.
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "### TIER 5: Gemma-2-9b size scaling (discriminator pair)"
run_job google/gemma-2-9b rhyme
run_job google/gemma-2-9b qa_neutral

# ─────────────────────────────────────────────────────────────────────
# TIER 6 — Bonus if time remains (~3-4 hr)
# Gemma-2-9b on the other two domains for full coverage.
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "### TIER 6: Gemma-2-9b full domain coverage (bonus)"
run_job google/gemma-2-9b trivia
run_job google/gemma-2-9b qa_suggestive

# ─────────────────────────────────────────────────────────────────────
# AGGREGATION — runs no matter what tiers finished
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "==================================================="
echo "[$(date '+%H:%M:%S')] Aggregating results"
echo "==================================================="
python3 scripts/lookahead/experiments/analyze_staircase_v2.py \
    --results_dir results/v2 \
    --output_dir results/v2

T1=$(date +%s)
ELAPSED=$((T1 - T0))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))

echo ""
echo "==================================================="
echo "Overnight run complete at $(date)"
echo "Total wall time: ${HOURS}h ${MINS}m"
echo "==================================================="
echo ""
echo "Results in results/v2/:"
ls -la results/v2/*.json 2>/dev/null | wc -l | xargs -I {} echo "  {} JSON results"
ls -la results/v2/*.csv results/v2/*.md 2>/dev/null
echo ""
echo "Quick summary:"
[ -f results/v2/SUMMARY.md ] && head -30 results/v2/SUMMARY.md
