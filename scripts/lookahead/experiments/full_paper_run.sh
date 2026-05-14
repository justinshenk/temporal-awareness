#!/bin/bash
# full_paper_run.sh — COMPLETE from-scratch pipeline for EMNLP paper.
#
# Runs on a FRESH A100 80GB instance with repo cloned + Maar data in place.
# Produces ALL experiment JSONs, backfills dual baselines, generates figures,
# and pushes everything to GitHub.
#
# Phases (sequential, resume-safe via skip-if-exists):
#   1. Code anchor: 7 small Pythia/GPT-2 × code (~1.5 hr)
#   2. Small Maar: Qwen3-1.7B × rhyme/qa_neutral/trivia (~1.5 hr)
#   3. Medium Maar: Gemma-2-2b × 4 domains + 2b-it × 2 domains (~3 hr)
#   4. 9B scaling: Gemma-2-9b × 4 domains (~6 hr)
#   5. 8B diversity: Qwen3-8B-Base × rhyme/qa_neutral (~4 hr)
#   6. 27B scaling: Gemma-2-27b × rhyme/qa_neutral (~6 hr)  [A100-only]
#   7. Backfill: mean-pool baseline + grouped BoW on all JSONs (~1.5 hr)
#   8. Analyze + figures + stats (~10 min)
#   9. Git push all results
#
# Total: ~24-28 hours.  Cost: ~$40-47 at $1.70/hr A100.
#
# Launch:
#   cd /workspace/temporal-awareness
#   export HF_TOKEN=hf_...
#   export HF_HOME=/workspace/.hf_home
#   nohup bash scripts/lookahead/experiments/full_paper_run.sh > full_run.log 2>&1 &
#   disown

set -uo pipefail
cd /workspace/temporal-awareness
mkdir -p logs results/v2 results/v2/figures

echo "==================================================="
echo "full_paper_run.sh started at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Disk: $(df -h /workspace | tail -1)"
echo "==================================================="

# ──────────────────────────────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Export it first."; exit 1
fi
python3 -c "
import sys; sys.path.insert(0,'.')
from src.lookahead.datasets.code_untyped import load_code_untyped
from src.lookahead.datasets.maar_data import load_maar_rhyme, load_maar_qa_neutral
assert len(load_code_untyped()) > 0, 'code dataset empty'
assert len(load_maar_rhyme()) > 0, 'rhyme dataset empty'
assert len(load_maar_qa_neutral()) > 0, 'qa_neutral dataset empty'
print('✓ All datasets verified')
" || { echo "Dataset verification failed"; exit 1; }

COMMON="--output_dir results/v2 --quantization bf16 --probe_types linear --ablation zero,mean --n_boot 500"

# ──────────────────────────────────────────────────────────────────────
# Job runner (idempotent, never fatal)
# ──────────────────────────────────────────────────────────────────────
run_job() {
    local model=$1 domain=$2 layer_mode=${3:-maar_range}
    local slug=$(echo "$model" | sed 's|/|__|g')
    local out="results/v2/${slug}__${domain}__staircase.json"
    local log="logs/${slug}__${domain}.log"

    if [ -f "$out" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP: $model × $domain"
        return 0
    fi

    echo ""
    echo "=========================================================="
    echo "[$(date '+%H:%M:%S')] START: $model × $domain ($layer_mode)"
    echo "=========================================================="

    python3 scripts/lookahead/experiments/run_staircase_v2.py \
        --model "$model" --domain "$domain" --layer_mode "$layer_mode" \
        $COMMON 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}

    if [ "$rc" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ DONE: $model × $domain"
    else
        echo "[$(date '+%H:%M:%S')] ✗ FAIL (rc=$rc): $model × $domain"
    fi
    return 0
}

T0=$(date +%s)

# ──────────────────────────────────────────────────────────────────────
# PHASE 1: Code anchor — 7 small models × code (~1.5 hr)
# Workshop's untyped DATASET_500. Expected: +8-15pp weak positive gap.
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 1: Code anchor (7 small models × code)"
run_job EleutherAI/pythia-410m-deduped code workshop_6
run_job EleutherAI/pythia-1b-deduped   code workshop_6
run_job EleutherAI/pythia-1.4b-deduped code workshop_6
run_job EleutherAI/pythia-2.8b-deduped code workshop_6
run_job gpt2                            code workshop_6
run_job gpt2-medium                     code workshop_6
run_job gpt2-xl                         code workshop_6

# Clean small model caches (~22GB freed)
echo "### Cleaning Pythia + GPT-2 caches..."
for m in EleutherAI--pythia-410m-deduped EleutherAI--pythia-1b-deduped \
         EleutherAI--pythia-1.4b-deduped EleutherAI--pythia-2.8b-deduped \
         gpt2 gpt2-medium gpt2-xl; do
    rm -rf /workspace/.hf_home/hub/models--${m} /workspace/.hf_home/hub/.locks/models--${m} 2>/dev/null
done
df -h /workspace | tail -1

# ──────────────────────────────────────────────────────────────────────
# PHASE 2: Small Maar — Qwen3-1.7B-Base × 3 domains (~1.5 hr)
# Architecture diversity at small scale (non-Gemma, non-gated).
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 2: Qwen3-1.7B-Base × Maar domains"
run_job Qwen/Qwen3-1.7B-Base rhyme
run_job Qwen/Qwen3-1.7B-Base qa_neutral
run_job Qwen/Qwen3-1.7B-Base trivia

# ──────────────────────────────────────────────────────────────────────
# PHASE 3: Medium Maar — Gemma-2-2b family × domains (~3 hr)
# Within-model discriminator proof + base-vs-instruct.
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 3: Gemma-2-2b family"
run_job google/gemma-2-2b rhyme
run_job google/gemma-2-2b trivia
run_job google/gemma-2-2b qa_suggestive
run_job google/gemma-2-2b qa_neutral
run_job google/gemma-2-2b-it rhyme
run_job google/gemma-2-2b-it qa_neutral

# Clean Gemma-2-2b caches (~10GB freed) before loading 9b
echo "### Cleaning Gemma-2-2b caches..."
rm -rf /workspace/.hf_home/hub/models--google--gemma-2-2b \
       /workspace/.hf_home/hub/models--google--gemma-2-2b-it \
       /workspace/.hf_home/hub/.locks/models--google--gemma-2-2b \
       /workspace/.hf_home/hub/.locks/models--google--gemma-2-2b-it 2>/dev/null

# Also clean Qwen3-1.7B (~3.5GB)
rm -rf /workspace/.hf_home/hub/models--Qwen--Qwen3-1.7B-Base \
       /workspace/.hf_home/hub/.locks/models--Qwen--Qwen3-1.7B-Base 2>/dev/null
df -h /workspace | tail -1

# ──────────────────────────────────────────────────────────────────────
# PHASE 4: 9B scaling — Gemma-2-9b × 4 domains (~6 hr)
# Size-invariance test: rhyme gap should match 2B (~+72pp).
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 4: Gemma-2-9b × all 4 domains"
run_job google/gemma-2-9b rhyme
run_job google/gemma-2-9b qa_neutral
run_job google/gemma-2-9b trivia
run_job google/gemma-2-9b qa_suggestive

# Keep Gemma-2-9b cached for now (Phase 7 backfill might use it)

# ──────────────────────────────────────────────────────────────────────
# PHASE 5: 8B architecture diversity — Qwen3-8B-Base (~4 hr)
# Non-Gemma, non-gated, 8B scale. Closes architecture objection.
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 5: Qwen3-8B-Base × discriminator pair"
run_job Qwen/Qwen3-8B-Base rhyme
run_job Qwen/Qwen3-8B-Base qa_neutral

# Clean Qwen3-8B cache (~16GB) before loading 27b
echo "### Cleaning Qwen3-8B cache..."
rm -rf /workspace/.hf_home/hub/models--Qwen--Qwen3-8B-Base \
       /workspace/.hf_home/hub/.locks/models--Qwen--Qwen3-8B-Base 2>/dev/null

# Also clean Gemma-2-9b now (~18GB) to make room for 27b
rm -rf /workspace/.hf_home/hub/models--google--gemma-2-9b \
       /workspace/.hf_home/hub/.locks/models--google--gemma-2-9b 2>/dev/null
df -h /workspace | tail -1

# ──────────────────────────────────────────────────────────────────────
# PHASE 6: 27B scaling — Gemma-2-27b × discriminator pair (~6 hr)
# A100-ONLY. This is the flagship big-model result.
# bf16: 54GB VRAM → fits on 80GB A100 but NOT on 48GB A6000.
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 6: Gemma-2-27b × discriminator pair (A100-only)"
run_job google/gemma-2-27b rhyme
run_job google/gemma-2-27b qa_neutral

# Clean 27b cache (~54GB)
echo "### Cleaning Gemma-2-27b cache..."
rm -rf /workspace/.hf_home/hub/models--google--gemma-2-27b \
       /workspace/.hf_home/hub/.locks/models--google--gemma-2-27b 2>/dev/null
df -h /workspace | tail -1

# ──────────────────────────────────────────────────────────────────────
# PHASE 7: Backfill mean-pool baseline on all JSONs (~1.5 hr)
# Re-extracts activations (cheap), computes workshop-style mean-pool
# baseline + grouped BoW for qa_neutral. Appends to each JSON.
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 7: Mean-pool baseline backfill"
python3 scripts/lookahead/experiments/patch_meanpool_baseline.py \
    --results_dir results/v2 \
    --maar_data_root data/maar_supplementary_material \
    2>&1 | tee logs/phase7_backfill.log

# ──────────────────────────────────────────────────────────────────────
# PHASE 8: Analyze + figures + stats (~10 min)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 8: Analysis + figures"
python3 scripts/lookahead/experiments/analyze_staircase_v2.py \
    --results_dir results/v2 --output_dir results/v2 \
    2>&1 | tee logs/phase8_analyze.log

python3 scripts/lookahead/experiments/make_paper_figures.py \
    --results_dir results/v2 --anchor_model google/gemma-2-2b \
    2>&1 | tee logs/phase8_figures.log

# ──────────────────────────────────────────────────────────────────────
# PHASE 9: Push results to GitHub
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 9: Pushing results to GitHub"
cd /workspace/temporal-awareness
git add results/v2/*.json results/v2/*.csv results/v2/*.md results/v2/figures/ 2>/dev/null
git commit -m "Full paper results: $(ls results/v2/*__staircase.json | wc -l) JSONs + figures

Models: $(ls results/v2/*__staircase.json | sed 's|.*v2/||;s|__.*||' | sort -u | tr '\n' ', ')
Domains: code, rhyme, qa_neutral, qa_suggestive, trivia
Includes: dual baselines (mean-pool + max-across-earlier),
  bootstrap CIs, ablation drops, pre-registration checks.
Figures: fig1-4.pdf + STATS.md

Generated by full_paper_run.sh on A100-SXM4-80GB." 2>&1 | tail -5
git push origin psycoplankton/emnlp-staircase-v2 2>&1 | tail -3

# ──────────────────────────────────────────────────────────────────────
# DONE
# ──────────────────────────────────────────────────────────────────────
T1=$(date +%s)
ELAPSED=$((T1 - T0))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))

echo ""
echo "==================================================="
echo "full_paper_run.sh COMPLETE at $(date)"
echo "Wall time: ${HOURS}h ${MINS}m"
echo "==================================================="
echo ""
echo "Results:"
ls results/v2/*__staircase.json | wc -l | xargs -I {} echo "  {} experiment JSONs"
echo ""
echo "Figures:"
ls results/v2/figures/*.pdf 2>/dev/null
echo ""
echo "Tables:"
ls results/v2/*.csv results/v2/*.md 2>/dev/null
echo ""
echo "Quick summary:"
[ -f results/v2/SUMMARY.md ] && head -40 results/v2/SUMMARY.md
echo ""
echo "Stats:"
[ -f results/v2/figures/STATS.md ] && cat results/v2/figures/STATS.md
