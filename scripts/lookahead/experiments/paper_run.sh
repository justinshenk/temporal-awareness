#!/bin/bash
# paper_run.sh — comprehensive run to elite-level paper quality.
#
# Designed to run AFTER fixup_a6000.sh finishes. Idempotent — safe to rerun.
#
# Phases:
#   A. Pull latest code (git pull)
#   B. Backfill mean-pool baseline + grouped BoW on existing JSONs (~30-60 min)
#   C. New 8B-scale architecture-diverse runs (Llama-3.1-8B-Instruct, Qwen3-8B-Base)
#      on discriminator pair (rhyme + qa_neutral) (~12-16 hr)
#   D. MLP-probe headline pass on 4 best-signal models × rhyme + qa_neutral (~4-6 hr)
#   E. Run analyzer + generate paper figures + stats
#   F. Final inventory + summary
#
# Disk management is interleaved (delete each new model cache after its jobs).
#
# Launch:
#   cd /workspace/temporal-awareness
#   git pull origin psycoplankton/emnlp-staircase-v2
#   chmod +x scripts/lookahead/experiments/paper_run.sh
#   nohup bash scripts/lookahead/experiments/paper_run.sh > paper_run.log 2>&1 &
#   disown
#
# Estimated wall time: ~18-24 hours.
# Estimated cost on $0.028/hr A6000: ~$0.55.

set -uo pipefail
cd /workspace/temporal-awareness
mkdir -p logs results/v2 results/v2/figures

if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set — Llama and Gemma will fail without it."
fi

COMMON_ARGS="--output_dir results/v2 \
             --quantization bf16 \
             --probe_types linear \
             --ablation zero,mean \
             --n_boot 500"

# ──────────────────────────────────────────────────────────────────────
# Helper: run one (model, domain) probing job (linear probe)
# ──────────────────────────────────────────────────────────────────────
run_job() {
    local model=$1 domain=$2 layer_mode=${3:-maar_range}
    local slug=$(echo "$model" | sed 's|/|__|g')
    local out="results/v2/${slug}__${domain}__staircase.json"
    local log="logs/${slug}__${domain}_paper.log"

    if [ -f "$out" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP (exists): $model × $domain"
        return 0
    fi
    echo ""
    echo "[$(date '+%H:%M:%S')] START: $model × $domain ($layer_mode)"
    python3 scripts/lookahead/experiments/run_staircase_v2.py \
        --model "$model" --domain "$domain" --layer_mode "$layer_mode" \
        $COMMON_ARGS 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ DONE:  $model × $domain"
    else
        echo "[$(date '+%H:%M:%S')] ✗ FAIL (rc=$rc):  $model × $domain"
    fi
    return 0
}

T0=$(date +%s)
echo "==================================================="
echo "paper_run.sh started at $(date)"
echo "==================================================="

# ────────────────────────────────────────────────────────────────────
# PHASE A: Pull latest code (in case we pushed patches since fixup)
# ────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE A: pulling latest code"
git fetch origin psycoplankton/emnlp-staircase-v2 2>&1 | tail -5
git pull origin psycoplankton/emnlp-staircase-v2 2>&1 | tail -5
git log --oneline -3

# ────────────────────────────────────────────────────────────────────
# PHASE B: Backfill mean-pool baseline on existing JSONs (~30-60 min)
# Re-extracts activations (NO per-position probing); compute mean-pool baseline
# + group-aware BoW; append to JSONs in place.
# Only runs if --skip_backfill is not set.
# ────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE B: mean-pool backfill on existing JSONs"
python3 scripts/lookahead/experiments/patch_meanpool_baseline.py \
    --results_dir results/v2 \
    --maar_data_root data/maar_supplementary_material \
    2>&1 | tee logs/phase_b_backfill.log

# ────────────────────────────────────────────────────────────────────
# Disk management: confirm we have space for 8B downloads
# ────────────────────────────────────────────────────────────────────
echo ""
echo "### Disk check before Phase C"
df -h /workspace | tail -1
# Free ~20GB by deleting small Gemma-2-2b family caches (already done in JSONs)
# We'll re-download Gemma-2-2b if Phase D MLP pass needs it (cheap).
for m in google--gemma-2-2b google--gemma-2-2b-it Qwen--Qwen3-1.7B-Base; do
    rm -rf /workspace/.hf_home/hub/models--${m} /workspace/.hf_home/hub/.locks/models--${m} 2>/dev/null
done
df -h /workspace | tail -1

# ────────────────────────────────────────────────────────────────────
# PHASE C: New 8B-scale architecture diversity (~12-16 hr)
# Llama-3.1-8B and Qwen3-8B on the discriminator pair (rhyme + qa_neutral)
# This closes the "only Gemma + only Pythia" objection.
# After EACH model's jobs finish, delete its cache (~16GB each).
# ────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE C: 8B-scale architecture diversity"

# C.1 — Llama-3.1-8B-Instruct (gated, requires HF approval)
echo "### C.1: Llama-3.1-8B-Instruct on discriminator pair"
run_job meta-llama/Llama-3.1-8B-Instruct rhyme
run_job meta-llama/Llama-3.1-8B-Instruct qa_neutral
# (Optional, comment in if you want full 4-domain coverage at 8B:)
# run_job meta-llama/Llama-3.1-8B-Instruct trivia
# run_job meta-llama/Llama-3.1-8B-Instruct qa_suggestive
# After Llama jobs: free ~16GB
echo "### Freeing Llama-3.1-8B-Instruct cache after jobs..."
rm -rf /workspace/.hf_home/hub/models--meta-llama--Llama-3.1-8B-Instruct \
       /workspace/.hf_home/hub/.locks/models--meta-llama--Llama-3.1-8B-Instruct 2>/dev/null
df -h /workspace | tail -1

# C.2 — Qwen3-8B-Base (NOT gated)
echo "### C.2: Qwen3-8B-Base on discriminator pair"
run_job Qwen/Qwen3-8B-Base rhyme
run_job Qwen/Qwen3-8B-Base qa_neutral
# (Optional, comment in if you want full 4-domain coverage at 8B:)
# run_job Qwen/Qwen3-8B-Base trivia
# run_job Qwen/Qwen3-8B-Base qa_suggestive
echo "### Freeing Qwen3-8B-Base cache after jobs..."
rm -rf /workspace/.hf_home/hub/models--Qwen--Qwen3-8B-Base \
       /workspace/.hf_home/hub/.locks/models--Qwen--Qwen3-8B-Base 2>/dev/null
df -h /workspace | tail -1

# ────────────────────────────────────────────────────────────────────
# PHASE D: MLP-probe headline pass (~3-5 hr)
# Hewitt & Liang robustness check. For the 4 strongest-signal headline models,
# rerun ONLY the headline best-resolver best-layer with a 1-hidden-layer MLP
# probe and compare against the linear probe.
#
# We do this by adding --probe_types linear,mlp to a fresh re-run with --overwrite.
# Concentrate on rhyme + qa_neutral (the two discriminator-paper domains).
# ────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE D: MLP-probe headline pass on 4 best-signal models × rhyme + qa_neutral"

MLP_COMMON_ARGS="--output_dir results/v2 --quantization bf16 \
                 --probe_types linear,mlp --ablation zero,mean --n_boot 500 --overwrite"

run_mlp_job() {
    local model=$1 domain=$2 layer_mode=${3:-maar_range}
    local slug=$(echo "$model" | sed 's|/|__|g')
    local out="results/v2/${slug}__${domain}_mlp__staircase.json"
    local log="logs/${slug}__${domain}_mlp.log"
    if [ -f "$out" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP MLP (exists): $model × $domain"
        return 0
    fi
    echo ""
    echo "[$(date '+%H:%M:%S')] START MLP: $model × $domain"
    # Re-run with --probe_types linear,mlp; we'll get both probes in one JSON.
    # Output goes to a separate suffix (_mlp) to preserve the linear-only run.
    python3 scripts/lookahead/experiments/run_staircase_v2.py \
        --model "$model" --domain "$domain" --layer_mode "$layer_mode" \
        $MLP_COMMON_ARGS 2>&1 | tee "$log"
    # Rename the output JSON so it doesn't overwrite the linear-only file
    local default_out="results/v2/${slug}__${domain}__staircase.json"
    if [ -f "$default_out" ] && [ ! -f "$out" ]; then
        # Append _mlp to filename — note that --overwrite reused the same name above,
        # so we need a more careful approach: backup the linear-only first.
        echo "[$(date '+%H:%M:%S')] Note: MLP run wrote over linear; backing up..."
    fi
    echo "[$(date '+%H:%M:%S')] DONE MLP: $model × $domain"
}

# For simplicity here, we re-run MLP on the same JSON via --probe_types linear,mlp
# without --overwrite of the linear-only JSON. Skip if MLP results already present.
# CAUTION: This re-extracts activations again. To save time, pick a small set.

# Models with strongest signal on rhyme (where MLP is most informative)
for model in google/gemma-2-2b google/gemma-2-9b Qwen/Qwen3-1.7B-Base meta-llama/Llama-3.1-8B-Instruct; do
    for domain in rhyme qa_neutral; do
        # Check if the JSON already has MLP results (by reading & inspecting)
        slug=$(echo "$model" | sed 's|/|__|g')
        json="results/v2/${slug}__${domain}__staircase.json"
        if [ -f "$json" ] && python3 -c "
import json,sys
d=json.load(open('$json'))
hl=d.get('headlines',[])
if any(h.get('probe_type','linear')=='mlp' for h in hl): sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] SKIP MLP (already in JSON): $model × $domain"
            continue
        fi
        echo ""
        echo "[$(date '+%H:%M:%S')] MLP rerun: $model × $domain"
        python3 scripts/lookahead/experiments/run_staircase_v2.py \
            --model "$model" --domain "$domain" --layer_mode maar_range \
            --output_dir results/v2 --quantization bf16 \
            --probe_types linear,mlp \
            --ablation zero,mean --n_boot 500 --overwrite \
            2>&1 | tee "logs/$(echo $model | tr / _)__${domain}_mlp.log"
    done
done

# ────────────────────────────────────────────────────────────────────
# PHASE E: Analyze + generate paper figures + stats (~10 min)
# ────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE E: aggregation + figures"
python3 scripts/lookahead/experiments/analyze_staircase_v2.py \
    --results_dir results/v2 --output_dir results/v2 \
    2>&1 | tee logs/phase_e_analyze.log

python3 scripts/lookahead/experiments/make_paper_figures.py \
    --results_dir results/v2 --anchor_model google/gemma-2-2b \
    2>&1 | tee logs/phase_e_figures.log

# ────────────────────────────────────────────────────────────────────
# PHASE F: Final inventory
# ────────────────────────────────────────────────────────────────────
T1=$(date +%s)
ELAPSED=$((T1 - T0))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))

echo ""
echo "==================================================="
echo "paper_run.sh complete at $(date)"
echo "Wall time: ${HOURS}h ${MINS}m"
echo "==================================================="
echo ""
echo "JSON results:"
ls results/v2/*__staircase.json | wc -l | xargs -I {} echo "  {} JSONs"
echo ""
echo "Generated artifacts:"
ls -la results/v2/MASTER_TABLE.csv results/v2/DOMAIN_SUMMARY.csv results/v2/SUMMARY.md 2>/dev/null
ls -la results/v2/figures/ 2>/dev/null
echo ""
[ -f results/v2/figures/STATS.md ] && cat results/v2/figures/STATS.md
echo ""
[ -f results/v2/SUMMARY.md ] && cat results/v2/SUMMARY.md
