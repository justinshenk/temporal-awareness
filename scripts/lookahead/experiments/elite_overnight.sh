#!/bin/bash
# elite_overnight.sh — the "insane game changer" experiments
#
# Runs AFTER full_paper_run.sh has completed (24 base JSONs in results/v2/).
# Adds:
#   Phase A: Complete Gemma-2-27b coverage (trivia + qa_suggestive + code) (~4 hr)
#   Phase B: Pythia-1.4b TRAINING DYNAMICS sweep — 8 checkpoints × code + rhyme (~5 hr)
#            Watch the staircase gap emerge during training. Nobody has done this.
#   Phase C: MLP-probe robustness on 3 headline models × rhyme + qa_neutral (~6 hr)
#   Phase D: Mean-pool baseline backfill on ALL JSONs (~3 hr)
#   Phase E: Regenerate figures + stats + push to GitHub
#
# Total: ~18-20 hours on A100 80GB.
#
# Launch:
#   cd /workspace/temporal-awareness
#   export HF_TOKEN=hf_... && export HF_HOME=/workspace/.hf_home
#   nohup bash scripts/lookahead/experiments/elite_overnight.sh > elite.log 2>&1 &
#   disown

set -uo pipefail
cd /workspace/temporal-awareness
mkdir -p logs results/v2 results/v2/figures

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set"; exit 1
fi

COMMON="--output_dir results/v2 --quantization bf16 --probe_types linear --ablation zero,mean --n_boot 500"

run_job() {
    local model=$1 domain=$2 layer_mode=${3:-maar_range} revision=${4:-}
    local slug=$(echo "$model" | sed 's|/|__|g')
    if [ -n "$revision" ]; then
        slug="${slug}__${revision}"
    fi
    local out="results/v2/${slug}__${domain}__staircase.json"
    local log="logs/${slug}__${domain}.log"

    if [ -f "$out" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP: $model × $domain ${revision:+(rev=$revision)}"
        return 0
    fi

    echo ""
    echo "[$(date '+%H:%M:%S')] START: $model × $domain ${revision:+(rev=$revision)} ($layer_mode)"

    local rev_arg=""
    if [ -n "$revision" ]; then
        rev_arg="--revision $revision"
    fi

    python3 scripts/lookahead/experiments/run_staircase_v2.py \
        --model "$model" --domain "$domain" --layer_mode "$layer_mode" \
        $rev_arg $COMMON 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}

    if [ "$rc" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ DONE: $model × $domain ${revision:+(rev=$revision)}"
    else
        echo "[$(date '+%H:%M:%S')] ✗ FAIL: $model × $domain ${revision:+(rev=$revision)}"
    fi
    return 0
}

T0=$(date +%s)
echo "==================================================="
echo "elite_overnight.sh started at $(date)"
echo "Base JSONs present: $(ls results/v2/*__staircase.json 2>/dev/null | wc -l)"
echo "==================================================="

# ──────────────────────────────────────────────────────────────────────
# PHASE A: Complete Gemma-2-27b (all 5 domains) (~4 hr)
# We already have rhyme + qa_neutral from full_paper_run.sh.
# Add: trivia, qa_suggestive, code.
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE A: Complete Gemma-2-27b to all 5 domains"
run_job google/gemma-2-27b trivia
run_job google/gemma-2-27b qa_suggestive
run_job google/gemma-2-27b code workshop_6

# Clean 27b cache (~54GB)
echo "### Cleaning Gemma-2-27b cache..."
rm -rf /workspace/.hf_home/hub/models--google--gemma-2-27b \
       /workspace/.hf_home/hub/.locks/models--google--gemma-2-27b 2>/dev/null
df -h /workspace | tail -1

# ──────────────────────────────────────────────────────────────────────
# PHASE B: Pythia-1.4b TRAINING DYNAMICS SWEEP (~5 hr)
#
# Run the staircase at 8 checkpoints during training on code + rhyme.
# Checkpoints are log-spaced across 143K training steps.
#
# This is the "insane game changer" experiment:
#   - If the code gap appears SUDDENLY at some step → phase transition
#   - If it appears GRADUALLY → smooth development
#   - If the rhyme gap never appears on Pythia → architecture-specific
#
# Pythia checkpoints live on HuggingFace as revision branches:
#   EleutherAI/pythia-1.4b-deduped, revision="step128000"
#
# Each checkpoint is ~5.3GB. We delete after use.
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE B: Pythia-1.4b training dynamics sweep"

PYTHIA_MODEL="EleutherAI/pythia-1.4b-deduped"
# Log-spaced checkpoints: random init → final
CHECKPOINTS="step0 step512 step4000 step16000 step32000 step64000 step128000 step143000"

for ckpt in $CHECKPOINTS; do
    echo ""
    echo "--- Checkpoint: $ckpt ---"

    # Run code domain (the main training dynamics target)
    run_job "$PYTHIA_MODEL" code workshop_6 "$ckpt"

    # Run rhyme domain (does Pythia develop rhyme planning?)
    run_job "$PYTHIA_MODEL" rhyme maar_range "$ckpt"

    # Clean this checkpoint's cache to save disk
    # HF caches revisions inside the model dir, but cleaning the whole dir is simpler
    # (next checkpoint re-downloads fresh — ~6s on this network)
    rm -rf /workspace/.hf_home/hub/models--EleutherAI--pythia-1.4b-deduped 2>/dev/null
done
echo ""
echo "### Phase B complete: $(ls results/v2/EleutherAI__pythia-1.4b-deduped__step*__*__staircase.json 2>/dev/null | wc -l) checkpoint JSONs"
df -h /workspace | tail -1

# ──────────────────────────────────────────────────────────────────────
# PHASE C: MLP-probe robustness on 3 headline models (~6 hr)
#
# Hewitt & Liang (2019) showed linear probes can find "phantom" signal.
# MLP probe that agrees with linear probe → finding is robust.
# We run both probes in one pass via --probe_types linear,mlp.
#
# The JSON will contain headlines for BOTH probe types.
# Separate output file (_mlp suffix) to not overwrite linear-only.
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE C: MLP-probe robustness"

MLP_COMMON="--output_dir results/v2 --quantization bf16 --probe_types linear,mlp --ablation zero,mean --n_boot 500"

for model in google/gemma-2-2b google/gemma-2-9b Qwen/Qwen3-8B-Base; do
    for domain in rhyme qa_neutral; do
        slug=$(echo "$model" | sed 's|/|__|g')
        out="results/v2/${slug}__${domain}__mlp__staircase.json"
        log="logs/${slug}__${domain}_mlp.log"

        if [ -f "$out" ]; then
            echo "[$(date '+%H:%M:%S')] SKIP MLP: $model × $domain"
            continue
        fi

        echo ""
        echo "[$(date '+%H:%M:%S')] START MLP: $model × $domain"

        # Run with both probe types; write to _mlp suffix output
        python3 -c "
import sys, os, json, time
sys.path.insert(0, os.getcwd())
os.environ['MAAR_DATA_ROOT'] = 'data/maar_supplementary_material'
# Monkey-patch the output path to use _mlp suffix
import scripts.lookahead.experiments.run_staircase_v2 as runner
orig_main = runner.main
args = runner.build_argparser().parse_args([
    '--model', '$model', '--domain', '$domain',
    '--layer_mode', 'maar_range',
    '--output_dir', 'results/v2',
    '--quantization', 'bf16',
    '--probe_types', 'linear,mlp',
    '--ablation', 'zero,mean',
    '--n_boot', '500',
])
runner.setup_logging('INFO')
from pathlib import Path
out_path = Path('$out')
out_path.parent.mkdir(parents=True, exist_ok=True)
t0 = time.time()
try:
    doc = runner.run(args)
    doc['meta']['total_seconds'] = round(time.time() - t0, 2)
    doc['meta']['mlp_run'] = True
    with open(out_path, 'w') as f:
        json.dump(doc, f, indent=2, default=str)
    print(f'✓ Wrote {out_path}')
except Exception as e:
    import traceback; traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee "$log"

        echo "[$(date '+%H:%M:%S')] DONE MLP: $model × $domain"
    done

    # Clean model cache between different models
    cache_slug=$(echo "$model" | sed 's|/|--|g')
    rm -rf "/workspace/.hf_home/hub/models--${cache_slug}" 2>/dev/null
done
df -h /workspace | tail -1

# ──────────────────────────────────────────────────────────────────────
# PHASE D: Mean-pool baseline backfill on ALL JSONs (~3 hr)
# Re-downloads each model, extracts activations, computes workshop
# baseline, appends to JSON. Enables fig3 (dual-baseline scatter).
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE D: Mean-pool baseline backfill"
python3 scripts/lookahead/experiments/patch_meanpool_baseline.py \
    --results_dir results/v2 \
    --maar_data_root data/maar_supplementary_material \
    --force \
    2>&1 | tee logs/phase_d_backfill.log

# ──────────────────────────────────────────────────────────────────────
# PHASE E: Regenerate figures + stats + push
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE E: Figures + stats + push"

# Aggregate
python3 scripts/lookahead/experiments/analyze_staircase_v2.py \
    --results_dir results/v2 --output_dir results/v2 \
    2>&1 | tee logs/phase_e_analyze.log

# Figures
python3 scripts/lookahead/experiments/make_paper_figures.py \
    --results_dir results/v2 --anchor_model google/gemma-2-2b \
    2>&1 | tee logs/phase_e_figures.log

# Push to GitHub
git add results/v2/ 2>/dev/null
git commit -m "Elite overnight: 27B full coverage + training dynamics + MLP + dual baselines

Additions over full_paper_run.sh:
  - Gemma-2-27b × trivia/qa_suggestive/code (complete 5-domain coverage at 27B)
  - Pythia-1.4b × 8 training checkpoints × code + rhyme (training dynamics)
  - MLP-probe robustness on Gemma-2-2b/9b + Qwen3-8B × rhyme + qa_neutral
  - Mean-pool baseline backfill on all JSONs (enables dual-baseline figure)
  - Regenerated figures with full data

Total JSONs: $(ls results/v2/*__staircase.json 2>/dev/null | wc -l)
Total checkpoint JSONs: $(ls results/v2/*step*__staircase.json 2>/dev/null | wc -l)" 2>&1 | tail -5
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
echo "elite_overnight.sh COMPLETE at $(date)"
echo "Wall time: ${HOURS}h ${MINS}m"
echo "==================================================="
echo ""
echo "JSON inventory:"
echo "  Base experiments: $(ls results/v2/*__staircase.json 2>/dev/null | grep -v step | grep -v mlp | wc -l)"
echo "  Checkpoint sweep: $(ls results/v2/*step*__staircase.json 2>/dev/null | wc -l)"
echo "  MLP probes:       $(ls results/v2/*mlp*__staircase.json 2>/dev/null | wc -l)"
echo "  Total:            $(ls results/v2/*__staircase.json 2>/dev/null | wc -l)"
echo ""
echo "Figures:"
ls results/v2/figures/*.pdf 2>/dev/null
echo ""
echo "Summary:"
[ -f results/v2/SUMMARY.md ] && head -30 results/v2/SUMMARY.md
