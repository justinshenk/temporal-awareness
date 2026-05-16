#!/bin/bash
# paired_and_mlp.sh — fill paired coverage gaps + MLP robustness
#
# After this script, every model has BOTH rhyme and code results,
# enabling clean paired Wilcoxon tests. Plus MLP probe on 4 key models.
#
# Phase 1: Rhyme on 7 Pythia/GPT-2 models (~2.5 hr)
# Phase 2: Code on 4 Gemma/Qwen models (~2.5 hr)
# Phase 3: MLP probe on 4 models × rhyme + qa_neutral (~4 hr, fixed MLPProbe)
# Phase 4: Regenerate figures + stats + push
#
# Total: ~9-10 hours.

set -uo pipefail
cd /workspace/temporal-awareness
mkdir -p logs results/v2

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set"; exit 1
fi

COMMON="--output_dir results/v2 --quantization bf16 --probe_types linear --ablation zero,mean --n_boot 500"

run_job() {
    local model=$1 domain=$2 layer_mode=${3:-maar_range}
    local slug=$(echo "$model" | sed 's|/|__|g')
    local out="results/v2/${slug}__${domain}__staircase.json"
    local log="logs/${slug}__${domain}_paired.log"

    if [ -f "$out" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP: $model × $domain"
        return 0
    fi

    echo ""
    echo "[$(date '+%H:%M:%S')] START: $model × $domain ($layer_mode)"
    python3 scripts/lookahead/experiments/run_staircase_v2.py \
        --model "$model" --domain "$domain" --layer_mode "$layer_mode" \
        $COMMON 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ DONE: $model × $domain"
    else
        echo "[$(date '+%H:%M:%S')] ✗ FAIL: $model × $domain"
    fi
    return 0
}

T0=$(date +%s)
echo "==================================================="
echo "paired_and_mlp.sh started at $(date)"
echo "==================================================="

# ──────────────────────────────────────────────────────────────────────
# PHASE 1: Rhyme on Pythia/GPT-2 (fills paired coverage gap) (~2.5 hr)
# These models already have code results. Adding rhyme gives us
# 7 new paired (rhyme, code) observations for Wilcoxon tests.
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 1: Rhyme on Pythia/GPT-2 (paired coverage)"
run_job EleutherAI/pythia-410m-deduped rhyme maar_range
run_job EleutherAI/pythia-1b-deduped   rhyme maar_range
run_job EleutherAI/pythia-1.4b-deduped rhyme maar_range
run_job EleutherAI/pythia-2.8b-deduped rhyme maar_range
run_job gpt2                            rhyme maar_range
run_job gpt2-medium                     rhyme maar_range
run_job gpt2-xl                         rhyme maar_range

# Clean small model caches
echo "### Cleaning Pythia/GPT-2 caches..."
for m in EleutherAI--pythia-410m-deduped EleutherAI--pythia-1b-deduped \
         EleutherAI--pythia-1.4b-deduped EleutherAI--pythia-2.8b-deduped \
         gpt2 gpt2-medium gpt2-xl; do
    rm -rf /workspace/.hf_home/hub/models--${m} 2>/dev/null
done

# ──────────────────────────────────────────────────────────────────────
# PHASE 2: Code on Gemma/Qwen (fills paired coverage gap) (~2.5 hr)
# These models already have rhyme results. Adding code gives us
# 4 more paired observations (+ Gemma-2-27b already has both = 12 total).
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 2: Code on Gemma/Qwen (paired coverage)"
run_job google/gemma-2-2b      code workshop_6
run_job google/gemma-2-9b      code workshop_6
run_job Qwen/Qwen3-1.7B-Base   code workshop_6
run_job Qwen/Qwen3-8B-Base     code workshop_6

# Clean medium model caches before MLP
echo "### Cleaning Gemma/Qwen caches..."
for m in google--gemma-2-9b Qwen--Qwen3-1.7B-Base Qwen--Qwen3-8B-Base; do
    rm -rf /workspace/.hf_home/hub/models--${m} 2>/dev/null
done

# ──────────────────────────────────────────────────────────────────────
# PHASE 3: MLP probe on 4 models × rhyme + qa_neutral (~4 hr)
# Uses fixed MLPProbe (now inherits from BaseEstimator).
# Output goes to separate _mlp JSONs to preserve linear-only results.
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 3: MLP probe robustness (fixed MLPProbe)"

MLP_ARGS="--output_dir results/v2 --quantization bf16 --probe_types linear,mlp --ablation zero,mean --n_boot 500"

for model in google/gemma-2-2b google/gemma-2-9b Qwen/Qwen3-1.7B-Base Qwen/Qwen3-8B-Base; do
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

        # Run with both probes; write to _mlp suffix
        python3 -c "
import sys, os, json, time
sys.path.insert(0, os.getcwd())
os.environ.setdefault('MAAR_DATA_ROOT', 'data/maar_supplementary_material')

import scripts.lookahead.experiments.run_staircase_v2 as runner
runner.setup_logging('INFO')

args = runner.build_argparser().parse_args([
    '--model', '$model', '--domain', '$domain',
    '--layer_mode', 'maar_range',
    '--output_dir', 'results/v2',
    '--quantization', 'bf16',
    '--probe_types', 'linear,mlp',
    '--ablation', 'zero,mean',
    '--n_boot', '500',
])

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

    # Clean each model cache after its MLP runs
    cache_slug=$(echo "$model" | sed 's|/|--|g')
    rm -rf "/workspace/.hf_home/hub/models--${cache_slug}" 2>/dev/null
done

# ──────────────────────────────────────────────────────────────────────
# PHASE 4: Mean-pool backfill on NEW JSONs only + regen figures + push
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 4: Backfill new JSONs + figures + push"

# Backfill only JSONs that don't have mean_pool yet
python3 scripts/lookahead/experiments/patch_meanpool_baseline.py \
    --results_dir results/v2 \
    --maar_data_root data/maar_supplementary_material \
    2>&1 | tee logs/backfill_paired.log

# Aggregate + figures
python3 scripts/lookahead/experiments/analyze_staircase_v2.py \
    --results_dir results/v2 --output_dir results/v2

python3 scripts/lookahead/experiments/make_paper_figures.py \
    --results_dir results/v2 --anchor_model google/gemma-2-2b

# Push
git add results/v2/ src/lookahead/probing/mlp_probe.py scripts/
git commit -m "Paired coverage + MLP probes: comprehensive statistical power

Phase 1: Rhyme on 7 Pythia/GPT-2 models (paired with existing code results)
Phase 2: Code on 4 Gemma/Qwen models (paired with existing rhyme results)
Phase 3: MLP probe on 4 models × rhyme + qa_neutral (fixed sklearn compat)
Phase 4: Mean-pool backfill + figures + stats

Now have 12+ models with both rhyme AND code results for paired Wilcoxon.
MLP probe confirms linear probe findings (or flags discrepancies)."
git push origin psycoplankton/emnlp-staircase-v2

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
T1=$(date +%s)
ELAPSED=$((T1 - T0))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))

echo ""
echo "==================================================="
echo "paired_and_mlp.sh COMPLETE at $(date)"
echo "Wall: ${HOURS}h ${MINS}m"
echo "==================================================="
echo ""
echo "=== Paired coverage check ==="
python3 -c "
import json, glob
models_rhyme = set()
models_code = set()
for f in glob.glob('results/v2/*__staircase.json'):
    if 'step' in f or 'mlp' in f: continue
    d = json.load(open(f))
    m = d['meta']['model'].split('/')[-1]
    dom = d['meta']['domain']
    if dom == 'rhyme': models_rhyme.add(m)
    if dom == 'code': models_code.add(m)
paired = models_rhyme & models_code
print(f'Models with rhyme: {len(models_rhyme)}')
print(f'Models with code:  {len(models_code)}')
print(f'Paired (both):     {len(paired)}')
print(f'Paired models: {sorted(paired)}')
"

echo ""
echo "=== MLP results ==="
ls results/v2/*mlp*__staircase.json 2>/dev/null | wc -l | xargs -I {} echo "{} MLP JSONs"

echo ""
echo "=== Updated stats ==="
cat results/v2/figures/STATS.md
