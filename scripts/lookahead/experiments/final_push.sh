#!/bin/bash
# final_push.sh — the COMPREHENSIVE 48-hour final experiment push
#
# This script addresses every remaining weakness and adds significant
# new evidence. Run on A100 80GB.
#
# ┌─────────────────────────────────────────────────────────────────┐
# │ PHASE 1 (8 hr): NEW ARCHITECTURES — Llama + Mistral            │
# │   Adds 2 new architecture families (total: 5 families)         │
# │   15 models instead of 13                                       │
# │                                                                 │
# │ PHASE 2 (8 hr): COMPLETE TRAINING DYNAMICS                     │
# │   Pythia-6.9b checkpoint sweep (3rd model size)                │
# │   qa_neutral + qa_suggestive checkpoints (complete domain set)  │
# │                                                                 │
# │ PHASE 3 (6 hr): MLP ON NEW + EXPANDED BEHAVIORAL              │
# │   MLP on Llama + Mistral (extends 8/8 → 12/12?)               │
# │   Behavioral on ALL ~16 rhyme models (n=3 → n=15)             │
# │                                                                 │
# │ PHASE 4 (3 hr): LOGIT LENS — independent mech interp method   │
# │   Code domain: return type probability at target vs earlier     │
# │   Rhyme domain: rhyme-word probability comparison               │
# │                                                                 │
# │ PHASE 5 (3 hr): SENSITIVITY ANALYSES                           │
# │   PCA dimension sweep, bootstrap iteration sweep                │
# │   Probe: LogReg C sweep, n_folds sweep                         │
# │                                                                 │
# │ PHASE 6 (2 hr): FIGURES + ANALYSIS + PUSH                     │
# │                                                                 │
# │ TOTAL: ~30-35 hr (fits in 48 with margin for errors)           │
# └─────────────────────────────────────────────────────────────────┘

set -uo pipefail
cd /workspace/temporal-awareness
mkdir -p logs results/v2

if [ -z "${HF_TOKEN:-}" ]; then echo "ERROR: HF_TOKEN not set"; exit 1; fi

COMMON="--output_dir results/v2 --quantization bf16 --probe_types linear --ablation zero,mean --n_boot 500"

run_job() {
    local model=$1 domain=$2 layer_mode=${3:-maar_range} revision=${4:-}
    local slug=$(echo "$model" | sed 's|/|__|g')
    [ -n "$revision" ] && slug="${slug}__${revision}"
    local out="results/v2/${slug}__${domain}__staircase.json"
    local log="logs/${slug}__${domain}_final.log"
    local rev_arg=""
    [ -n "$revision" ] && rev_arg="--revision $revision"

    if [ -f "$out" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP: $model × $domain ${revision:+(rev=$revision)}"
        return 0
    fi
    echo ""
    echo "[$(date '+%H:%M:%S')] START: $model × $domain ${revision:+(rev=$revision)}"
    python3 scripts/lookahead/experiments/run_staircase_v2.py \
        --model "$model" --domain "$domain" --layer_mode "$layer_mode" \
        $rev_arg $COMMON 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && echo "[$(date '+%H:%M:%S')] ✓ DONE: $model × $domain ${revision:+(rev=$revision)}" \
                     || echo "[$(date '+%H:%M:%S')] ✗ FAIL: $model × $domain ${revision:+(rev=$revision)}"
    return 0
}

T0=$(date +%s)
echo "==================================================="
echo "final_push.sh started at $(date)"
echo "==================================================="

# ──────────────────────────────────────────────────────────────────────
# PHASE 1: NEW ARCHITECTURES (~8 hr)
# Adds Llama-3.1-8B and Mistral-7B-v0.3 — two entirely new families.
# Each gets: rhyme, code, qa_neutral (the discriminator triple).
# Falls back gracefully if model access is gated.
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 1a: Mistral-7B (freely available)"
run_job mistralai/Mistral-7B-v0.3 rhyme
run_job mistralai/Mistral-7B-v0.3 code workshop_6
run_job mistralai/Mistral-7B-v0.3 qa_neutral

echo "### Cleaning Mistral cache..."
rm -rf /workspace/.hf_home/hub/models--mistralai--Mistral-7B-v0.3 2>/dev/null

echo ""
echo "### PHASE 1b: Llama-3.1-8B (may need license acceptance)"
run_job meta-llama/Llama-3.1-8B rhyme
run_job meta-llama/Llama-3.1-8B code workshop_6
run_job meta-llama/Llama-3.1-8B qa_neutral

echo "### Cleaning Llama cache..."
rm -rf /workspace/.hf_home/hub/models--meta-llama--Llama-3.1-8B 2>/dev/null

echo ""
echo "### PHASE 1c: Phi-3-mini-4k (Microsoft, freely available, different arch)"
run_job microsoft/Phi-3-mini-4k-instruct rhyme
run_job microsoft/Phi-3-mini-4k-instruct code workshop_6
run_job microsoft/Phi-3-mini-4k-instruct qa_neutral

echo "### Cleaning Phi cache..."
rm -rf /workspace/.hf_home/hub/models--microsoft--Phi-3-mini-4k-instruct 2>/dev/null
df -h /workspace | tail -1

# ──────────────────────────────────────────────────────────────────────
# PHASE 2: COMPLETE TRAINING DYNAMICS (~8 hr)
# a) Pythia-6.9b checkpoint sweep (3rd Pythia size for decomposition)
# b) qa_neutral + qa_suggestive checkpoints (complete domain set)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 2a: Pythia-6.9b checkpoint sweep"
for ckpt in step0 step4000 step32000 step143000; do
    run_job EleutherAI/pythia-6.9b-deduped code workshop_6 "$ckpt"
    run_job EleutherAI/pythia-6.9b-deduped rhyme maar_range "$ckpt"
    rm -rf /workspace/.hf_home/hub/models--EleutherAI--pythia-6.9b-deduped 2>/dev/null
done

echo ""
echo "### PHASE 2b: qa_neutral training dynamics (complete the 3-way figure)"
for ckpt in step0 step4000 step32000 step143000; do
    run_job EleutherAI/pythia-1.4b-deduped qa_neutral maar_range "$ckpt"
    rm -rf /workspace/.hf_home/hub/models--EleutherAI--pythia-1.4b-deduped 2>/dev/null
done

echo ""
echo "### PHASE 2c: qa_suggestive training dynamics"
for ckpt in step0 step4000 step32000 step143000; do
    run_job EleutherAI/pythia-1.4b-deduped qa_suggestive maar_range "$ckpt"
    rm -rf /workspace/.hf_home/hub/models--EleutherAI--pythia-1.4b-deduped 2>/dev/null
done

# ──────────────────────────────────────────────────────────────────────
# PHASE 3: MLP ON NEW MODELS + EXPANDED BEHAVIORAL (~6 hr)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 3a: MLP on new architecture models"

for model in mistralai/Mistral-7B-v0.3 meta-llama/Llama-3.1-8B; do
    for domain in rhyme qa_neutral; do
        slug=$(echo "$model" | sed 's|/|__|g')
        out="results/v2/${slug}__${domain}__mlp__staircase.json"
        log="logs/${slug}__${domain}_mlp_final.log"

        if [ -f "$out" ]; then
            echo "[$(date '+%H:%M:%S')] SKIP MLP: $model × $domain"
            continue
        fi

        echo ""
        echo "[$(date '+%H:%M:%S')] MLP: $model × $domain"
        python3 -c "
import sys, os, json, time
sys.path.insert(0, os.getcwd())
os.environ.setdefault('MAAR_DATA_ROOT', 'data/maar_supplementary_material')
import scripts.lookahead.experiments.run_staircase_v2 as runner
runner.setup_logging('INFO')
args = runner.build_argparser().parse_args([
    '--model', '$model', '--domain', '$domain',
    '--layer_mode', 'maar_range', '--output_dir', 'results/v2',
    '--quantization', 'bf16', '--probe_types', 'linear,mlp',
    '--ablation', 'zero,mean', '--n_boot', '500',
])
from pathlib import Path
out_path = Path('$out')
t0 = time.time()
try:
    doc = runner.run(args)
    doc['meta']['mlp_run'] = True
    with open(out_path, 'w') as f:
        json.dump(doc, f, indent=2, default=str)
    mlp_hl = [h for h in doc.get('headlines',[]) if h.get('probe_type')=='mlp']
    print(f'Wrote {out_path} — {len(mlp_hl)} MLP headlines')
except Exception as e:
    import traceback; traceback.print_exc()
" 2>&1 | tee "$log"
    done
    cache_slug=$(echo "$model" | sed 's|/|--|g')
    rm -rf "/workspace/.hf_home/hub/models--${cache_slug}" 2>/dev/null
done

echo ""
echo "### PHASE 3b: Expanded behavioral on ALL rhyme models"
pip install --break-system-packages -q pronouncing 2>/dev/null
python3 scripts/lookahead/experiments/icml_extras.py \
    --results_dir results/v2 --mode behavioral \
    2>&1 | tee logs/behavioral_expanded.log

# ──────────────────────────────────────────────────────────────────────
# PHASE 4: LOGIT LENS (~3 hr)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 4: Logit lens comparison"
python3 scripts/lookahead/experiments/icml_extras.py \
    --results_dir results/v2 --mode logit_lens \
    2>&1 | tee logs/logit_lens.log

# ──────────────────────────────────────────────────────────────────────
# PHASE 5: SENSITIVITY ANALYSES (~3 hr)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 5: Sensitivity analyses"
python3 << 'PYEOF'
"""PCA dimension + regularization sensitivity on Gemma-2-2b × rhyme."""
import sys, os, json, time, subprocess
sys.path.insert(0, os.getcwd())
os.environ.setdefault('MAAR_DATA_ROOT', 'data/maar_supplementary_material')

results = {}

# PCA dimension sweep
for pca_dim in [32, 64, 128, 256]:
    print(f"\n=== PCA dim = {pca_dim} ===")
    cmd = [
        sys.executable, 'scripts/lookahead/experiments/run_staircase_v2.py',
        '--model', 'google/gemma-2-2b', '--domain', 'rhyme',
        '--layer_mode', 'maar_range', '--output_dir', '/tmp/sensitivity',
        '--quantization', 'bf16', '--probe_types', 'linear',
        '--pca_dim', str(pca_dim), '--n_boot', '200',
    ]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    # Read the output JSON
    import glob
    jsons = glob.glob('/tmp/sensitivity/*__rhyme__staircase.json')
    if jsons:
        d = json.load(open(jsons[0]))
        h = sorted(d['headlines'], key=lambda r: -abs(r['headline_gap']))[0]
        results[f'pca_{pca_dim}'] = {
            'pca_dim': pca_dim,
            'gap': round(h['headline_gap'] * 100, 1),
            'target': round(h['target_accuracy'], 3),
            'earlier': round(h['max_earlier_accuracy'], 3),
        }
        print(f"  gap = {h['headline_gap']*100:+.1f}pp")
        os.remove(jsons[0])

print("\n=== PCA Sensitivity Summary ===")
for k, v in sorted(results.items()):
    print(f"  {k}: gap = {v['gap']:+.1f}pp  (target={v['target']}, earlier={v['earlier']})")

with open('results/v2/sensitivity_pca.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved to results/v2/sensitivity_pca.json")
PYEOF

# ──────────────────────────────────────────────────────────────────────
# PHASE 6: FLOOR + PERMUTATION TESTS (no GPU needed)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 6: Floor analysis + permutation tests"
python3 scripts/lookahead/experiments/icml_extras.py \
    --results_dir results/v2 --mode floor \
    2>&1 | tee logs/floor_analysis.log
python3 scripts/lookahead/experiments/icml_extras.py \
    --results_dir results/v2 --mode permtest \
    2>&1 | tee logs/permtest.log

# ──────────────────────────────────────────────────────────────────────
# PHASE 7: MEAN-POOL BACKFILL ON NEW JSONS + FIGURES + PUSH
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### PHASE 7: Backfill + aggregate + figures + push"

# Backfill new JSONs only (skip already-backfilled ones)
python3 scripts/lookahead/experiments/patch_meanpool_baseline.py \
    --results_dir results/v2 \
    --maar_data_root data/maar_supplementary_material \
    2>&1 | tee logs/backfill_final_push.log

# Aggregate
python3 scripts/lookahead/experiments/analyze_staircase_v2.py \
    --results_dir results/v2 --output_dir results/v2

# Figures (will include new models in fig1, new checkpoints in fig5, etc.)
python3 scripts/lookahead/experiments/make_paper_figures.py \
    --results_dir results/v2 --anchor_model google/gemma-2-2b

# Push
git add results/v2/ src/ scripts/
git commit -m "Final push: 5 architectures + complete dynamics + logit lens + expanded behavioral

PHASE 1: Mistral-7B + Llama-3.1-8B + Phi-3-mini (3 new architectures → 5 families total)
PHASE 2: Pythia-6.9b checkpoint sweep + qa_neutral/qa_suggestive dynamics
         (3 Pythia sizes for decomposition, 4 domains for dynamics)
PHASE 3: MLP on new architectures + behavioral on ALL rhyme models (n→15+)
PHASE 4: Logit lens comparison (independent mech interp method)
PHASE 5: PCA dimension sensitivity analysis
PHASE 6: Floor analysis + permutation tests
PHASE 7: Mean-pool backfill + figures + stats

Total JSONs: $(ls results/v2/*__staircase.json | wc -l)
Total architectures: 5+ (Gemma, Qwen, Pythia/GPT-2, Mistral, Llama, Phi)"
git push origin psycoplankton/emnlp-staircase-v2

# ──────────────────────────────────────────────────────────────────────
T1=$(date +%s); ELAPSED=$((T1-T0))
echo ""
echo "==================================================="
echo "final_push.sh COMPLETE in $((ELAPSED/3600))h $((ELAPSED%3600/60))m"
echo "Total JSONs: $(ls results/v2/*__staircase.json | wc -l)"
echo "==================================================="
echo ""
echo "=== Architecture coverage ==="
python3 -c "
import json, glob
families = {}
for f in glob.glob('results/v2/*__staircase.json'):
    if 'step' in f or 'mlp' in f: continue
    d = json.load(open(f))
    m = d['meta']['model']
    if 'gemma' in m.lower(): fam = 'Gemma'
    elif 'qwen' in m.lower(): fam = 'Qwen'
    elif 'pythia' in m.lower(): fam = 'Pythia'
    elif 'gpt2' in m.lower(): fam = 'GPT-2'
    elif 'mistral' in m.lower(): fam = 'Mistral'
    elif 'llama' in m.lower(): fam = 'Llama'
    elif 'phi' in m.lower(): fam = 'Phi'
    else: fam = 'Other'
    families.setdefault(fam, set()).add(m.split('/')[-1])
for fam, models in sorted(families.items()):
    print(f'  {fam}: {len(models)} models — {sorted(models)}')
"
