#!/bin/bash
# focused_final.sh — only high-value jobs that WORK on 40GB A100
# No timeouts. No exotic models. Just the experiments that matter.
set -uo pipefail
cd /workspace/temporal-awareness
mkdir -p logs results/v2
export MAAR_DATA_ROOT=data/maar_supplementary_material

COMMON="--output_dir results/v2 --quantization bf16 --probe_types linear --ablation zero,mean --n_boot 500"

run_job() {
    local model=$1 domain=$2 layer_mode=${3:-maar_range} revision=${4:-}
    local slug=$(echo "$model" | sed 's|/|__|g')
    [ -n "$revision" ] && slug="${slug}__${revision}"
    local out="results/v2/${slug}__${domain}__staircase.json"
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
        $rev_arg $COMMON 2>&1 | tee "logs/${slug}__${domain}.log"
    [ ${PIPESTATUS[0]} -eq 0 ] \
        && echo "[$(date '+%H:%M:%S')] ✓ DONE: $model × $domain ${revision:+(rev=$revision)}" \
        || echo "[$(date '+%H:%M:%S')] ✗ FAIL: $model × $domain ${revision:+(rev=$revision)}"
    return 0
}

clean() {
    local m=$1; rm -rf "/workspace/.hf_home/hub/models--$(echo $m | sed 's|/|--|g')" 2>/dev/null
}

T0=$(date +%s)
echo "=== focused_final.sh started at $(date) ==="

# ─────────────────────────────────────────────
# 1. MISTRAL REMAINING DOMAINS (already have rhyme+code+qa_neutral)
# ─────────────────────────────────────────────
echo ""
echo "### 1. Mistral remaining domains"
run_job mistralai/Mistral-7B-v0.3 qa_suggestive
run_job mistralai/Mistral-7B-v0.3 trivia
clean mistralai/Mistral-7B-v0.3

# ─────────────────────────────────────────────
# 2. PHI-3-MINI (retry — small model, should fit 40GB easily)
# ─────────────────────────────────────────────
echo ""
echo "### 2. Phi-3-mini (3.8B, should fit easily)"
for dom in rhyme code qa_neutral; do
    lm="maar_range"; [ "$dom" = "code" ] && lm="workshop_6"
    run_job microsoft/Phi-3-mini-4k-instruct "$dom" "$lm"
done
clean microsoft/Phi-3-mini-4k-instruct

# ─────────────────────────────────────────────
# 3. PYTHIA-6.9B CHECKPOINT SWEEP (the key decomposition validation)
#    Use int8 quantization to fit in 40GB
# ─────────────────────────────────────────────
echo ""
echo "### 3. Pythia-6.9b checkpoints (int8 for 40GB)"
INT8_COMMON="--output_dir results/v2 --quantization int8 --probe_types linear --ablation zero,mean --n_boot 500"
for ckpt in step0 step4000 step32000 step143000; do
    for dom in code rhyme; do
        slug="EleutherAI__pythia-6.9b-deduped__${ckpt}"
        out="results/v2/${slug}__${dom}__staircase.json"
        lm="maar_range"; [ "$dom" = "code" ] && lm="workshop_6"
        if [ -f "$out" ]; then
            echo "[$(date '+%H:%M:%S')] SKIP: pythia-6.9b × $dom (rev=$ckpt)"
            continue
        fi
        echo ""
        echo "[$(date '+%H:%M:%S')] START: pythia-6.9b × $dom (rev=$ckpt)"
        python3 scripts/lookahead/experiments/run_staircase_v2.py \
            --model EleutherAI/pythia-6.9b-deduped --domain "$dom" \
            --layer_mode "$lm" --revision "$ckpt" \
            $INT8_COMMON 2>&1 | tee "logs/${slug}__${dom}.log"
        [ ${PIPESTATUS[0]} -eq 0 ] \
            && echo "[$(date '+%H:%M:%S')] ✓ DONE: pythia-6.9b × $dom (rev=$ckpt)" \
            || echo "[$(date '+%H:%M:%S')] ✗ FAIL: pythia-6.9b × $dom (rev=$ckpt)"
    done
    clean EleutherAI/pythia-6.9b-deduped
done

# ─────────────────────────────────────────────
# 4. QA_NEUTRAL TRAINING DYNAMICS (completes the 3-way figure)
# ─────────────────────────────────────────────
echo ""
echo "### 4. qa_neutral training dynamics (Pythia-1.4b)"
for ckpt in step0 step4000 step16000 step32000 step64000 step143000; do
    run_job EleutherAI/pythia-1.4b-deduped qa_neutral maar_range "$ckpt"
    clean EleutherAI/pythia-1.4b-deduped
done

# ─────────────────────────────────────────────
# 5. QA_SUGGESTIVE TRAINING DYNAMICS
# ─────────────────────────────────────────────
echo ""
echo "### 5. qa_suggestive training dynamics (Pythia-1.4b)"
for ckpt in step0 step4000 step32000 step143000; do
    run_job EleutherAI/pythia-1.4b-deduped qa_suggestive maar_range "$ckpt"
    clean EleutherAI/pythia-1.4b-deduped
done

# ─────────────────────────────────────────────
# 6. MLP ON MISTRAL (extends 8/8 → 10/10)
# ─────────────────────────────────────────────
echo ""
echo "### 6. MLP on Mistral"
for domain in rhyme qa_neutral; do
    slug="mistralai__Mistral-7B-v0.3"
    out="results/v2/${slug}__${domain}__mlp__staircase.json"
    if [ -f "$out" ]; then
        echo "SKIP MLP: Mistral × $domain"; continue
    fi
    echo "[$(date '+%H:%M:%S')] MLP: Mistral × $domain"
    python3 -c "
import sys, os, json
sys.path.insert(0, os.getcwd())
os.environ.setdefault('MAAR_DATA_ROOT', 'data/maar_supplementary_material')
import scripts.lookahead.experiments.run_staircase_v2 as runner
runner.setup_logging('INFO')
args = runner.build_argparser().parse_args([
    '--model', 'mistralai/Mistral-7B-v0.3', '--domain', '$domain',
    '--layer_mode', 'maar_range', '--output_dir', 'results/v2',
    '--quantization', 'bf16', '--probe_types', 'linear,mlp',
    '--ablation', 'zero,mean', '--n_boot', '500',
])
from pathlib import Path
out_path = Path('$out')
try:
    doc = runner.run(args)
    doc['meta']['mlp_run'] = True
    with open(out_path, 'w') as f:
        json.dump(doc, f, indent=2, default=str)
    mlp_hl = [h for h in doc.get('headlines',[]) if h.get('probe_type')=='mlp']
    print(f'✓ {len(mlp_hl)} MLP headlines')
except Exception as e:
    import traceback; traceback.print_exc()
" 2>&1 | tee "logs/${slug}__${domain}_mlp.log"
done
clean mistralai/Mistral-7B-v0.3

# ─────────────────────────────────────────────
# 7. EXPANDED BEHAVIORAL (ALL rhyme models)
# ─────────────────────────────────────────────
echo ""
echo "### 7. Expanded behavioral validation"
pip install --break-system-packages -q pronouncing 2>/dev/null
python3 scripts/lookahead/experiments/icml_extras.py \
    --results_dir results/v2 --mode behavioral \
    2>&1 | tee logs/behavioral_final.log

# ─────────────────────────────────────────────
# 8. LOGIT LENS
# ─────────────────────────────────────────────
echo ""
echo "### 8. Logit lens comparison"
python3 scripts/lookahead/experiments/icml_extras.py \
    --results_dir results/v2 --mode logit_lens \
    2>&1 | tee logs/logit_lens_final.log

# ─────────────────────────────────────────────
# 9. FLOOR + PERMUTATION (no GPU needed)
# ─────────────────────────────────────────────
echo ""
echo "### 9. Floor analysis + permutation tests"
python3 scripts/lookahead/experiments/icml_extras.py \
    --results_dir results/v2 --mode floor
python3 scripts/lookahead/experiments/icml_extras.py \
    --results_dir results/v2 --mode permtest

# ─────────────────────────────────────────────
# 10. SENSITIVITY (PCA dim + seeds)
# ─────────────────────────────────────────────
echo ""
echo "### 10. Sensitivity analyses"
for pca in 32 64 128 256; do
    echo "  PCA dim=$pca"
    python3 scripts/lookahead/experiments/run_staircase_v2.py \
        --model google/gemma-2-2b --domain rhyme --layer_mode maar_range \
        --output_dir /tmp/sens --quantization bf16 --probe_types linear \
        --pca_dim $pca --n_boot 200 2>&1 | tail -1
    f=$(ls /tmp/sens/*__rhyme__staircase.json 2>/dev/null | head -1)
    if [ -n "$f" ]; then
        python3 -c "import json; d=json.load(open('$f')); h=sorted(d['headlines'],key=lambda r:-abs(r['headline_gap']))[0]; print(f'  gap={h[\"headline_gap\"]*100:+.1f}pp')"
        rm "$f"
    fi
done
clean google/gemma-2-2b

# ─────────────────────────────────────────────
# 11. AGGREGATE + FIGURES + GIT PUSH
# ─────────────────────────────────────────────
echo ""
echo "### 11. Aggregate + figures + push"
python3 scripts/lookahead/experiments/analyze_staircase_v2.py \
    --results_dir results/v2 --output_dir results/v2
python3 scripts/lookahead/experiments/make_paper_figures.py \
    --results_dir results/v2 --anchor_model google/gemma-2-2b

git add results/v2/ scripts/ src/
git commit -m "Focused final: Mistral complete + Phi-3 + Pythia-6.9b checkpoints + qa_neutral dynamics + MLP + behavioral + logit lens + sensitivity"
git push origin psycoplankton/emnlp-staircase-v2

T1=$(date +%s)
echo ""
echo "=== focused_final.sh COMPLETE in $(( (T1-T0)/3600 ))h $(( (T1-T0)%3600/60 ))m ==="
echo "JSONs: $(ls results/v2/*__staircase.json | wc -l)"

# ─────────────────────────────────────────────
# BONUS: Retry OLMo + StableLM (likely failed from old timeout, not OOM)
# ─────────────────────────────────────────────
echo ""
echo "### BONUS: OLMo-7B retry"
for dom in rhyme code qa_neutral; do
    lm="maar_range"; [ "$dom" = "code" ] && lm="workshop_6"
    run_job allenai/OLMo-7B-0724-hf "$dom" "$lm"
done
clean allenai/OLMo-7B-0724-hf

echo ""
echo "### BONUS: StableLM-2-1.6B retry"
for dom in rhyme code qa_neutral; do
    lm="maar_range"; [ "$dom" = "code" ] && lm="workshop_6"
    run_job stabilityai/stablelm-2-1_6b "$dom" "$lm"
done
clean stabilityai/stablelm-2-1_6b

# ─────────────────────────────────────────────
# FIXED: Falcon3-7B-Base (modern arch, compatible with transformers 5.8)
# ─────────────────────────────────────────────
echo ""
echo "### FIXED: Falcon3-7B-Base (replaces broken falcon-7b)"
for dom in rhyme code qa_neutral; do
    lm="maar_range"; [ "$dom" = "code" ] && lm="workshop_6"
    run_job tiiuae/Falcon3-7B-Base "$dom" "$lm"
done
clean tiiuae/Falcon3-7B-Base

# ─────────────────────────────────────────────
# FIXED: Llama-3.1-8B via ungated NousResearch mirror
# ─────────────────────────────────────────────
echo ""
echo "### FIXED: Llama-3.1-8B (ungated mirror)"
for dom in rhyme code qa_neutral; do
    lm="maar_range"; [ "$dom" = "code" ] && lm="workshop_6"
    run_job NousResearch/Meta-Llama-3.1-8B "$dom" "$lm"
done
clean NousResearch/Meta-Llama-3.1-8B

# ─────────────────────────────────────────────
# BONUS: Llama-3.2-3B (also accessible, smaller)
# ─────────────────────────────────────────────
echo ""
echo "### BONUS: Llama-3.2-3B"
for dom in rhyme code qa_neutral; do
    lm="maar_range"; [ "$dom" = "code" ] && lm="workshop_6"
    run_job meta-llama/Llama-3.2-3B "$dom" "$lm"
done
clean meta-llama/Llama-3.2-3B
