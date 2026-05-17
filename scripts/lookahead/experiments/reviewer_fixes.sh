#!/bin/bash
# reviewer_fixes.sh — addresses ALL major reviewer weaknesses
#
# W3: Rerun qa_neutral on all 6 models with FIXED BoW/ablation/bootstrap
# W4: Pythia-2.8b checkpoint sweep (validates decomposition on 2nd model)
# W5: MLP probe retries on 4 models that had missing headlines
# W6: Behavioral validation (rhyme generation accuracy)
#
# Total: ~10-12 hours on A100.

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
    local log="logs/${slug}__${domain}_fix.log"
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
    [ "$rc" -eq 0 ] && echo "[$(date '+%H:%M:%S')] ✓ DONE: $model × $domain" \
                     || echo "[$(date '+%H:%M:%S')] ✗ FAIL: $model × $domain"
    return 0
}

T0=$(date +%s)
echo "==================================================="
echo "reviewer_fixes.sh started at $(date)"
echo "==================================================="

# ──────────────────────────────────────────────────────────────────────
# W3: Rerun qa_neutral with FIXED grouped BoW/ablation/bootstrap (~4 hr)
# Delete old broken JSONs first, then rerun with --overwrite
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### W3: qa_neutral reruns (grouped BoW + cluster bootstrap + grouped ablation)"
for slug in google__gemma-2-2b google__gemma-2-2b-it google__gemma-2-9b \
            google__gemma-2-27b Qwen__Qwen3-1.7B-Base Qwen__Qwen3-8B-Base; do
    rm -fv results/v2/${slug}__qa_neutral__staircase.json
done

run_job google/gemma-2-2b      qa_neutral
run_job google/gemma-2-2b-it   qa_neutral
run_job google/gemma-2-9b      qa_neutral
run_job google/gemma-2-27b     qa_neutral
run_job Qwen/Qwen3-1.7B-Base   qa_neutral
run_job Qwen/Qwen3-8B-Base     qa_neutral

# ──────────────────────────────────────────────────────────────────────
# W4: Pythia-2.8b checkpoint sweep (validates decomposition) (~2 hr)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### W4: Pythia-2.8b checkpoint sweep"
for ckpt in step0 step4000 step32000 step143000; do
    run_job EleutherAI/pythia-2.8b-deduped code workshop_6 "$ckpt"
    run_job EleutherAI/pythia-2.8b-deduped rhyme maar_range "$ckpt"
    rm -rf /workspace/.hf_home/hub/models--EleutherAI--pythia-2.8b-deduped 2>/dev/null
done

# ──────────────────────────────────────────────────────────────────────
# W5: MLP retries on models that had missing headlines (~3 hr)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### W5: MLP probe retries (fixed MLPProbe + probe_type tagging)"
for model in google/gemma-2-2b google/gemma-2-9b Qwen/Qwen3-8B-Base; do
    for domain in rhyme qa_neutral; do
        slug=$(echo "$model" | sed 's|/|__|g')
        out="results/v2/${slug}__${domain}__mlp__staircase.json"
        # Delete the broken MLP JSON (had linear-only headlines)
        rm -f "$out"
        log="logs/${slug}__${domain}_mlp_retry.log"

        echo ""
        echo "[$(date '+%H:%M:%S')] MLP retry: $model × $domain"
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
t0 = time.time()
try:
    doc = runner.run(args)
    doc['meta']['mlp_run'] = True
    with open(out_path, 'w') as f:
        json.dump(doc, f, indent=2, default=str)
    # Check if MLP headlines exist
    mlp_hl = [h for h in doc.get('headlines',[]) if h.get('probe_type')=='mlp']
    print(f'✓ Wrote {out_path} — {len(mlp_hl)} MLP headlines')
except Exception as e:
    import traceback; traceback.print_exc()
" 2>&1 | tee "$log"
    done
    # Clean model cache
    cache_slug=$(echo "$model" | sed 's|/|--|g')
    rm -rf "/workspace/.hf_home/hub/models--${cache_slug}" 2>/dev/null
done

# ──────────────────────────────────────────────────────────────────────
# W6: Behavioral validation (do models generate correct rhymes?) (~1 hr)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### W6: Behavioral validation — rhyme generation accuracy"
pip install --break-system-packages -q pronouncing 2>/dev/null
python3 << 'PYEOF'
import sys, os, json, glob
sys.path.insert(0, os.getcwd())
os.environ.setdefault('MAAR_DATA_ROOT', 'data/maar_supplementary_material')

import torch
import numpy as np
from src.lookahead.datasets.maar_data import load_maar_rhyme

examples = load_maar_rhyme(split="test")
if not examples:
    examples = load_maar_rhyme()
print(f"Rhyme examples: {len(examples)}")

# Extract the rhyming word from each prompt (last word of the line)
def get_rhyme_word(prompt):
    words = prompt.strip().rstrip(",.:;!?").split()
    return words[-1].lower().strip(".,;:!?'\"") if words else ""

try:
    import pronouncing
    def words_rhyme(w1, w2):
        if w1 == w2: return True
        phones1 = pronouncing.phones_for_word(w1)
        phones2 = pronouncing.phones_for_word(w2)
        if not phones1 or not phones2: return w1[-3:] == w2[-3:]  # fallback
        # Check if they share the rhyming part
        r1 = pronouncing.rhyming_part(phones1[0])
        r2 = pronouncing.rhyming_part(phones2[0])
        return r1 == r2
    print("Using CMU pronouncing dictionary for rhyme checking")
except ImportError:
    def words_rhyme(w1, w2):
        if w1 == w2: return True
        return len(w1) >= 3 and len(w2) >= 3 and w1[-3:] == w2[-3:]
    print("Using simple suffix matching for rhyme checking (no pronouncing lib)")

# Test each model that has a rhyme JSON
models_to_test = []
for f in sorted(glob.glob('results/v2/*__rhyme__staircase.json')):
    if 'step' in f or 'mlp' in f: continue
    d = json.load(open(f))
    models_to_test.append(d['meta']['model'])

# Only test a subset (the headline models) to save time
test_models = ['google/gemma-2-2b', 'google/gemma-2-9b', 'Qwen/Qwen3-1.7B-Base']
models_to_test = [m for m in models_to_test if m in test_models]

from transformers import AutoTokenizer, AutoModelForCausalLM

results = {}
for model_id in models_to_test:
    print(f"\n=== Behavioral: {model_id} ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model.eval()
    except Exception as e:
        print(f"  Failed to load: {e}")
        continue

    n_correct = 0
    n_total = 0
    for ex in examples[:50]:  # test on first 50 for speed
        prompt = ex.prompt
        rhyme_word = get_rhyme_word(prompt)
        if not rhyme_word: continue

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
        generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Get last word of first generated line
        gen_lines = generated.strip().split('\n')
        if gen_lines:
            gen_words = gen_lines[0].strip().rstrip(".,;:!?'\"").split()
            if gen_words:
                gen_rhyme = gen_words[-1].lower().strip(".,;:!?'\"")
                n_total += 1
                if words_rhyme(rhyme_word, gen_rhyme):
                    n_correct += 1

    acc = n_correct / max(n_total, 1)
    results[model_id] = {"correct": n_correct, "total": n_total, "accuracy": acc}
    print(f"  Rhyme accuracy: {n_correct}/{n_total} = {acc:.1%}")

    del model; torch.cuda.empty_cache()

# Correlate with probe gap
print("\n=== Behavioral vs Probe Gap ===")
print(f"{'Model':>25s}  {'Probe gap':>10s}  {'Rhyme acc':>10s}")
for f in sorted(glob.glob('results/v2/*__rhyme__staircase.json')):
    if 'step' in f or 'mlp' in f: continue
    d = json.load(open(f))
    m = d['meta']['model']
    if m not in results: continue
    h = sorted(d['headlines'], key=lambda r: -abs(r['headline_gap']))[0]
    gap = h['headline_gap'] * 100
    acc = results[m]['accuracy']
    print(f"{m.split('/')[-1]:>25s}  {gap:>+9.1f}pp  {acc:>9.1%}")

# Save results
with open('results/v2/behavioral_rhyme.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to results/v2/behavioral_rhyme.json")
PYEOF

# ──────────────────────────────────────────────────────────────────────
# Regenerate everything + push
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "### Final: aggregate + figures + push"
python3 scripts/lookahead/experiments/analyze_staircase_v2.py \
    --results_dir results/v2 --output_dir results/v2
python3 scripts/lookahead/experiments/make_paper_figures.py \
    --results_dir results/v2 --anchor_model google/gemma-2-2b

git add results/v2/ src/ scripts/
git commit -m "Reviewer fixes: W3 grouped CV everywhere + W4 2nd checkpoint sweep + W5 MLP + W6 behavioral

W3: qa_neutral reruns with grouped BoW, cluster bootstrap, grouped ablation.
    BoW should now be ~0.50 (not 0.124), ablation ~0pp (not +35pp),
    bootstrap CIs should contain the headline gap.
W4: Pythia-2.8b checkpoint sweep (step0/4K/32K/143K × code+rhyme).
    Validates floor-vs-learned decomposition on second model.
W5: MLP probe retries with fixed MLPProbe (BaseEstimator inheritance).
W6: Behavioral validation — do models actually generate correct rhymes?"
git push origin psycoplankton/emnlp-staircase-v2

T1=$(date +%s); ELAPSED=$((T1-T0)); echo ""
echo "reviewer_fixes.sh COMPLETE in $((ELAPSED/3600))h $((ELAPSED%3600/60))m"
echo "Total JSONs: $(ls results/v2/*__staircase.json | wc -l)"
