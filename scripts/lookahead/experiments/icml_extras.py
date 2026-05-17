#!/usr/bin/env python3
"""icml_extras.py — Additional analyses addressing ICML reviewer weaknesses.

W4: Logit lens comparison on code domain (return types are single tokens)
W5: Expanded behavioral validation on all rhyme models (Spearman + p-value)
W2: Floor analysis from existing mean-pool data (architecture comparison)
W6: Permutation test — is trained gap significantly > floor?

Run:
    export HF_TOKEN=... && export HF_HOME=/workspace/.hf_home
    export MAAR_DATA_ROOT=data/maar_supplementary_material
    python3 scripts/lookahead/experiments/icml_extras.py \
        --results_dir results/v2 --mode all
"""
import sys, os, json, glob, argparse, time
import numpy as np
sys.path.insert(0, os.getcwd())
os.environ.setdefault("MAAR_DATA_ROOT", "data/maar_supplementary_material")


# ──────────────────────────────────────────────────────────────────────
# W4: Logit lens — do hidden states at target position predict return type
#     in vocabulary space (not just probe space)?
# ──────────────────────────────────────────────────────────────────────
def logit_lens_code(results_dir: str, models_to_test: list[str] | None = None):
    """For code domain, check if logit lens also shows target > earlier."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.lookahead.datasets.code_untyped import load_code_return_examples

    examples = load_code_return_examples()
    # Return type tokens
    type_tokens = ["int", "str", "float", "bool", "list"]
    print(f"\n{'='*60}")
    print(f"W4: LOGIT LENS — code domain ({len(examples)} examples)")
    print(f"{'='*60}")

    if models_to_test is None:
        models_to_test = ["google/gemma-2-2b", "Qwen/Qwen3-1.7B-Base",
                          "EleutherAI/pythia-1.4b-deduped"]

    results = {}
    for model_id in models_to_test:
        print(f"\n--- {model_id} ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True)
            model.eval()
        except Exception as e:
            print(f"  Failed to load: {e}"); continue

        # Get token IDs for return types
        type_token_ids = {}
        for t in type_tokens:
            # Try both with and without space prefix
            ids = tokenizer.encode(t, add_special_tokens=False)
            ids_sp = tokenizer.encode(f" {t}", add_special_tokens=False)
            type_token_ids[t] = list(set(ids + ids_sp))

        target_ranks = []  # rank of correct return type at target position
        earlier_ranks = [] # rank of correct return type at best earlier position

        for ex in examples[:100]:  # cap at 100 for speed
            prompt = ex.prompt
            label = ex.label  # return type string

            if label not in type_token_ids:
                continue

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            logits = outputs.logits[0]  # (seq_len, vocab_size)
            seq_len = logits.shape[0]

            # Target position = last token (where probe measures)
            target_logits = logits[-1]
            # Earlier positions: try positions before target
            n_earlier = max(1, seq_len // 2)

            # Softmax at target
            target_probs = torch.softmax(target_logits, dim=-1)
            target_prob = max(target_probs[tid].item() for tid in type_token_ids[label])

            # Best earlier position's probability
            best_earlier_prob = 0.0
            for pos in range(min(n_earlier, seq_len - 1)):
                probs = torch.softmax(logits[pos], dim=-1)
                p = max(probs[tid].item() for tid in type_token_ids[label])
                best_earlier_prob = max(best_earlier_prob, p)

            target_ranks.append(target_prob)
            earlier_ranks.append(best_earlier_prob)

        if target_ranks:
            mean_target = np.mean(target_ranks)
            mean_earlier = np.mean(earlier_ranks)
            gap = mean_target - mean_earlier
            # Paired Wilcoxon on the per-example differences
            from scipy.stats import wilcoxon
            diffs = np.array(target_ranks) - np.array(earlier_ranks)
            try:
                stat, p_val = wilcoxon(diffs, alternative='greater')
            except Exception:
                p_val = float('nan')

            results[model_id] = {
                "n_examples": len(target_ranks),
                "mean_target_prob": round(mean_target, 4),
                "mean_earlier_prob": round(mean_earlier, 4),
                "logit_lens_gap": round(gap, 4),
                "wilcoxon_p": round(p_val, 6),
            }
            print(f"  Target prob: {mean_target:.4f}")
            print(f"  Earlier prob: {mean_earlier:.4f}")
            print(f"  Logit lens gap: {gap:+.4f}  (p={p_val:.4f})")
        del model; torch.cuda.empty_cache()

    return results


# ──────────────────────────────────────────────────────────────────────
# W5: Expanded behavioral validation — all rhyme models
# ──────────────────────────────────────────────────────────────────────
def behavioral_all_models(results_dir: str):
    """Run rhyme generation on all models with rhyme JSONs."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.lookahead.datasets.maar_data import load_maar_rhyme

    examples = load_maar_rhyme(split="test")
    if not examples:
        examples = load_maar_rhyme()
    print(f"\n{'='*60}")
    print(f"W5: BEHAVIORAL VALIDATION — {len(examples)} rhyme examples")
    print(f"{'='*60}")

    try:
        import pronouncing
        def words_rhyme(w1, w2):
            if w1.lower() == w2.lower(): return True
            p1 = pronouncing.phones_for_word(w1.lower())
            p2 = pronouncing.phones_for_word(w2.lower())
            if not p1 or not p2: return w1[-3:] == w2[-3:]
            return pronouncing.rhyming_part(p1[0]) == pronouncing.rhyming_part(p2[0])
        print("Using CMU pronouncing dictionary")
    except ImportError:
        def words_rhyme(w1, w2):
            return w1.lower() == w2.lower() or (len(w1)>=3 and len(w2)>=3 and w1[-3:]==w2[-3:])
        print("Using suffix matching (install 'pronouncing' for better results)")

    def get_rhyme_word(prompt):
        words = prompt.strip().rstrip(",.:;!?").split()
        return words[-1].lower().strip(".,;:!?'\"") if words else ""

    # Find all models with rhyme results (skip checkpoints and MLP)
    models_to_test = []
    for f in sorted(glob.glob(f'{results_dir}/*__rhyme__staircase.json')):
        if 'step' in f or 'mlp' in f: continue
        d = json.load(open(f))
        models_to_test.append(d['meta']['model'])

    results = {}
    for model_id in models_to_test:
        short = model_id.split('/')[-1]
        print(f"\n--- {short} ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True)
            model.eval()
        except Exception as e:
            print(f"  Skip (load failed): {e}"); continue

        n_correct = 0; n_total = 0
        for ex in examples[:50]:
            rhyme_word = get_rhyme_word(ex.prompt)
            if not rhyme_word: continue
            inputs = tokenizer(ex.prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=30, do_sample=False,
                                      pad_token_id=tokenizer.pad_token_id)
            gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            gen_lines = gen.strip().split('\n')
            if gen_lines:
                gw = gen_lines[0].strip().rstrip(".,;:!?'\"").split()
                if gw:
                    n_total += 1
                    if words_rhyme(rhyme_word, gw[-1]):
                        n_correct += 1

        acc = n_correct / max(n_total, 1)
        results[model_id] = {"correct": n_correct, "total": n_total, "accuracy": round(acc, 3)}
        print(f"  {n_correct}/{n_total} = {acc:.1%}")
        del model; torch.cuda.empty_cache()

    # Correlate with probe gap
    print(f"\n--- Correlation: Probe gap vs Rhyme accuracy ---")
    gaps = []; accs = []
    print(f"{'Model':>25s}  {'Probe gap':>10s}  {'Rhyme acc':>10s}")
    for f in sorted(glob.glob(f'{results_dir}/*__rhyme__staircase.json')):
        if 'step' in f or 'mlp' in f: continue
        d = json.load(open(f))
        m = d['meta']['model']
        if m not in results: continue
        h = sorted(d['headlines'], key=lambda r: -abs(r['headline_gap']))[0]
        gap = h['headline_gap'] * 100
        acc = results[m]['accuracy']
        gaps.append(gap); accs.append(acc)
        print(f"{m.split('/')[-1]:>25s}  {gap:>+9.1f}pp  {acc:>9.1%}")

    if len(gaps) >= 4:
        from scipy.stats import spearmanr
        rho, p = spearmanr(gaps, accs)
        print(f"\nSpearman rho = {rho:.3f}, p = {p:.4f} (n={len(gaps)})")
        results["_correlation"] = {"spearman_rho": round(rho, 3), "p_value": round(p, 4), "n": len(gaps)}
    else:
        print(f"\nToo few models for correlation (n={len(gaps)})")

    return results


# ──────────────────────────────────────────────────────────────────────
# W2: Floor analysis — does the floor vary by architecture/PE scheme?
# ──────────────────────────────────────────────────────────────────────
def floor_analysis(results_dir: str):
    """Analyze positional floor across architectures using mean-pool data."""
    print(f"\n{'='*60}")
    print(f"W2: FLOOR ANALYSIS — positional floor by architecture")
    print(f"{'='*60}")

    # Approach: positional_component ≈ gap_per_position - gap_mean_pool
    # If mean_pool_gap ≈ 0 but per_position_gap > 0, the gap is positional
    results = {}
    print(f"\n{'Model':>25s} {'Domain':>6s} {'PerPos gap':>11s} {'MeanPool gap':>13s} {'Positional':>11s} {'PE scheme':>10s}")

    pe_schemes = {
        'pythia': 'rotary', 'gpt2': 'absolute', 'gemma': 'RoPE+norm', 'qwen': 'RoPE'
    }

    for f in sorted(glob.glob(f'{results_dir}/*__staircase.json')):
        if 'step' in f or 'mlp' in f: continue
        d = json.load(open(f))
        m = d['meta']['model']; dom = d['meta']['domain']
        if dom not in ['code', 'rhyme']: continue

        h = sorted(d['headlines'], key=lambda r: -abs(r['headline_gap']))[0]
        gap = h['headline_gap'] * 100
        mp = d['baselines'].get('mean_pool_accuracy', {}).get(str(h['layer']))
        if mp is None: continue
        mp_gap = (h['target_accuracy'] - mp) * 100
        positional = gap - mp_gap

        # Determine PE scheme
        short = m.split('/')[-1].lower()
        pe = 'unknown'
        for prefix, scheme in pe_schemes.items():
            if prefix in short: pe = scheme; break

        results.setdefault(dom, []).append({
            'model': m.split('/')[-1], 'gap': gap, 'mp_gap': mp_gap,
            'positional': positional, 'pe': pe
        })
        print(f"{m.split('/')[-1]:>25s} {dom:>6s} {gap:>+10.1f}pp {mp_gap:>+12.1f}pp {positional:>+10.1f}pp {pe:>10s}")

    # Summary by PE scheme
    print(f"\n--- Summary by PE scheme ---")
    for dom in ['code', 'rhyme']:
        if dom not in results: continue
        print(f"\n  {dom.upper()}:")
        by_pe = {}
        for r in results[dom]:
            by_pe.setdefault(r['pe'], []).append(r['positional'])
        for pe, vals in sorted(by_pe.items()):
            print(f"    {pe:>12s}: mean positional = {np.mean(vals):+.1f}pp (n={len(vals)})")

    return results


# ──────────────────────────────────────────────────────────────────────
# W6: Permutation test — is trained gap significantly > floor?
# ──────────────────────────────────────────────────────────────────────
def floor_permutation_test(results_dir: str):
    """Test whether trained gaps are significantly larger than random-init floors."""
    print(f"\n{'='*60}")
    print(f"W6: PERMUTATION TEST — trained gap vs floor")
    print(f"{'='*60}")

    # Load checkpoint data: step0 (floor) vs step143000 (trained)
    results = {}
    for model_name in ['EleutherAI__pythia-1.4b-deduped', 'EleutherAI__pythia-2.8b-deduped']:
        for dom in ['code', 'rhyme']:
            floor_f = f'{results_dir}/{model_name}__step0__{dom}__staircase.json'
            trained_f = f'{results_dir}/{model_name}__step143000__{dom}__staircase.json'
            if not os.path.exists(floor_f) or not os.path.exists(trained_f):
                continue

            d0 = json.load(open(floor_f))
            dt = json.load(open(trained_f))

            # Get per-layer gaps for both
            h0 = sorted(d0['headlines'], key=lambda r: -abs(r['headline_gap']))[0]
            ht = sorted(dt['headlines'], key=lambda r: -abs(r['headline_gap']))[0]

            floor_gap = h0['headline_gap'] * 100
            trained_gap = ht['headline_gap'] * 100
            learned = trained_gap - floor_gap

            # Collect per-position gaps from both for a permutation test
            # Use all headline rows to get distribution
            gaps_floor = [h['headline_gap']*100 for h in d0['headlines']]
            gaps_trained = [h['headline_gap']*100 for h in dt['headlines']]

            # Permutation test: are trained gaps systematically larger?
            from scipy.stats import mannwhitneyu
            try:
                stat, p_val = mannwhitneyu(gaps_trained, gaps_floor, alternative='greater')
            except Exception:
                p_val = float('nan')

            short = model_name.replace('EleutherAI__', '')
            key = f"{short} × {dom}"
            results[key] = {
                'floor': round(floor_gap, 1), 'trained': round(trained_gap, 1),
                'learned': round(learned, 1), 'p_value': round(p_val, 4),
                'significant': p_val < 0.05
            }
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            print(f"  {key:>35s}  floor={floor_gap:+.1f}  trained={trained_gap:+.1f}  "
                  f"learned={learned:+.1f}pp  p={p_val:.4f} {sig}")

    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/v2")
    ap.add_argument("--mode", default="all", choices=["all", "logit_lens", "behavioral", "floor", "permtest"])
    args = ap.parse_args()

    all_results = {}

    if args.mode in ("all", "floor"):
        all_results["floor_analysis"] = floor_analysis(args.results_dir)

    if args.mode in ("all", "permtest"):
        all_results["permutation_test"] = floor_permutation_test(args.results_dir)

    if args.mode in ("all", "behavioral"):
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages",
                       "-q", "pronouncing"], capture_output=True)
        all_results["behavioral"] = behavioral_all_models(args.results_dir)

    if args.mode in ("all", "logit_lens"):
        all_results["logit_lens"] = logit_lens_code(args.results_dir)

    # Save
    out = os.path.join(args.results_dir, "icml_extras.json")
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out}")

if __name__ == "__main__":
    main()
