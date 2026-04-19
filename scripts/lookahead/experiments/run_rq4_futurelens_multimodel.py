#!/usr/bin/env python3
"""
FUTURE LENS MULTI-MODEL WITH BOOTSTRAP CIs
=============================================
Runs on: Pythia-2.8B, Llama-3.2-1B, GPT-J-6B (re-run with CIs)
Proper train/test split (no leakage), bootstrap CIs, 3 probe seeds.

This addresses reviewer concern: "Is the K decay GPT-J-specific or general?"
"""

import json, os, sys, time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter, defaultdict

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

PCA_DIM = 128
N_GEN_TOKENS = 80
MIN_TARGET_COUNT = 20
K_VALUES = [1, 2, 3, 5]
N_BOOTSTRAP = 300
PROBE_SEEDS = [42, 123, 456]  # 3 seeds for stability

PROMPTS = [
    # TRAIN (0-24)
    "The process of photosynthesis in plants involves the conversion of",
    "Quantum mechanics fundamentally changed our understanding of physics by",
    "The theory of plate tectonics explains how the earth surface",
    "In molecular biology, DNA replication is the process by which",
    "The second law of thermodynamics states that entropy in an",
    "Neurons communicate with each other through electrical and chemical",
    "The periodic table organizes all known chemical elements according to",
    "General relativity predicts that massive objects cause a distortion in",
    "Evolution through natural selection occurs when organisms with favorable",
    "The human immune system protects the body from pathogens by",
    "The Industrial Revolution transformed European society beginning in the",
    "Ancient Rome expanded its territory through military conquest and",
    "The French Revolution of 1789 was caused by widespread discontent",
    "World War Two ended in 1945 after the Allied powers",
    "The Renaissance was a period of cultural and intellectual rebirth",
    "The fall of the Berlin Wall in 1989 symbolized the",
    "The American Civil War was fought between the northern and",
    "The Age of Exploration led European nations to discover new",
    "The Cold War was a period of geopolitical tension between",
    "Ancient Egyptian civilization developed along the banks of the",
    "Artificial intelligence systems learn from data by identifying patterns",
    "The internet was originally developed as a military communication network",
    "Machine learning algorithms can be broadly classified into supervised and",
    "Cloud computing allows organizations to access computing resources over",
    "Blockchain technology creates a decentralized and distributed digital ledger",
    # TEST (25-49)
    "Neural networks are computational models inspired by the structure of",
    "The development of transistors in the 1950s revolutionized electronics",
    "Programming languages provide abstractions that allow developers to write",
    "Cybersecurity involves protecting computer systems and networks from digital",
    "Quantum computing leverages quantum mechanical phenomena to perform calculations",
    "The largest ocean on Earth is the Pacific Ocean which",
    "Democracy as a form of government gives citizens the power",
    "The global economy is interconnected through international trade and",
    "Climate change is primarily driven by the emission of greenhouse",
    "The United Nations was established to promote international peace and",
    "Education plays a crucial role in the development of modern",
    "The stock market serves as a platform where investors can",
    "Renewable energy sources include solar wind and hydroelectric power",
    "The human brain is the most complex organ in the",
    "Globalization has increased the interconnectedness of economies and cultures",
    "In Python programming the def keyword is used to define",
    "A function that takes a list of numbers and returns",
    "The algorithm works by first sorting the input array and",
    "To implement a binary search tree you need to define",
    "The main difference between a list and a tuple in",
    "Object oriented programming encapsulates data and behavior within classes",
    "A recursive function must have a base case to prevent",
    "The time complexity of quicksort in the average case is",
    "Database normalization reduces data redundancy by organizing tables according",
    "Version control systems like git track changes to source code",
]

TRAIN_IDX = list(range(25))
TEST_IDX = list(range(25, 50))


def bootstrap_accuracy(X, y, n_bootstrap, seed, cv_splits=5):
    """Compute probe accuracy with bootstrap CIs across multiple seeds."""
    all_accs = []
    
    for probe_seed in PROBE_SEEDS:
        # CV accuracy
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=probe_seed)
        fold_accs = []
        for train_idx, test_idx in cv.split(X, y):
            clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=probe_seed)
            clf.fit(X[train_idx], y[train_idx])
            fold_accs.append(clf.score(X[test_idx], y[test_idx]))
        all_accs.append(np.mean(fold_accs))
    
    mean_acc = np.mean(all_accs)
    
    # Bootstrap CI
    rng = np.random.RandomState(seed)
    boot_accs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(X), len(X), replace=True)
        oob = list(set(range(len(X))) - set(idx))
        if len(oob) < 5 or len(np.unique(y[idx])) < max(2, len(np.unique(y)) // 2):
            continue
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
        clf.fit(X[idx], y[idx])
        boot_accs.append(clf.score(X[oob], y[oob]))
    
    if boot_accs:
        ci_lo, ci_hi = np.percentile(boot_accs, [2.5, 97.5])
    else:
        ci_lo, ci_hi = mean_acc, mean_acc
    
    return float(mean_acc), float(ci_lo), float(ci_hi), float(np.std(all_accs))


def run_model(model_name, dtype):
    """Run Future Lens staircase on one model."""
    from transformer_lens import HookedTransformer
    
    logger.info(f"\n{'='*70}")
    logger.info(f"MODEL: {model_name}")
    logger.info(f"{'='*70}")
    
    model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=dtype)
    model.eval()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    logger.info(f"  {n_layers} layers, d_model={d_model}")
    
    layers = sorted(set([0, n_layers//6, n_layers//3, n_layers//2,
                         2*n_layers//3, 5*n_layers//6, n_layers-1]))
    
    W_E = model.W_E.detach()
    
    # Generate
    logger.info(f"  Generating {N_GEN_TOKENS} tokens for {len(PROMPTS)} prompts...")
    all_sequences = []
    for pi, prompt in enumerate(PROMPTS):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            gen = model.generate(tokens, max_new_tokens=N_GEN_TOKENS, temperature=0.0)
        all_sequences.append({
            "prompt_len": tokens.shape[1],
            "full_ids": gen[0].cpu().tolist()
        })
        if (pi + 1) % 10 == 0:
            logger.info(f"    Generated {pi+1}/{len(PROMPTS)}")
    
    train_seqs = [all_sequences[i] for i in TRAIN_IDX]
    test_seqs = [all_sequences[i] for i in TEST_IDX]
    
    model_results = {"model": model_name, "n_layers": n_layers, "d_model": d_model}
    
    for k in K_VALUES:
        logger.info(f"\n  === K={k} ===")
        
        # Build trigram on TRAIN
        bigram_table = defaultdict(Counter)
        trigram_table = defaultdict(Counter)
        for seq in train_seqs:
            ids = seq["full_ids"]
            pl = seq["prompt_len"]
            for n in range(pl, len(ids) - k):
                bigram_table[ids[n]][ids[n+k]] += 1
                if n >= 1:
                    trigram_table[(ids[n-1], ids[n])][ids[n+k]] += 1
        
        # Find frequent targets in TEST
        test_targets = []
        for seq in test_seqs:
            ids = seq["full_ids"]
            pl = seq["prompt_len"]
            for n in range(pl, len(ids) - k):
                test_targets.append(ids[n+k])
        
        target_counts = Counter(test_targets)
        frequent = {t for t, c in target_counts.items() if c >= MIN_TARGET_COUNT}
        t2i = {t: i for i, t in enumerate(sorted(frequent))}
        n_classes = len(t2i)
        
        if n_classes < 5:
            logger.warning(f"    Only {n_classes} classes, skipping")
            continue
        
        # Extract TEST data
        activations = {l: [] for l in layers}
        token_embs = []
        ctx_embs = []
        labels = []
        src_unis = []
        src_bis = []
        
        for seq in test_seqs:
            ids = seq["full_ids"]
            pl = seq["prompt_len"]
            full_input = torch.tensor([ids], device="cuda")
            
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    full_input,
                    names_filter=[f"blocks.{l}.hook_resid_post" for l in layers])
            
            for n in range(pl, len(ids) - k):
                target = ids[n+k]
                if target not in t2i:
                    continue
                labels.append(t2i[target])
                src_unis.append(ids[n])
                src_bis.append((ids[max(0,n-1)], ids[n]))
                for layer in layers:
                    activations[layer].append(
                        cache[f"blocks.{layer}.hook_resid_post"][0, n, :].cpu().numpy())
                token_embs.append(W_E[ids[n]].cpu().numpy())
                ws = max(0, n-4)
                ctx = W_E[torch.tensor(ids[ws:n+1], device="cuda")].cpu().numpy()
                ctx_embs.append(ctx.mean(axis=0))
            
            del cache
            torch.cuda.empty_cache()
        
        labels = np.array(labels)
        n_ex = len(labels)
        class_counts = Counter(labels)
        min_class = min(class_counts.values())
        max_class = max(class_counts.values())
        majority_class = class_counts.most_common(1)[0][0]
        chance = max_class / n_ex
        
        logger.info(f"    Examples: {n_ex}, classes: {n_classes}, chance: {chance:.4f}")
        
        n_splits = min(5, min_class)
        if n_splits < 2:
            logger.warning(f"    Skipping — min class {min_class}")
            continue
        
        k_results = {"k": k, "n_examples": n_ex, "n_classes": n_classes, "chance": float(chance)}
        
        # Bigram (train→test)
        bigram_preds = []
        for s in src_unis:
            if s in bigram_table and bigram_table[s]:
                best = bigram_table[s].most_common(1)[0][0]
                bigram_preds.append(t2i.get(best, majority_class))
            else:
                bigram_preds.append(majority_class)
        bigram_acc = float(np.mean(np.array(bigram_preds) == labels))
        k_results["bigram"] = bigram_acc
        
        # Trigram (train→test)
        trigram_preds = []
        for s in src_bis:
            if s in trigram_table and trigram_table[s]:
                best = trigram_table[s].most_common(1)[0][0]
                trigram_preds.append(t2i.get(best, majority_class))
            else:
                trigram_preds.append(majority_class)
        trigram_acc = float(np.mean(np.array(trigram_preds) == labels))
        k_results["trigram"] = trigram_acc
        
        # Token-N embedding with bootstrap CI
        X_emb = np.stack(token_embs)
        scaler_e = StandardScaler()
        X_emb_s = scaler_e.fit_transform(X_emb)
        X_emb_p = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(X_emb_s)
        emb_acc, emb_lo, emb_hi, emb_std = bootstrap_accuracy(X_emb_p, labels, N_BOOTSTRAP, 42)
        logger.info(f"    Emb:     {emb_acc:.4f} [{emb_lo:.4f}, {emb_hi:.4f}]")
        k_results["embedding"] = {"acc": emb_acc, "ci_lo": emb_lo, "ci_hi": emb_hi}
        
        # Context-window embedding with bootstrap CI
        X_ctx = np.stack(ctx_embs)
        scaler_c = StandardScaler()
        X_ctx_s = scaler_c.fit_transform(X_ctx)
        X_ctx_p = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(X_ctx_s)
        ctx_acc, ctx_lo, ctx_hi, ctx_std = bootstrap_accuracy(X_ctx_p, labels, N_BOOTSTRAP, 42)
        logger.info(f"    Context: {ctx_acc:.4f} [{ctx_lo:.4f}, {ctx_hi:.4f}]")
        k_results["context"] = {"acc": ctx_acc, "ci_lo": ctx_lo, "ci_hi": ctx_hi}
        
        # Probes with bootstrap CI
        logger.info(f"    {'Layer':>6} {'Probe':>8} {'CI_lo':>8} {'CI_hi':>8} {'Gap_ctx':>10} {'Gap_sig?':>10}")
        
        probe_results = {}
        for layer in layers:
            X_p = np.stack(activations[layer])
            scaler_p = StandardScaler()
            X_p_s = scaler_p.fit_transform(X_p)
            X_p_pca = PCA(n_components=min(PCA_DIM, n_ex-1),
                          random_state=42).fit_transform(X_p_s)
            
            p_acc, p_lo, p_hi, p_std = bootstrap_accuracy(X_p_pca, labels, N_BOOTSTRAP, 42)
            gap_ctx = p_acc - ctx_acc
            
            # Is the gap significant? Check if probe CI_lo > context CI_hi
            gap_sig = "YES" if p_lo > ctx_hi else "no"
            
            logger.info(f"    L{layer:>4} {p_acc:>8.4f} {p_lo:>8.4f} {p_hi:>8.4f} "
                        f"{gap_ctx:>+10.4f} {gap_sig:>10}")
            
            probe_results[str(layer)] = {
                "probe": p_acc, "ci_lo": p_lo, "ci_hi": p_hi, "std": p_std,
                "gap_vs_context": gap_ctx,
                "gap_significant": gap_sig == "YES",
            }
        
        k_results["probes"] = probe_results
        
        # Summary
        best = max(probe_results.items(), key=lambda x: x[1]["probe"])
        logger.info(f"\n    STAIRCASE K={k}: chance={chance:.3f} | bigram={bigram_acc:.3f} | "
                    f"trigram={trigram_acc:.3f} | emb={emb_acc:.3f} [{emb_lo:.3f},{emb_hi:.3f}] | "
                    f"ctx={ctx_acc:.3f} [{ctx_lo:.3f},{ctx_hi:.3f}] | "
                    f"probe(L{best[0]})={best[1]['probe']:.3f} [{best[1]['ci_lo']:.3f},{best[1]['ci_hi']:.3f}] | "
                    f"gap={best[1]['gap_vs_context']:+.3f}")
        
        model_results[f"k{k}"] = k_results
    
    del model
    torch.cuda.empty_cache()
    return model_results


def main():
    logger.info("=" * 70)
    logger.info("FUTURE LENS MULTI-MODEL WITH BOOTSTRAP CIs")
    logger.info("Models: Pythia-2.8B, Llama-3.2-1B, GPT-J-6B")
    logger.info(f"Bootstrap: {N_BOOTSTRAP}, Probe seeds: {PROBE_SEEDS}")
    logger.info("=" * 70)
    
    # Install deps if needed
    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        os.system("pip install transformer-lens==2.11.0 transformers==4.44.0 scikit-learn scipy --break-system-packages")
    
    models = [
        ("pythia-2.8b", torch.float16),
        ("Qwen/Qwen2.5-1.5B", torch.float16),
        ("EleutherAI/gpt-j-6b", torch.float16),
    ]
    
    all_results = {}
    for model_name, dtype in models:
        try:
            all_results[model_name] = run_model(model_name, dtype)
        except Exception as e:
            logger.error(f"  {model_name} FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results[model_name] = {"error": str(e)}
    
    # Cross-model summary
    logger.info(f"\n{'='*70}")
    logger.info("CROSS-MODEL SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'Model':<25} {'K':>3} {'Chance':>8} {'Context':>10} {'Probe':>10} {'Gap':>8} {'CI':>20}")
    logger.info("-" * 80)
    
    for model_name, mresults in all_results.items():
        if "error" in mresults:
            continue
        for k in K_VALUES:
            key = f"k{k}"
            if key not in mresults:
                continue
            r = mresults[key]
            best = max(r["probes"].items(), key=lambda x: x[1]["probe"])
            ctx = r["context"]
            logger.info(f"{model_name:<25} {k:>3} {r['chance']:>8.4f} "
                        f"{ctx['acc']:>10.4f} {best[1]['probe']:>10.4f} "
                        f"{best[1]['gap_vs_context']:>+8.4f} "
                        f"[{best[1]['ci_lo']:.3f},{best[1]['ci_hi']:.3f}]")
    
    # Save
    outfile = "results/lookahead/final/future_lens_multimodel.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved to {outfile}")
    
    logger.info("\n" + "=" * 70)
    logger.info("DONE — MULTI-MODEL FUTURE LENS")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
