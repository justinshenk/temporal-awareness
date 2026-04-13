#!/usr/bin/env python3
"""
FUTURE LENS v4 — ADDS K=5 + N-GRAM BASELINES
===============================================
Builds on v3. Additions:
1. K=5 to test if gap vanishes at longer range
2. Bigram/trigram baseline — does token co-occurrence explain the signal?
3. Fixed convergence — lbfgs with max_iter=5000 (saga was failing)
4. Reports per-layer results for ALL K values

This is the final version for the paper.
"""

import json, os, sys, time, random
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter, defaultdict

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

MODEL_NAME = "EleutherAI/gpt-j-6b"
PCA_DIM = 128
N_GEN_TOKENS = 80
MIN_TARGET_COUNT = 10
K_VALUES = [1, 2, 3, 5]

PROMPTS = [
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


def build_ngram_baseline(all_sequences, k, t2i, frequent_targets):
    """
    Build n-gram baseline: for each token at position N, what's the most
    likely token at N+K based on training co-occurrence statistics?
    
    Uses the generated text itself as the corpus (empirical bigram/trigram).
    This is conservative — real n-gram tables from training data would be stronger.
    """
    # Build co-occurrence: token_at_N -> Counter of tokens at N+K
    if k == 1:
        # Bigram: P(token_{N+1} | token_N)
        cooccurrence = defaultdict(Counter)
        for seq in all_sequences:
            ids = seq["full_ids"]
            pl = seq["prompt_len"]
            for n in range(pl, len(ids) - k):
                src = ids[n]
                tgt = ids[n + k]
                if tgt in frequent_targets:
                    cooccurrence[src][tgt] += 1
    else:
        # Trigram-like: P(token_{N+K} | token_{N-1}, token_N)
        cooccurrence = defaultdict(Counter)
        for seq in all_sequences:
            ids = seq["full_ids"]
            pl = seq["prompt_len"]
            for n in range(max(pl, 1), len(ids) - k):
                src = (ids[n-1], ids[n])  # bigram context
                tgt = ids[n + k]
                if tgt in frequent_targets:
                    cooccurrence[src][tgt] += 1
    
    return cooccurrence


def predict_ngram(cooccurrence, src_key, t2i, majority_class):
    """Predict using n-gram table. Returns class index."""
    if src_key in cooccurrence and cooccurrence[src_key]:
        # Most common target for this source
        best_target = cooccurrence[src_key].most_common(1)[0][0]
        if best_target in t2i:
            return t2i[best_target]
    return majority_class


def main():
    logger.info("=" * 70)
    logger.info("FUTURE LENS v4 — K=1,2,3,5 + N-GRAM BASELINES")
    logger.info("=" * 70)
    
    from transformer_lens import HookedTransformer
    
    logger.info("Loading model...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda", dtype=torch.float16)
    model.eval()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    logger.info(f"  Loaded in {time.time()-t0:.1f}s — {n_layers} layers, d_model={d_model}")
    
    layers = sorted(set([0, n_layers//6, n_layers//3, n_layers//2,
                         2*n_layers//3, 5*n_layers//6, n_layers-1]))
    
    # Generate
    logger.info(f"\n  Generating {N_GEN_TOKENS} tokens for {len(PROMPTS)} prompts...")
    all_sequences = []
    for pi, prompt in enumerate(PROMPTS):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        prompt_len = tokens.shape[1]
        with torch.no_grad():
            gen = model.generate(tokens, max_new_tokens=N_GEN_TOKENS, temperature=0.0)
        all_sequences.append({"prompt_len": prompt_len, "full_ids": gen[0].cpu().tolist()})
        if (pi + 1) % 10 == 0:
            logger.info(f"    Generated {pi+1}/{len(PROMPTS)}")
    
    W_E = model.W_E.detach()
    all_results = {"model": MODEL_NAME, "n_layers": n_layers, "d_model": d_model,
                   "k_values": K_VALUES, "n_prompts": len(PROMPTS)}
    
    for k in K_VALUES:
        logger.info(f"\n{'='*60}")
        logger.info(f"K={k}: PREDICTING TOKEN N+{k}")
        logger.info(f"{'='*60}")
        
        # Collect target tokens
        all_targets = []
        for seq in all_sequences:
            pl = seq["prompt_len"]
            ids = seq["full_ids"]
            for n in range(pl, len(ids) - k):
                all_targets.append(ids[n + k])
        
        target_counts = Counter(all_targets)
        frequent_targets = {t for t, c in target_counts.items() if c >= MIN_TARGET_COUNT}
        t2i = {t: i for i, t in enumerate(sorted(frequent_targets))}
        n_classes = len(t2i)
        
        top5 = [(model.to_string(torch.tensor([t])), c) for t, c in target_counts.most_common(5)]
        logger.info(f"  Prediction points: {len(all_targets)}, unique: {len(target_counts)}, "
                    f"frequent(≥{MIN_TARGET_COUNT}): {n_classes}")
        logger.info(f"  Top-5 targets: {top5}")
        
        if n_classes < 5:
            logger.warning(f"  Skipping K={k}")
            continue
        
        # Extract everything
        logger.info(f"  Extracting activations...")
        activations = {l: [] for l in layers}
        token_n_embs = []
        context_window_embs = []
        labels = []
        src_tokens = []       # for bigram baseline
        src_bigrams = []      # for trigram baseline
        
        for si, seq in enumerate(all_sequences):
            pl = seq["prompt_len"]
            ids = seq["full_ids"]
            full_input = torch.tensor([ids], device="cuda")
            
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    full_input,
                    names_filter=[f"blocks.{l}.hook_resid_post" for l in layers]
                )
            
            for n in range(pl, len(ids) - k):
                target = ids[n + k]
                if target not in t2i:
                    continue
                
                labels.append(t2i[target])
                src_tokens.append(ids[n])
                src_bigrams.append((ids[max(0,n-1)], ids[n]))
                
                for layer in layers:
                    activations[layer].append(
                        cache[f"blocks.{layer}.hook_resid_post"][0, n, :].cpu().numpy())
                
                token_n_embs.append(W_E[ids[n]].cpu().numpy())
                
                ws = max(0, n - 4)
                ctx = W_E[torch.tensor(ids[ws:n+1], device="cuda")].cpu().numpy()
                context_window_embs.append(ctx.mean(axis=0))
            
            del cache
            torch.cuda.empty_cache()
            if (si + 1) % 10 == 0:
                logger.info(f"    {si+1}/{len(all_sequences)}, examples: {len(labels)}")
        
        labels = np.array(labels)
        n_ex = len(labels)
        class_counts = Counter(labels)
        min_class = min(class_counts.values())
        max_class = max(class_counts.values())
        chance_majority = max_class / n_ex
        majority_class = class_counts.most_common(1)[0][0]
        
        logger.info(f"  Examples: {n_ex}, classes: {n_classes}, "
                    f"class range: [{min_class}, {max_class}]")
        
        n_splits = min(5, min_class)
        if n_splits < 2:
            logger.warning(f"  Skipping — min class {min_class}")
            continue
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        k_results = {
            "k": int(k), "n_examples": int(n_ex), "n_classes": int(n_classes),
            "chance_majority": float(chance_majority), "chance_uniform": float(1/n_classes),
        }
        
        # ============================================================
        # BASELINE 1: N-gram
        # ============================================================
        logger.info(f"\n  --- N-gram baselines ---")
        
        # Bigram: predict from token_N alone
        bigram_table = build_ngram_baseline(all_sequences, k, t2i, frequent_targets)
        bigram_preds = np.array([predict_ngram(bigram_table, s, t2i, majority_class) 
                                 for s in src_tokens])
        bigram_acc = float(np.mean(bigram_preds == labels))
        logger.info(f"    Bigram accuracy:  {bigram_acc:.4f}")
        k_results["bigram"] = bigram_acc
        
        # Trigram: predict from (token_{N-1}, token_N)
        if k <= 3:
            trigram_table = defaultdict(Counter)
            for seq in all_sequences:
                ids = seq["full_ids"]
                pl = seq["prompt_len"]
                for n in range(max(pl, 1), len(ids) - k):
                    src = (ids[n-1], ids[n])
                    tgt = ids[n + k]
                    if tgt in frequent_targets:
                        trigram_table[src][tgt] += 1
            
            trigram_preds = np.array([predict_ngram(trigram_table, s, t2i, majority_class)
                                      for s in src_bigrams])
            trigram_acc = float(np.mean(trigram_preds == labels))
            logger.info(f"    Trigram accuracy: {trigram_acc:.4f}")
            k_results["trigram"] = trigram_acc
        
        # ============================================================
        # BASELINE 2: Token-N embedding (FAIR)
        # ============================================================
        logger.info(f"\n  --- Embedding baselines (d={d_model} → PCA {PCA_DIM}) ---")
        X_emb = np.stack(token_n_embs)
        scaler_e = StandardScaler()
        X_emb_s = scaler_e.fit_transform(X_emb)
        X_emb_p = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(X_emb_s)
        
        emb_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
            X_emb_p, labels, cv=cv, scoring="accuracy")
        emb_acc = emb_scores.mean()
        logger.info(f"    Token-N embedding:      {emb_acc:.4f}")
        k_results["token_n_embedding"] = float(emb_acc)
        
        # ============================================================
        # BASELINE 3: Context-window embedding (FAIR)
        # ============================================================
        X_ctx = np.stack(context_window_embs)
        scaler_c = StandardScaler()
        X_ctx_s = scaler_c.fit_transform(X_ctx)
        X_ctx_p = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(X_ctx_s)
        
        ctx_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
            X_ctx_p, labels, cv=cv, scoring="accuracy")
        ctx_acc = ctx_scores.mean()
        logger.info(f"    Context-window emb:     {ctx_acc:.4f}")
        k_results["context_window_embedding"] = float(ctx_acc)
        
        # ============================================================
        # PROBE: Each layer
        # ============================================================
        logger.info(f"\n  --- Probes ---")
        logger.info(f"  {'Layer':>6} {'Probe':>8} {'vs Emb':>8} {'vs Ctx':>8} "
                    f"{'vs Bigram':>10} {'vs Chance':>10}")
        
        probe_results = {}
        for layer in layers:
            X_p = np.stack(activations[layer])
            scaler_p = StandardScaler()
            X_p_s = scaler_p.fit_transform(X_p)
            X_p_pca = PCA(n_components=min(PCA_DIM, n_ex-1), 
                          random_state=42).fit_transform(X_p_s)
            
            p_scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                X_p_pca, labels, cv=cv, scoring="accuracy")
            p_acc = p_scores.mean()
            
            logger.info(f"  L{layer:>4} {p_acc:>8.4f} {p_acc-emb_acc:>+8.4f} "
                        f"{p_acc-ctx_acc:>+8.4f} {p_acc-bigram_acc:>+10.4f} "
                        f"{p_acc-chance_majority:>+10.4f}")
            
            probe_results[str(layer)] = {
                "probe": float(p_acc),
                "gap_vs_embedding": float(p_acc - emb_acc),
                "gap_vs_context": float(p_acc - ctx_acc),
                "gap_vs_bigram": float(p_acc - bigram_acc),
            }
        
        k_results["probe_by_layer"] = probe_results
        
        # Summary
        best = max(probe_results.items(), key=lambda x: x[1]["probe"])
        best_acc = best[1]["probe"]
        best_gap_ctx = best[1]["gap_vs_context"]
        best_gap_bigram = best[1]["gap_vs_bigram"]
        
        logger.info(f"\n  === STAIRCASE (K={k}) ===")
        logger.info(f"  Chance (majority):    {chance_majority:.4f}")
        logger.info(f"  Bigram:               {bigram_acc:.4f}")
        if "trigram" in k_results:
            logger.info(f"  Trigram:              {k_results['trigram']:.4f}")
        logger.info(f"  Token-N embedding:    {emb_acc:.4f}")
        logger.info(f"  Context-window emb:   {ctx_acc:.4f}")
        logger.info(f"  Best Probe (L{best[0]}):   {best_acc:.4f}")
        logger.info(f"  Gap (probe − ctx):    {best_gap_ctx:+.4f}")
        logger.info(f"  Gap (probe − bigram): {best_gap_bigram:+.4f}")
        
        if best_gap_ctx < 0.02:
            logger.info(f"  >>> CONTEXT EXPLAINS SIGNAL <<<")
        elif best_gap_ctx < 0.05:
            logger.info(f"  >>> SMALL RESIDUAL <<<")
        else:
            logger.info(f"  >>> PROBE EXCEEDS ALL BASELINES <<<")
        
        all_results[f"k{k}"] = k_results
    
    # ================================================================
    # FINAL SUMMARY ACROSS ALL K
    # ================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"FINAL SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'K':>3} {'Chance':>8} {'Bigram':>8} {'Emb':>8} {'Context':>8} "
                f"{'Probe':>8} {'Gap(P-C)':>10} {'Gap(P-B)':>10}")
    logger.info("-" * 70)
    
    for k in K_VALUES:
        key = f"k{k}"
        if key not in all_results:
            continue
        r = all_results[key]
        best = max(r["probe_by_layer"].items(), key=lambda x: x[1]["probe"])
        logger.info(f"{k:>3} {r['chance_majority']:>8.4f} {r['bigram']:>8.4f} "
                    f"{r['token_n_embedding']:>8.4f} {r['context_window_embedding']:>8.4f} "
                    f"{best[1]['probe']:>8.4f} {best[1]['gap_vs_context']:>+10.4f} "
                    f"{best[1]['gap_vs_bigram']:>+10.4f}")
    
    # Save
    outfile = "results/lookahead/final/future_lens_v4_final.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved to {outfile}")
    
    logger.info("\n" + "=" * 70)
    logger.info("DONE — FUTURE LENS v4 FINAL")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
