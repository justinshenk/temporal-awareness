#!/usr/bin/env python3
"""
FUTURE LENS REPLICATION v3 — RIGOROUS
=======================================
Fixes from v2:
1. EXACT token prediction (not categories) — matches Pal et al.
2. FAIR baselines — same dimensionality as probe (embedding-based)
3. Proper scale — 200 prompts × 100 generated tokens = 20,000 prediction points
4. Filter to tokens appearing ≥10 times as targets → tractable classification
5. Fix convergence — max_iter=2000, saga solver

Baseline staircase (all using d_model-dimensional features):
- Chance: majority class frequency
- Token-N embedding: embedding of current token (same dim as probe)
- Context-window embedding: mean of last 5 token embeddings (same dim as probe)  
- Probe: residual stream at layer L, position N (same dim)

The key test: does PROBE beat EMBEDDING BASELINES?
- If yes → model's internal processing adds future info beyond token identity
- If no → "future prediction" is just token co-occurrence statistics

Model: GPT-J-6B (same as Pal et al.)
"""

import json, os, sys, time, random
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

MODEL_NAME = "EleutherAI/gpt-j-6b"
PCA_DIM = 128
N_GEN_TOKENS = 80       # generate 80 tokens per prompt
MIN_TARGET_COUNT = 10    # only predict tokens appearing ≥10 times as targets

# Diverse seed prompts — mix of domains
PROMPTS = [
    # Science
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
    # History
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
    # Technology
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
    # General knowledge
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
    # Code-like / technical
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


def main():
    logger.info("=" * 70)
    logger.info("FUTURE LENS v3 — RIGOROUS REPLICATION")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Prompts: {len(PROMPTS)}, Gen tokens: {N_GEN_TOKENS}")
    logger.info(f"Expected prediction points: ~{len(PROMPTS) * N_GEN_TOKENS}")
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
    logger.info(f"  Probing layers: {layers}")
    
    # ================================================================
    # STEP 1: Generate continuations
    # ================================================================
    logger.info(f"\n  Generating {N_GEN_TOKENS} tokens for {len(PROMPTS)} prompts...")
    
    all_sequences = []  # list of (prompt_ids, gen_ids) 
    for pi, prompt in enumerate(PROMPTS):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        prompt_len = tokens.shape[1]
        
        with torch.no_grad():
            gen = model.generate(tokens, max_new_tokens=N_GEN_TOKENS, temperature=0.0)
        
        full_ids = gen[0].cpu().tolist()
        all_sequences.append({
            "prompt_len": prompt_len,
            "full_ids": full_ids,
        })
        
        if pi < 2:
            gen_text = model.to_string(gen[0, prompt_len:prompt_len+20])
            logger.info(f"    [{pi}] ...{gen_text[:60]}")
        if (pi + 1) % 10 == 0:
            logger.info(f"    Generated {pi+1}/{len(PROMPTS)}")
    
    # ================================================================
    # STEP 2: For each K, collect prediction data
    # ================================================================
    all_results = {"model": MODEL_NAME, "n_layers": n_layers, "d_model": d_model}
    
    for k in [1, 2, 3]:
        logger.info(f"\n{'='*60}")
        logger.info(f"FUTURE LENS: PREDICTING TOKEN AT N+{k}")
        logger.info(f"{'='*60}")
        
        # Collect all (position N) → (target token at N+K) pairs
        # For each, we need: activation at N, embedding at N, context embeddings
        
        # First pass: just collect target tokens to find frequent ones
        target_token_ids = []
        for seq in all_sequences:
            pl = seq["prompt_len"]
            ids = seq["full_ids"]
            # Prediction points: every position from prompt_len to end-K
            for n in range(pl, len(ids) - k):
                target_token_ids.append(ids[n + k])
        
        target_counts = Counter(target_token_ids)
        frequent_targets = {t for t, c in target_counts.items() if c >= MIN_TARGET_COUNT}
        n_classes = len(frequent_targets)
        t2i = {t: i for i, t in enumerate(sorted(frequent_targets))}
        
        # Show top targets
        top10 = target_counts.most_common(10)
        top10_str = [(model.to_string(torch.tensor([t])), c) for t, c in top10]
        logger.info(f"  Total prediction points: {len(target_token_ids)}")
        logger.info(f"  Unique target tokens: {len(target_counts)}")
        logger.info(f"  Frequent targets (≥{MIN_TARGET_COUNT}): {n_classes}")
        logger.info(f"  Top-10 targets: {top10_str}")
        
        if n_classes < 5:
            logger.warning(f"  Too few classes, skipping K={k}")
            continue
        
        # Second pass: extract activations + embeddings for frequent targets only
        logger.info(f"\n  Extracting activations for {n_classes}-class prediction...")
        
        activations = {l: [] for l in layers}
        token_n_embeddings = []      # embedding of token at position N
        context_window_embs = []     # mean embedding of tokens N-4..N
        labels = []
        
        W_E = model.W_E.detach()  # [vocab, d_model]
        
        for si, seq in enumerate(all_sequences):
            pl = seq["prompt_len"]
            ids = seq["full_ids"]
            
            # Run full sequence through model once
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
                
                # Probe features: activation at position N, each layer
                for layer in layers:
                    act = cache[f"blocks.{layer}.hook_resid_post"][0, n, :].cpu().numpy()
                    activations[layer].append(act)
                
                # Baseline 1: embedding of token at position N
                tok_emb = W_E[ids[n]].cpu().numpy()
                token_n_embeddings.append(tok_emb)
                
                # Baseline 2: mean embedding of context window (N-4 to N)
                window_start = max(0, n - 4)
                window_ids = ids[window_start:n+1]
                ctx_embs = W_E[torch.tensor(window_ids, device="cuda")].cpu().numpy()
                context_window_embs.append(ctx_embs.mean(axis=0))
            
            del cache
            torch.cuda.empty_cache()
            
            if (si + 1) % 10 == 0:
                logger.info(f"    Processed {si+1}/{len(all_sequences)}, "
                            f"collected {len(labels)} examples")
        
        labels = np.array(labels)
        n_examples = len(labels)
        logger.info(f"  Total examples: {n_examples}, classes: {n_classes}")
        
        # Check class balance
        class_counts = Counter(labels)
        min_class = min(class_counts.values())
        max_class = max(class_counts.values())
        logger.info(f"  Class sizes: min={min_class}, max={max_class}")
        
        n_splits = min(5, min_class)
        if n_splits < 2:
            logger.warning(f"  Min class too small for CV, skipping")
            continue
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        chance_acc = max_class / n_examples  # majority class
        
        k_results = {
            "n_examples": int(n_examples),
            "n_classes": int(n_classes),
            "chance_majority": float(chance_acc),
            "chance_uniform": float(1/n_classes),
            "min_class_size": int(min_class),
        }
        
        # ============================================================
        # BASELINE 1: Token-N embedding (FAIR — same dim as probe)
        # ============================================================
        logger.info(f"\n  --- Baseline: Token-N embedding (d={d_model}) ---")
        X_emb = np.stack(token_n_embeddings)
        scaler_emb = StandardScaler()
        X_emb_s = scaler_emb.fit_transform(X_emb)
        X_emb_pca = PCA(n_components=min(PCA_DIM, X_emb_s.shape[0]-1),
                        random_state=42).fit_transform(X_emb_s)
        
        emb_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=2000, solver="saga", n_jobs=-1),
            X_emb_pca, labels, cv=cv, scoring="accuracy")
        emb_acc = emb_scores.mean()
        logger.info(f"    Token-N embedding: {emb_acc:.4f}")
        k_results["token_n_embedding"] = float(emb_acc)
        
        # ============================================================
        # BASELINE 2: Context-window embedding (FAIR — same dim)
        # ============================================================
        logger.info(f"  --- Baseline: Context-window embedding (last 5, d={d_model}) ---")
        X_ctx = np.stack(context_window_embs)
        scaler_ctx = StandardScaler()
        X_ctx_s = scaler_ctx.fit_transform(X_ctx)
        X_ctx_pca = PCA(n_components=min(PCA_DIM, X_ctx_s.shape[0]-1),
                        random_state=42).fit_transform(X_ctx_s)
        
        ctx_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=2000, solver="saga", n_jobs=-1),
            X_ctx_pca, labels, cv=cv, scoring="accuracy")
        ctx_acc = ctx_scores.mean()
        logger.info(f"    Context-window embedding: {ctx_acc:.4f}")
        k_results["context_window_embedding"] = float(ctx_acc)
        
        # ============================================================
        # PROBE: Each layer (same dim after PCA)
        # ============================================================
        logger.info(f"\n  --- Probe: Layer activations ---")
        logger.info(f"  {'Layer':>6} {'Probe':>8} {'vs Emb':>8} {'vs Ctx':>8} {'vs Chance':>10}")
        
        probe_results = {}
        for layer in layers:
            X_probe = np.stack(activations[layer])
            scaler_p = StandardScaler()
            X_p_s = scaler_p.fit_transform(X_probe)
            X_p_pca = PCA(n_components=min(PCA_DIM, X_p_s.shape[0]-1),
                          random_state=42).fit_transform(X_p_s)
            
            p_scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=2000, solver="saga", n_jobs=-1),
                X_p_pca, labels, cv=cv, scoring="accuracy")
            p_acc = p_scores.mean()
            
            gap_emb = p_acc - emb_acc
            gap_ctx = p_acc - ctx_acc
            gap_chance = p_acc - chance_acc
            
            logger.info(f"  L{layer:>4} {p_acc:>8.4f} {gap_emb:>+8.4f} {gap_ctx:>+8.4f} {gap_chance:>+10.4f}")
            
            probe_results[str(layer)] = {
                "probe": float(p_acc),
                "gap_vs_embedding": float(gap_emb),
                "gap_vs_context": float(gap_ctx),
                "gap_vs_chance": float(gap_chance),
            }
        
        k_results["probe_by_layer"] = probe_results
        
        # ============================================================
        # SUMMARY
        # ============================================================
        best_layer = max(probe_results.items(), key=lambda x: x[1]["probe"])
        best_probe = best_layer[1]["probe"]
        best_gap_ctx = best_layer[1]["gap_vs_context"]
        
        logger.info(f"\n  === STAIRCASE SUMMARY (K={k}) ===")
        logger.info(f"  Chance (majority):      {chance_acc:.4f}")
        logger.info(f"  Chance (uniform):       {1/n_classes:.4f}")
        logger.info(f"  Token-N embedding:      {emb_acc:.4f}")
        logger.info(f"  Context-window emb:     {ctx_acc:.4f}")
        logger.info(f"  Best Probe (L{best_layer[0]}):     {best_probe:.4f}")
        logger.info(f"  Gap (probe − context):  {best_gap_ctx:+.4f}")
        
        if best_gap_ctx < 0.02:
            logger.info(f"  >>> CONTEXT EMBEDDING EXPLAINS PROBE SIGNAL <<<")
        elif best_gap_ctx < 0.05:
            logger.info(f"  >>> SMALL RESIDUAL — probe slightly exceeds context <<<")
        else:
            logger.info(f"  >>> PROBE EXCEEDS CONTEXT BY {best_gap_ctx:+.4f} — possible planning <<<")
        
        all_results[f"k{k}"] = k_results
    
    # Save
    outfile = "results/lookahead/final/future_lens_v3_rigorous.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved to {outfile}")
    
    logger.info("\n" + "=" * 70)
    logger.info("DONE — FUTURE LENS v3 RIGOROUS")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
