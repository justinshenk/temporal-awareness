#!/usr/bin/env python3
"""
FUTURE LENS REPLICATION WITH BASELINE STAIRCASE
=================================================
Replicates the core protocol from Pal et al. (2023) "Future Lens: 
Anticipating Subsequent Tokens from a Single Hidden State"
on GPT-J-6B, then applies our baseline staircase to test whether 
context features explain away the "future prediction" signal.

Protocol:
1. Take natural text prompts, split into tokens
2. At position N, probe intermediate layer activations to predict token at N+K
3. Compare: does a context baseline (BoW of tokens 0..N) predict N+K equally well?

Baseline staircase:
- Chance: 1/V (vocabulary size) or top-1 frequency
- Last-token baseline: predict N+K from just embedding of token N
- Context BoW: predict N+K from bag-of-words of tokens 0..N  
- Context window: predict N+K from embeddings of tokens N-3..N
- Probe: predict N+K from intermediate layer activation at position N

If context baselines match the probe → Future Lens detects context, not planning.

Models: GPT-J-6B (same as Pal et al.)
Dataset: OpenWebText samples (same domain as training data)
"""

import json, os, sys, hashlib, time, random
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

# ================================================================
# CONFIG
# ================================================================
MODEL_NAME = "EleutherAI/gpt-j-6b"
N_PROMPTS = 200         # number of text passages
CONTEXT_LEN = 50        # tokens of context before prediction point
K_VALUES = [1, 2, 3]    # predict token N+1, N+2, N+3
TOP_K_CLASSES = 500      # reduce vocabulary to top-500 most frequent next-tokens
PCA_DIM = 128
N_BOOTSTRAP = 300

# ================================================================
# DATASET: Generate natural text prompts
# ================================================================

# We'll use diverse prompt templates that create natural continuation contexts
# These simulate the kind of text GPT-J was trained on
SEED_PROMPTS = [
    "The capital of France is Paris, which is known for",
    "In 2020, the world experienced a global pandemic that",
    "The theory of relativity was developed by Albert Einstein who",
    "Machine learning algorithms can be broadly categorized into supervised and",
    "The Python programming language was created by Guido van Rossum in",
    "Climate change is one of the most pressing issues facing humanity because",
    "The United States Constitution was ratified in 1788 and established",
    "Photosynthesis is the process by which plants convert sunlight into",
    "The stock market crashed in 1929 leading to the Great Depression which",
    "Artificial intelligence has made significant progress in recent years especially in",
    "The human brain contains approximately 86 billion neurons that communicate through",
    "Shakespeare wrote many famous plays including Hamlet and Romeo and",
    "The speed of light in a vacuum is approximately 299792458 meters per",
    "DNA stores genetic information using four nucleotide bases adenine thymine guanine and",
    "The Renaissance was a period of cultural rebirth in Europe that began in",
    "Quantum mechanics describes the behavior of particles at the atomic and subatomic",
    "The Amazon rainforest is the largest tropical rainforest in the world covering",
    "World War II ended in 1945 after the Allied forces defeated",
    "The periodic table organizes chemical elements by their atomic number and",
    "Neural networks are inspired by the structure and function of the human",
    "The Industrial Revolution began in Britain in the late 18th century and",
    "Evolution by natural selection was proposed by Charles Darwin in his book",
    "The internet was originally developed by the US military as a communication",
    "Black holes are regions of spacetime where gravity is so strong that",
    "The Great Wall of China was built over many centuries to protect against",
    "Electricity is generated in power plants by converting various forms of energy into",
    "The Olympic Games originated in ancient Greece and were revived in",
    "Antibiotics are medications used to treat bacterial infections by either killing or",
    "The Mona Lisa was painted by Leonardo da Vinci and is displayed in",
    "Volcanic eruptions occur when magma from beneath the earth surface rises through",
    "The solar system consists of the sun and eight planets including",
    "Democracy is a system of government in which power is held by the",
    "The Great Barrier Reef is the world largest coral reef system located off the coast of",
    "Inflation occurs when the general price level of goods and services rises causing",
    "The theory of evolution explains how species change over time through",
    "Vaccines work by stimulating the immune system to recognize and fight specific",
    "The Sahara Desert is the largest hot desert in the world spanning",
    "Gravity is a fundamental force that attracts objects with mass toward each",
    "The printing press was invented by Johannes Gutenberg in the 15th century and",
    "Plate tectonics describes the movement of large sections of the earth crust called",
    "The French Revolution began in 1789 and led to major political changes including",
    "Artificial neural networks consist of layers of interconnected nodes that process",
    "The human genome contains approximately 3 billion base pairs of DNA organized into",
    "Cryptocurrency is a digital currency that uses cryptography for security and operates on",
    "The Renaissance artists including Michelangelo and Raphael created works that",
    "Superconductivity is a phenomenon where certain materials exhibit zero electrical resistance when cooled below",
    "The United Nations was established in 1945 to promote international cooperation and",
    "Photovoltaic cells convert sunlight directly into electricity using semiconductor materials such as",
    "The Roman Empire at its peak controlled territory spanning from Britain to",
    "CRISPR is a revolutionary gene editing technology that allows scientists to modify",
]


def generate_continuations(model, tokenizer, n_prompts=N_PROMPTS, context_len=CONTEXT_LEN):
    """Generate text continuations to create natural prompt-continuation pairs."""
    logger.info(f"  Generating {n_prompts} continuations of length {context_len + max(K_VALUES) + 5}...")
    
    all_data = []
    prompts_per_seed = max(1, n_prompts // len(SEED_PROMPTS)) + 1
    
    for seed_idx, seed in enumerate(SEED_PROMPTS):
        if len(all_data) >= n_prompts:
            break
            
        tokens = model.to_tokens(seed, prepend_bos=True)
        gen_len = context_len + max(K_VALUES) + 10
        
        # Generate with some temperature for diversity
        with torch.no_grad():
            for temp in [0.0, 0.7, 1.0]:
                if len(all_data) >= n_prompts:
                    break
                    
                if temp == 0.0:
                    gen_tokens = model.generate(tokens, max_new_tokens=gen_len, temperature=0.0)
                else:
                    gen_tokens = model.generate(tokens, max_new_tokens=gen_len, temperature=temp, top_k=50)
                
                full_tokens = gen_tokens[0].cpu().tolist()
                
                if len(full_tokens) >= context_len + max(K_VALUES) + 5:
                    all_data.append(full_tokens)
        
        if (seed_idx + 1) % 10 == 0:
            logger.info(f"    Seeds processed: {seed_idx + 1}/{len(SEED_PROMPTS)}, data: {len(all_data)}")
    
    # If we need more, generate from random starting points
    while len(all_data) < n_prompts:
        seed = random.choice(SEED_PROMPTS)
        tokens = model.to_tokens(seed, prepend_bos=True)
        with torch.no_grad():
            gen_tokens = model.generate(tokens, max_new_tokens=context_len + max(K_VALUES) + 10, 
                                        temperature=0.8, top_k=50)
        full_tokens = gen_tokens[0].cpu().tolist()
        if len(full_tokens) >= context_len + max(K_VALUES) + 5:
            all_data.append(full_tokens)
    
    logger.info(f"  Generated {len(all_data)} continuations")
    return all_data[:n_prompts]


def extract_prediction_data(model, token_sequences, context_len, k_values):
    """
    For each sequence, extract:
    - Activations at position N (context_len) for each layer
    - Target token at position N+K
    - Context tokens 0..N for baseline
    """
    n_layers = model.cfg.n_layers
    layers = sorted(set([0, n_layers//6, n_layers//3, n_layers//2, 
                         2*n_layers//3, 5*n_layers//6, n_layers-1]))
    
    logger.info(f"  Extracting activations at {len(layers)} layers for {len(token_sequences)} sequences...")
    
    # For each K value, collect: activations, target tokens, context tokens
    data = {k: {"activations": {l: [] for l in layers}, 
                "targets": [], 
                "context_tokens": [],
                "last_token_emb": []} 
            for k in k_values}
    
    for i, seq in enumerate(token_sequences):
        tokens = torch.tensor([seq[:context_len + max(k_values) + 1]], device="cuda")
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[f"blocks.{l}.hook_resid_post" for l in layers]
            )
        
        for k in k_values:
            N = context_len  # prediction point
            target_pos = N + k
            
            if target_pos < len(seq):
                target_token = seq[target_pos]
                data[k]["targets"].append(target_token)
                data[k]["context_tokens"].append(seq[1:N+1])  # skip BOS
                
                for layer in layers:
                    act = cache[f"blocks.{layer}.hook_resid_post"][0, N, :].cpu().numpy()
                    data[k]["activations"][layer].append(act)
                
                # Last token embedding (for last-token baseline)
                emb = cache[f"blocks.0.hook_resid_post"][0, N, :].cpu().numpy()
                data[k]["last_token_emb"].append(emb)
        
        del cache
        torch.cuda.empty_cache()
        
        if (i + 1) % 50 == 0:
            logger.info(f"    Extracted {i+1}/{len(token_sequences)}")
    
    return data, layers


def build_context_features(context_tokens_list, model, max_vocab=50000):
    """Build BoW features from context tokens."""
    # Simple: count occurrences of each token in context
    n = len(context_tokens_list)
    
    # Find all unique tokens
    all_tokens = set()
    for ctx in context_tokens_list:
        all_tokens.update(ctx)
    
    # Limit to most common
    token_counts = Counter()
    for ctx in context_tokens_list:
        token_counts.update(ctx)
    top_tokens = [t for t, _ in token_counts.most_common(min(5000, len(all_tokens)))]
    t2i = {t: i for i, t in enumerate(top_tokens)}
    
    X = np.zeros((n, len(top_tokens)), dtype=np.float32)
    for i, ctx in enumerate(context_tokens_list):
        for t in ctx:
            if t in t2i:
                X[i, t2i[t]] += 1
    
    return X


def build_window_features(context_tokens_list, model, window=5):
    """Build features from last W tokens of context (embeddings concatenated)."""
    d_model = model.cfg.d_model
    W = model.W_E  # embedding matrix
    
    features = []
    for ctx in context_tokens_list:
        last_w = ctx[-window:] if len(ctx) >= window else ctx
        embs = W[torch.tensor(last_w, device="cuda")].cpu().numpy()  # [W, d_model]
        # Mean pool
        feat = embs.mean(axis=0)
        features.append(feat)
    
    return np.stack(features)


def run_future_lens_staircase(model, token_sequences, context_len=CONTEXT_LEN):
    """Main experiment: Future Lens with baseline staircase."""
    
    logger.info("=" * 70)
    logger.info("FUTURE LENS REPLICATION WITH BASELINE STAIRCASE")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Context length: {context_len}, K values: {K_VALUES}")
    logger.info("=" * 70)
    
    # Extract data
    data, layers = extract_prediction_data(model, token_sequences, context_len, K_VALUES)
    
    all_results = {}
    
    for k in K_VALUES:
        logger.info(f"\n{'='*60}")
        logger.info(f"PREDICTING TOKEN N+{k}")
        logger.info(f"{'='*60}")
        
        targets = np.array(data[k]["targets"])
        n_examples = len(targets)
        logger.info(f"  {n_examples} examples")
        
        # Reduce to top-K most frequent target classes
        target_counts = Counter(targets)
        top_targets = [t for t, _ in target_counts.most_common(TOP_K_CLASSES)]
        top_set = set(top_targets)
        
        # Filter to only examples with top-K targets
        mask = np.array([t in top_set for t in targets])
        targets_filtered = targets[mask]
        
        # Re-map to class indices
        t2i = {t: i for i, t in enumerate(sorted(set(targets_filtered)))}
        labels = np.array([t2i[t] for t in targets_filtered])
        n_classes = len(t2i)
        n_filtered = len(labels)
        
        logger.info(f"  After filtering to top-{TOP_K_CLASSES}: {n_filtered} examples, {n_classes} classes")
        logger.info(f"  Chance accuracy: {1/n_classes:.4f}")
        
        if n_filtered < 50 or n_classes < 5:
            logger.warning(f"  Too few examples or classes, skipping K={k}")
            continue
        
        # Ensure minimum class size for CV
        min_class_size = min(Counter(labels).values())
        n_splits = min(5, min_class_size)
        if n_splits < 2:
            logger.warning(f"  Min class size {min_class_size} too small for CV, skipping K={k}")
            continue
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        k_results = {
            "n_examples": int(n_filtered),
            "n_classes": int(n_classes),
            "chance": float(1/n_classes),
            "k": int(k),
        }
        
        # ============================================================
        # BASELINE 1: Context BoW
        # ============================================================
        logger.info(f"\n  --- Baseline: Context BoW ---")
        ctx_filtered = [data[k]["context_tokens"][i] for i in range(n_examples) if mask[i]]
        X_bow = build_context_features(ctx_filtered, model)
        
        scaler_bow = StandardScaler()
        X_bow_s = scaler_bow.fit_transform(X_bow)
        if X_bow_s.shape[1] > PCA_DIM:
            X_bow_s = PCA(n_components=min(PCA_DIM, X_bow_s.shape[0]-1), random_state=42).fit_transform(X_bow_s)
        
        bow_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
            X_bow_s, labels, cv=cv, scoring="accuracy")
        bow_acc = bow_scores.mean()
        logger.info(f"    BoW accuracy: {bow_acc:.4f}")
        k_results["bow"] = float(bow_acc)
        
        # ============================================================
        # BASELINE 2: Context window (last 5 tokens, mean embedding)
        # ============================================================
        logger.info(f"\n  --- Baseline: Context Window (last 5 tokens) ---")
        X_window = build_window_features(ctx_filtered, model, window=5)
        
        scaler_win = StandardScaler()
        X_win_s = scaler_win.fit_transform(X_window)
        if X_win_s.shape[1] > PCA_DIM:
            X_win_s = PCA(n_components=min(PCA_DIM, X_win_s.shape[0]-1), random_state=42).fit_transform(X_win_s)
        
        win_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
            X_win_s, labels, cv=cv, scoring="accuracy")
        win_acc = win_scores.mean()
        logger.info(f"    Window accuracy: {win_acc:.4f}")
        k_results["context_window"] = float(win_acc)
        
        # ============================================================
        # BASELINE 3: Last-token embedding only
        # ============================================================
        logger.info(f"\n  --- Baseline: Last-token embedding (layer 0) ---")
        X_last = np.stack([data[k]["last_token_emb"][i] for i in range(n_examples) if mask[i]])
        
        scaler_last = StandardScaler()
        X_last_s = scaler_last.fit_transform(X_last)
        if X_last_s.shape[1] > PCA_DIM:
            X_last_s = PCA(n_components=min(PCA_DIM, X_last_s.shape[0]-1), random_state=42).fit_transform(X_last_s)
        
        last_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
            X_last_s, labels, cv=cv, scoring="accuracy")
        last_acc = last_scores.mean()
        logger.info(f"    Last-token accuracy: {last_acc:.4f}")
        k_results["last_token"] = float(last_acc)
        
        # ============================================================
        # PROBE: Each layer
        # ============================================================
        logger.info(f"\n  --- Probe: Layer activations ---")
        probe_results = {}
        
        for layer in layers:
            X_probe = np.stack([data[k]["activations"][layer][i] 
                               for i in range(n_examples) if mask[i]])
            
            scaler_p = StandardScaler()
            X_p_s = scaler_p.fit_transform(X_probe)
            if X_p_s.shape[1] > PCA_DIM:
                X_p_s = PCA(n_components=min(PCA_DIM, X_p_s.shape[0]-1), 
                           random_state=42).fit_transform(X_p_s)
            
            probe_scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
                X_p_s, labels, cv=cv, scoring="accuracy")
            probe_acc = probe_scores.mean()
            
            gap_bow = probe_acc - bow_acc
            gap_win = probe_acc - win_acc
            
            logger.info(f"    L{layer:>3}: probe={probe_acc:.4f} | "
                        f"gap_bow={gap_bow:+.4f} gap_win={gap_win:+.4f}")
            
            probe_results[str(layer)] = {
                "probe": float(probe_acc),
                "gap_vs_bow": float(gap_bow),
                "gap_vs_window": float(gap_win),
            }
        
        k_results["probe_by_layer"] = probe_results
        
        # ============================================================
        # SUMMARY for this K
        # ============================================================
        best_layer = max(probe_results.items(), key=lambda x: x[1]["probe"])
        best_probe = best_layer[1]["probe"]
        
        logger.info(f"\n  === STAIRCASE SUMMARY (K={k}) ===")
        logger.info(f"  Chance:          {1/n_classes:.4f}")
        logger.info(f"  Last-token:      {last_acc:.4f}")
        logger.info(f"  Context BoW:     {bow_acc:.4f}")
        logger.info(f"  Context Window:  {win_acc:.4f}")
        logger.info(f"  Best Probe (L{best_layer[0]}): {best_probe:.4f}")
        logger.info(f"  Gap (probe-BoW): {best_probe - bow_acc:+.4f}")
        logger.info(f"  Gap (probe-win): {best_probe - win_acc:+.4f}")
        
        if best_probe - win_acc < 0.02:
            logger.info(f"  >>> CONTEXT WINDOW EXPLAINS PROBE SIGNAL <<<")
        elif best_probe - bow_acc < 0.02:
            logger.info(f"  >>> BOW EXPLAINS PROBE SIGNAL <<<")
        else:
            logger.info(f"  >>> PROBE EXCEEDS BASELINES — possible planning signal <<<")
        
        all_results[f"k{k}"] = k_results
    
    return all_results


def main():
    logger.info("=" * 70)
    logger.info("FUTURE LENS REPLICATION — GPT-J-6B")
    logger.info("=" * 70)
    
    # Install transformer-lens if needed
    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        os.system("pip install transformer-lens transformers scikit-learn scipy --break-system-packages")
        from transformer_lens import HookedTransformer
    
    # Load model
    logger.info("Loading GPT-J-6B...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device="cuda", dtype=torch.float16
    )
    model.eval()
    logger.info(f"  Loaded in {time.time()-t0:.1f}s")
    logger.info(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    
    # Generate data
    logger.info("\nGenerating text continuations...")
    token_sequences = generate_continuations(model, model.tokenizer, n_prompts=N_PROMPTS)
    
    # Run staircase
    results = run_future_lens_staircase(model, token_sequences)
    
    # Save
    outdir = "results/lookahead/final"
    os.makedirs(outdir, exist_ok=True)
    outfile = f"{outdir}/future_lens_staircase.json"
    
    results["model"] = MODEL_NAME
    results["n_prompts"] = N_PROMPTS
    results["context_len"] = CONTEXT_LEN
    results["top_k_classes"] = TOP_K_CLASSES
    
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved to {outfile}")
    
    logger.info("\n" + "=" * 70)
    logger.info("DONE — FUTURE LENS STAIRCASE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
