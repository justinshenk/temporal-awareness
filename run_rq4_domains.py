#!/usr/bin/env python3
"""
INTERMEDIATE DOMAINS: BASELINE STAIRCASE ACROSS DOMAINS
=========================================================
Opus said: "Natural-text-vs-code is too clean a dichotomy."

Tests the baseline staircase on 5 domains with varying structural predictability:
1. FREE PROSE — low structure (essays, stories)
2. STRUCTURED PROSE — medium structure (recipes, instructions)
3. CHAIN-OF-THOUGHT MATH — high structure (step-by-step reasoning)
4. CODE — highest structure (Python functions)
5. POETRY — constrained structure (meter, rhyme)

Prediction: probe-vs-baseline gap tracks structural predictability.
If true, the staircase is a general diagnostic, not code-specific.

Uses GPT-J-6B. Predicts next token (K=1) and K=3.
Fair baselines: token-N embedding, context-window embedding (same dim as probe).
"""

import json, os, sys, time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

MODEL_NAME = "EleutherAI/gpt-j-6b"
PCA_DIM = 128
N_GEN = 60
MIN_TARGET = 8
K_VALUES = [1, 3]

# ================================================================
# DOMAIN PROMPTS (10 per domain, varied)
# ================================================================

DOMAINS = {
    "free_prose": [
        "The old man sat by the river and watched the sun slowly",
        "She had always dreamed of traveling to distant lands where",
        "In the quiet moments before dawn the world seemed to hold",
        "He remembered the summer they spent together at the lake",
        "The city streets were empty except for a few stray cats",
        "Growing up in a small town meant everyone knew your",
        "The rain had been falling for three days straight and the",
        "After years of searching she finally found what she was",
        "The library was his favorite place in the world because",
        "When the music stopped everyone turned to look at the",
    ],
    "structured_prose": [
        "Step 1: Preheat the oven to 350 degrees. Step 2:",
        "To assemble the furniture, first lay out all the pieces",
        "WARNING: Before operating this device, ensure that the power",
        "Instructions for use: Apply a thin layer of the solution",
        "Day 1 of the workout plan: Begin with a five minute",
        "Section 3.2: The applicant must submit all required documents",
        "Ingredients: 2 cups flour, 1 cup sugar, 3 eggs.",
        "Troubleshooting guide: If the device does not turn on check",
        "Meeting agenda: 1. Review of previous minutes. 2. Budget",
        "Installation guide: First, download the latest version from",
    ],
    "chain_of_thought": [
        "Question: What is 247 + 389? Let me solve this step by step.",
        "Problem: If a train travels at 60 mph for 3 hours, how far does it go? Solution:",
        "Calculate: 15% of 240. Step 1: Convert 15% to decimal:",
        "Question: How many seconds are in 3.5 hours? Let me think through this.",
        "If x + 7 = 15, what is x? Let me solve: First, subtract",
        "Problem: A rectangle has length 12 and width 8. Find the area.",
        "Calculate the average of 23, 45, 67, and 89. First, add them:",
        "Question: What is 1000 minus 347? Let me work through this:",
        "If 3 apples cost $2.25, how much does one apple cost? Solution:",
        "Problem: Convert 72 degrees Fahrenheit to Celsius. Formula: C =",
    ],
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "def sort_list(items):\n    for i in range(len(items)):\n        for",
        "def is_prime(n):\n    if n < 2:\n        return False\n    for",
        "class Stack:\n    def __init__(self):\n        self.items = []\n    def",
        "def binary_search(arr, target):\n    left, right = 0, len(arr)",
        "def reverse_string(s):\n    result = ''\n    for char in",
        "def count_words(text):\n    words = text.split()\n    return",
        "def merge_lists(a, b):\n    result = []\n    i, j = 0,",
        "def factorial(n):\n    if n == 0:\n        return 1\n    return",
        "def flatten(nested):\n    result = []\n    for item in nested:",
    ],
    "poetry": [
        "Roses are red, violets are blue, sugar is sweet and",
        "Once upon a midnight dreary, while I pondered weak and",
        "Shall I compare thee to a summer day? Thou art more",
        "Two roads diverged in a yellow wood, and sorry I could",
        "The fog comes on little cat feet. It sits looking over",
        "I wandered lonely as a cloud that floats on high over",
        "Do not go gentle into that good night. Old age should",
        "Because I could not stop for Death, he kindly stopped for",
        "Twas brillig and the slithy toves did gyre and gimble in",
        "A thing of beauty is a joy forever. Its loveliness increases",
    ],
}


def run_domain(model, domain_name, prompts, layers, W_E):
    """Run staircase on one domain."""
    logger.info(f"\n  {'='*50}")
    logger.info(f"  DOMAIN: {domain_name} ({len(prompts)} prompts)")
    logger.info(f"  {'='*50}")
    
    # Generate
    all_sequences = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            gen = model.generate(tokens, max_new_tokens=N_GEN, temperature=0.0)
        all_sequences.append({
            "prompt_len": tokens.shape[1],
            "full_ids": gen[0].cpu().tolist()
        })
    
    # Show sample
    sample_text = model.to_string(torch.tensor(all_sequences[0]["full_ids"][all_sequences[0]["prompt_len"]:all_sequences[0]["prompt_len"]+15]))
    logger.info(f"  Sample: ...{sample_text[:60]}")
    
    # Split: train=0-4, test=5-9
    train_seqs = all_sequences[:5]
    test_seqs = all_sequences[5:]
    
    domain_results = {"domain": domain_name, "n_prompts": len(prompts)}
    
    for k in K_VALUES:
        # Find frequent targets in test
        test_targets = []
        for seq in test_seqs:
            pl = seq["prompt_len"]
            ids = seq["full_ids"]
            for n in range(pl, len(ids) - k):
                test_targets.append(ids[n + k])
        
        target_counts = Counter(test_targets)
        frequent = {t for t, c in target_counts.items() if c >= MIN_TARGET}
        t2i = {t: i for i, t in enumerate(sorted(frequent))}
        n_classes = len(t2i)
        
        if n_classes < 3:
            logger.info(f"    K={k}: only {n_classes} classes, skipping")
            continue
        
        # Extract test data
        activations = {l: [] for l in layers}
        token_embs = []
        ctx_embs = []
        labels = []
        
        for seq in test_seqs:
            ids = seq["full_ids"]
            pl = seq["prompt_len"]
            full_input = torch.tensor([ids], device="cuda")
            
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    full_input,
                    names_filter=[f"blocks.{l}.hook_resid_post" for l in layers])
            
            for n in range(pl, len(ids) - k):
                target = ids[n + k]
                if target not in t2i:
                    continue
                labels.append(t2i[target])
                for layer in layers:
                    activations[layer].append(
                        cache[f"blocks.{layer}.hook_resid_post"][0, n, :].cpu().numpy())
                token_embs.append(W_E[ids[n]].cpu().numpy())
                ws = max(0, n - 4)
                ctx = W_E[torch.tensor(ids[ws:n+1], device="cuda")].cpu().numpy()
                ctx_embs.append(ctx.mean(axis=0))
            
            del cache
            torch.cuda.empty_cache()
        
        labels = np.array(labels)
        n_ex = len(labels)
        class_counts = Counter(labels)
        min_class = min(class_counts.values())
        chance = max(class_counts.values()) / n_ex
        
        n_splits = min(5, min_class)
        if n_splits < 2:
            logger.info(f"    K={k}: min class {min_class}, skipping")
            continue
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Context embedding baseline
        X_ctx = np.stack(ctx_embs)
        scaler = StandardScaler()
        X_ctx_s = scaler.fit_transform(X_ctx)
        X_ctx_p = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(X_ctx_s)
        ctx_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
            X_ctx_p, labels, cv=cv, scoring="accuracy")
        ctx_acc = ctx_scores.mean()
        
        # Best probe
        best_probe = 0
        best_layer = 0
        for layer in layers:
            X_p = np.stack(activations[layer])
            scaler_p = StandardScaler()
            X_p_s = scaler_p.fit_transform(X_p)
            X_p_pca = PCA(n_components=min(PCA_DIM, n_ex-1),
                          random_state=42).fit_transform(X_p_s)
            p_scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                X_p_pca, labels, cv=cv, scoring="accuracy")
            if p_scores.mean() > best_probe:
                best_probe = p_scores.mean()
                best_layer = layer
        
        gap = best_probe - ctx_acc
        
        logger.info(f"    K={k}: {n_ex} ex, {n_classes} cls | "
                    f"chance={chance:.3f} ctx={ctx_acc:.3f} "
                    f"probe(L{best_layer})={best_probe:.3f} gap={gap:+.3f}")
        
        domain_results[f"k{k}"] = {
            "n_examples": int(n_ex), "n_classes": int(n_classes),
            "chance": float(chance), "context": float(ctx_acc),
            "best_probe": float(best_probe), "best_layer": int(best_layer),
            "gap": float(gap),
        }
    
    return domain_results


def main():
    logger.info("=" * 70)
    logger.info("INTERMEDIATE DOMAINS EXPERIMENT")
    logger.info("=" * 70)
    
    from transformer_lens import HookedTransformer
    
    logger.info("Loading GPT-J-6B...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda", dtype=torch.float16)
    model.eval()
    n_layers = model.cfg.n_layers
    
    layers = sorted(set([0, n_layers//6, n_layers//3, n_layers//2,
                         2*n_layers//3, 5*n_layers//6, n_layers-1]))
    
    W_E = model.W_E.detach()
    
    all_results = {"model": MODEL_NAME}
    
    for domain_name, prompts in DOMAINS.items():
        all_results[domain_name] = run_domain(model, domain_name, prompts, layers, W_E)
    
    # Cross-domain summary
    logger.info(f"\n{'='*70}")
    logger.info("CROSS-DOMAIN SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'Domain':<20} {'K':>3} {'Chance':>8} {'Context':>8} {'Probe':>8} {'Gap':>8}")
    logger.info("-" * 60)
    
    for domain_name in DOMAINS:
        for k in K_VALUES:
            key = f"k{k}"
            if key in all_results[domain_name]:
                r = all_results[domain_name][key]
                logger.info(f"{domain_name:<20} {k:>3} {r['chance']:>8.3f} "
                            f"{r['context']:>8.3f} {r['best_probe']:>8.3f} "
                            f"{r['gap']:>+8.3f}")
    
    # Save
    outfile = "results/lookahead/final/intermediate_domains.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved to {outfile}")
    
    logger.info("\n" + "=" * 70)
    logger.info("DONE — INTERMEDIATE DOMAINS")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
