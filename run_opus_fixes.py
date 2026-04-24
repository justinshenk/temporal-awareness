#!/usr/bin/env python3
"""
OPUS FIXES: Qwen2.5-7B + Scrambled CoT Ablation
==================================================
1. Future Lens on Qwen2.5-7B (7B+ model — "non-negotiable" per Opus)
2. Intermediate domains on Qwen2.5-7B
3. Scrambled CoT ablation on GPT-J-6B — does +28% survive template disruption?

These are the last two experiments needed for EMNLP submission.
"""

import json, os, sys, time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter, defaultdict

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

PCA_DIM = 128
N_GEN = 80
MIN_TARGET_FL = 15
MIN_TARGET_DOM = 8
K_VALUES_FL = [1, 2, 3, 5]
K_VALUES_DOM = [1, 3]

# ================================================================
# FUTURE LENS PROMPTS (50 train / 50 test)
# ================================================================
FL_PROMPTS = [
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

# ================================================================
# INTERMEDIATE DOMAIN PROMPTS
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
    "chain_of_thought_scrambled": [
        "Question: What is 247 + 389? Therefore the answer. Step 3: carry over.",
        "Problem: If a train travels at 60 mph for 3 hours? The answer is: Step 2 multiply",
        "Calculate: 15% of 240. The result equals. Step 2: Now convert back:",
        "Question: How many seconds are in 3.5 hours? The answer is. Step 1: multiply",
        "If x + 7 = 15, the answer is x equals. To verify: add back",
        "Problem: A rectangle has length 12 and width 8. Therefore area equals. Step 1:",
        "Calculate the average of 23, 45, 67, and 89. The average is. To compute: add",
        "Question: What is 1000 minus 347? Answer: the result. Step 2: subtract",
        "If 3 apples cost $2.25, the answer is. Step 3: so each costs",
        "Problem: Convert 72 degrees Fahrenheit to Celsius. Answer equals. Step 1: subtract",
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


def run_future_lens(model, model_name):
    """Run Future Lens staircase on a model."""
    logger.info(f"\n{'='*70}")
    logger.info(f"FUTURE LENS: {model_name}")
    logger.info(f"{'='*70}")
    
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    layers = sorted(set([0, n_layers//6, n_layers//3, n_layers//2,
                         2*n_layers//3, 5*n_layers//6, n_layers-1]))
    W_E = model.W_E.detach()
    
    # Generate
    logger.info(f"  Generating {N_GEN} tokens for {len(FL_PROMPTS)} prompts...")
    all_sequences = []
    for pi, prompt in enumerate(FL_PROMPTS):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            gen = model.generate(tokens, max_new_tokens=N_GEN, temperature=0.0)
        all_sequences.append({"prompt_len": tokens.shape[1], "full_ids": gen[0].cpu().tolist()})
        if (pi + 1) % 10 == 0:
            logger.info(f"    {pi+1}/{len(FL_PROMPTS)}")
    
    train_seqs = [all_sequences[i] for i in range(25)]
    test_seqs = [all_sequences[i] for i in range(25, 50)]
    
    results = {"model": model_name, "n_layers": n_layers, "d_model": d_model}
    
    for k in K_VALUES_FL:
        logger.info(f"\n  K={k}")
        
        # Build trigram on train
        trigram = defaultdict(Counter)
        for seq in train_seqs:
            ids = seq["full_ids"]; pl = seq["prompt_len"]
            for n in range(max(pl, 1), len(ids) - k):
                trigram[(ids[n-1], ids[n])][ids[n+k]] += 1
        
        # Find frequent targets in test
        test_tgts = []
        for seq in test_seqs:
            ids = seq["full_ids"]; pl = seq["prompt_len"]
            for n in range(pl, len(ids) - k):
                test_tgts.append(ids[n+k])
        
        tc = Counter(test_tgts)
        freq = {t for t, c in tc.items() if c >= MIN_TARGET_FL}
        t2i = {t: i for i, t in enumerate(sorted(freq))}
        n_cls = len(t2i)
        
        if n_cls < 5:
            logger.info(f"    {n_cls} classes, skip")
            continue
        
        # Extract test data
        activations = {l: [] for l in layers}
        token_embs, ctx_embs, labels = [], [], []
        
        for seq in test_seqs:
            ids = seq["full_ids"]; pl = seq["prompt_len"]
            inp = torch.tensor([ids], device="cuda")
            with torch.no_grad():
                _, cache = model.run_with_cache(inp,
                    names_filter=[f"blocks.{l}.hook_resid_post" for l in layers])
            for n in range(pl, len(ids) - k):
                tgt = ids[n+k]
                if tgt not in t2i: continue
                labels.append(t2i[tgt])
                for l in layers:
                    activations[l].append(cache[f"blocks.{l}.hook_resid_post"][0, n, :].cpu().numpy())
                token_embs.append(W_E[ids[n]].cpu().numpy())
                ws = max(0, n-4)
                ctx = W_E[torch.tensor(ids[ws:n+1], device="cuda")].cpu().numpy()
                ctx_embs.append(ctx.mean(axis=0))
            del cache; torch.cuda.empty_cache()
        
        labels = np.array(labels)
        n_ex = len(labels)
        cc = Counter(labels)
        chance = max(cc.values()) / n_ex
        min_c = min(cc.values())
        n_splits = min(5, min_c)
        if n_splits < 2: continue
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Context baseline
        X_ctx = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(
            StandardScaler().fit_transform(np.stack(ctx_embs)))
        ctx_acc = cross_val_score(LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                                  X_ctx, labels, cv=cv, scoring="accuracy").mean()
        
        # Best probe
        best_p, best_l = 0, 0
        for l in layers:
            X = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(
                StandardScaler().fit_transform(np.stack(activations[l])))
            acc = cross_val_score(LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                                  X, labels, cv=cv, scoring="accuracy").mean()
            if acc > best_p: best_p, best_l = acc, l
        
        gap = best_p - ctx_acc
        logger.info(f"    {n_ex} ex, {n_cls} cls | ctx={ctx_acc:.3f} probe(L{best_l})={best_p:.3f} gap={gap:+.3f}")
        results[f"k{k}"] = {"n_examples": n_ex, "n_classes": n_cls, "chance": float(chance),
                             "context": float(ctx_acc), "probe": float(best_p), "layer": best_l,
                             "gap": float(gap)}
    
    return results


def run_domains(model, model_name, domains):
    """Run intermediate domains staircase."""
    logger.info(f"\n{'='*70}")
    logger.info(f"INTERMEDIATE DOMAINS: {model_name}")
    logger.info(f"{'='*70}")
    
    n_layers = model.cfg.n_layers
    layers = sorted(set([0, n_layers//6, n_layers//3, n_layers//2,
                         2*n_layers//3, 5*n_layers//6, n_layers-1]))
    W_E = model.W_E.detach()
    
    results = {"model": model_name}
    
    for domain_name, prompts in domains.items():
        logger.info(f"\n  Domain: {domain_name}")
        
        # Generate
        all_seqs = []
        for prompt in prompts:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                gen = model.generate(tokens, max_new_tokens=60, temperature=0.0)
            all_seqs.append({"prompt_len": tokens.shape[1], "full_ids": gen[0].cpu().tolist()})
        
        train_seqs = all_seqs[:5]
        test_seqs = all_seqs[5:]
        
        domain_results = {}
        for k in K_VALUES_DOM:
            # Find frequent targets in test
            test_tgts = []
            for seq in test_seqs:
                ids = seq["full_ids"]; pl = seq["prompt_len"]
                for n in range(pl, len(ids) - k):
                    test_tgts.append(ids[n+k])
            
            tc = Counter(test_tgts)
            freq = {t for t, c in tc.items() if c >= MIN_TARGET_DOM}
            t2i = {t: i for i, t in enumerate(sorted(freq))}
            n_cls = len(t2i)
            
            if n_cls < 3:
                logger.info(f"    K={k}: {n_cls} classes, skip")
                continue
            
            # Extract
            activations = {l: [] for l in layers}
            ctx_embs, labels = [], []
            
            for seq in test_seqs:
                ids = seq["full_ids"]; pl = seq["prompt_len"]
                inp = torch.tensor([ids], device="cuda")
                with torch.no_grad():
                    _, cache = model.run_with_cache(inp,
                        names_filter=[f"blocks.{l}.hook_resid_post" for l in layers])
                for n in range(pl, len(ids) - k):
                    tgt = ids[n+k]
                    if tgt not in t2i: continue
                    labels.append(t2i[tgt])
                    for l in layers:
                        activations[l].append(cache[f"blocks.{l}.hook_resid_post"][0, n, :].cpu().numpy())
                    ws = max(0, n-4)
                    ctx = W_E[torch.tensor(ids[ws:n+1], device="cuda")].cpu().numpy()
                    ctx_embs.append(ctx.mean(axis=0))
                del cache; torch.cuda.empty_cache()
            
            labels = np.array(labels)
            n_ex = len(labels)
            if n_ex < 20: continue
            cc = Counter(labels)
            chance = max(cc.values()) / n_ex
            min_c = min(cc.values())
            n_splits = min(5, min_c)
            if n_splits < 2: continue
            
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            X_ctx = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(
                StandardScaler().fit_transform(np.stack(ctx_embs)))
            ctx_acc = cross_val_score(LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                                      X_ctx, labels, cv=cv, scoring="accuracy").mean()
            
            best_p, best_l = 0, 0
            for l in layers:
                X = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(
                    StandardScaler().fit_transform(np.stack(activations[l])))
                acc = cross_val_score(LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                                      X, labels, cv=cv, scoring="accuracy").mean()
                if acc > best_p: best_p, best_l = acc, l
            
            gap = best_p - ctx_acc
            logger.info(f"    K={k}: {n_ex} ex, {n_cls} cls | ctx={ctx_acc:.3f} probe(L{best_l})={best_p:.3f} gap={gap:+.3f}")
            domain_results[f"k{k}"] = {"n_examples": n_ex, "n_classes": n_cls, "chance": float(chance),
                                        "context": float(ctx_acc), "probe": float(best_p), "gap": float(gap)}
        
        results[domain_name] = domain_results
    
    return results


def main():
    logger.info("=" * 70)
    logger.info("OPUS FIXES: Qwen-7B + Scrambled CoT")
    logger.info("=" * 70)
    
    from transformer_lens import HookedTransformer
    
    all_results = {}
    
    # ============================================================
    # PART 1: Qwen2.5-7B — Future Lens + Domains
    # ============================================================
    logger.info("\nLoading Qwen2.5-7B...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-7B", device="cuda", dtype=torch.float16)
    model.eval()
    logger.info(f"  Loaded in {time.time()-t0:.1f}s")
    
    all_results["qwen7b_future_lens"] = run_future_lens(model, "Qwen/Qwen2.5-7B")
    all_results["qwen7b_domains"] = run_domains(model, "Qwen/Qwen2.5-7B", DOMAINS)
    
    del model; torch.cuda.empty_cache()
    
    # ============================================================
    # PART 2: GPT-J-6B — Scrambled CoT comparison
    # ============================================================
    logger.info("\nLoading GPT-J-6B for scrambled CoT...")
    model = HookedTransformer.from_pretrained("EleutherAI/gpt-j-6b", device="cuda", dtype=torch.float16)
    model.eval()
    
    # Run domains including scrambled CoT
    all_results["gptj_domains_with_scrambled"] = run_domains(model, "EleutherAI/gpt-j-6b", DOMAINS)
    
    del model; torch.cuda.empty_cache()
    
    # ============================================================
    # CROSS-MODEL SUMMARY
    # ============================================================
    logger.info(f"\n{'='*70}")
    logger.info("CROSS-MODEL FUTURE LENS SUMMARY")
    logger.info(f"{'='*70}")
    
    if "qwen7b_future_lens" in all_results:
        r = all_results["qwen7b_future_lens"]
        logger.info(f"\nQwen2.5-7B:")
        for k in K_VALUES_FL:
            key = f"k{k}"
            if key in r:
                d = r[key]
                logger.info(f"  K={k}: ctx={d['context']:.3f} probe={d['probe']:.3f} gap={d['gap']:+.3f}")
    
    logger.info(f"\n{'='*70}")
    logger.info("SCRAMBLED CoT COMPARISON (GPT-J-6B)")
    logger.info(f"{'='*70}")
    
    gptj = all_results.get("gptj_domains_with_scrambled", {})
    for dom in ["chain_of_thought", "chain_of_thought_scrambled"]:
        if dom in gptj:
            logger.info(f"\n  {dom}:")
            for k in K_VALUES_DOM:
                key = f"k{k}"
                if key in gptj[dom]:
                    d = gptj[dom][key]
                    logger.info(f"    K={k}: ctx={d['context']:.3f} probe={d['probe']:.3f} gap={d['gap']:+.3f}")
    
    # Save
    outfile = "results/lookahead/final/opus_fixes_7b_scrambled.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved to {outfile}")
    
    logger.info("\n" + "=" * 70)
    logger.info("DONE — OPUS FIXES")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
