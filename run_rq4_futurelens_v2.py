#!/usr/bin/env python3
"""
FUTURE LENS REPLICATION v2 — FIXED
====================================
Problem with v1: natural text has too many unique next-tokens (130 classes 
for 200 examples). Can't do cross-validation.

Fix: Two complementary approaches:
A) CODE DOMAIN — use our function signatures, predict next tokens during 
   code generation. Code tokens are more repetitive (return, =, 0, for, etc.)
B) TOKEN CATEGORIES — bin tokens into categories (keyword, punctuation, 
   identifier, number, string, whitespace). 6-10 classes, plenty of examples.

This connects directly to our main experiment AND to Pal et al.'s protocol.

Model: GPT-J-6B
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

MODEL_NAME = "EleutherAI/gpt-j-6b"
PCA_DIM = 128

# ================================================================
# DATASET: 500 function signatures (same as main experiment)
# ================================================================
SIGNATURES = [
    ('def add(a, b):', 'int'), ('def subtract(a, b):', 'int'), ('def multiply(a, b):', 'int'),
    ('def divide_int(a, b):', 'int'), ('def modulo(a, b):', 'int'), ('def power(base, exp):', 'int'),
    ('def count_words(text):', 'int'), ('def count_chars(text):', 'int'), ('def count_lines(text):', 'int'),
    ('def factorial(n):', 'int'), ('def fibonacci(n):', 'int'), ('def find_max(numbers):', 'int'),
    ('def find_min(numbers):', 'int'), ('def sum_list(numbers):', 'int'), ('def product(numbers):', 'int'),
    ('def string_length(s):', 'int'), ('def index_of(items, target):', 'int'),
    ('def count_vowels(text):', 'int'), ('def hamming_distance(s1, s2):', 'int'),
    ('def num_digits(n):', 'int'), ('def gcd(a, b):', 'int'), ('def lcm(a, b):', 'int'),
    ('def abs_value(n):', 'int'), ('def sign(n):', 'int'), ('def clamp(val, lo, hi):', 'int'),
    ('def greet(name):', 'str'), ('def farewell(name):', 'str'), ('def to_upper(text):', 'str'),
    ('def to_lower(text):', 'str'), ('def capitalize(text):', 'str'), ('def strip_whitespace(text):', 'str'),
    ('def reverse_string(s):', 'str'), ('def repeat_string(s, n):', 'str'),
    ('def join_words(words):', 'str'), ('def first_word(text):', 'str'), ('def last_word(text):', 'str'),
    ('def remove_spaces(s):', 'str'), ('def replace_char(s, old, new):', 'str'),
    ('def first_name(full_name):', 'str'), ('def last_name(full_name):', 'str'),
    ('def format_date(year, month, day):', 'str'), ('def format_time(hours, minutes):', 'str'),
    ('def to_binary(n):', 'str'), ('def to_hex(n):', 'str'), ('def to_roman(n):', 'str'),
    ('def slug(text):', 'str'), ('def title_case(text):', 'str'), ('def snake_case(text):', 'str'),
    ('def camel_case(text):', 'str'), ('def pad_left(s, width, char):', 'str'),
    ('def is_even(n):', 'bool'), ('def is_odd(n):', 'bool'), ('def is_positive(x):', 'bool'),
    ('def is_negative(x):', 'bool'), ('def is_zero(x):', 'bool'), ('def is_prime(n):', 'bool'),
    ('def is_palindrome(s):', 'bool'), ('def is_empty(s):', 'bool'), ('def is_sorted(items):', 'bool'),
    ('def contains(items, target):', 'bool'), ('def starts_with(text, prefix):', 'bool'),
    ('def ends_with(text, suffix):', 'bool'), ('def is_alpha(text):', 'bool'),
    ('def is_digit(text):', 'bool'), ('def is_upper(text):', 'bool'), ('def is_lower(text):', 'bool'),
    ('def has_duplicates(items):', 'bool'), ('def all_positive(numbers):', 'bool'),
    ('def any_negative(numbers):', 'bool'), ('def is_valid_email(text):', 'bool'),
    ('def is_substring(s, sub):', 'bool'), ('def is_anagram(s1, s2):', 'bool'),
    ('def is_power_of_two(n):', 'bool'), ('def is_leap_year(year):', 'bool'),
    ('def get_evens(numbers):', 'list'), ('def get_odds(numbers):', 'list'),
    ('def filter_positive(numbers):', 'list'), ('def filter_negative(numbers):', 'list'),
    ('def unique(items):', 'list'), ('def flatten(nested):', 'list'),
    ('def sort_ascending(items):', 'list'), ('def sort_descending(items):', 'list'),
    ('def reverse_list(items):', 'list'), ('def split_words(text):', 'list'),
    ('def split_lines(text):', 'list'), ('def split_chars(text):', 'list'),
    ('def zip_lists(a, b):', 'list'), ('def merge_sorted(a, b):', 'list'),
    ('def remove_duplicates(items):', 'list'), ('def take(items, n):', 'list'),
    ('def drop(items, n):', 'list'), ('def chunk(items, size):', 'list'),
    ('def interleave(a, b):', 'list'), ('def get_keys(d):', 'list'),
    ('def get_values(d):', 'list'), ('def range_list(start, stop):', 'list'),
    ('def neighbors(graph, node):', 'list'), ('def find_all(text, pattern):', 'list'),
    ('def average(numbers):', 'float'), ('def median(numbers):', 'float'),
    ('def variance(numbers):', 'float'), ('def std_dev(numbers):', 'float'),
    ('def to_celsius(f):', 'float'), ('def to_fahrenheit(c):', 'float'),
    ('def percentage(part, total):', 'float'), ('def ratio(a, b):', 'float'),
    ('def distance(x1, y1, x2, y2):', 'float'), ('def magnitude(x, y, z):', 'float'),
    ('def dot_product(a, b):', 'float'), ('def cosine_similarity(a, b):', 'float'),
    ('def circle_area(radius):', 'float'), ('def sphere_volume(radius):', 'float'),
    ('def triangle_area(base, height):', 'float'), ('def hypotenuse(a, b):', 'float'),
    ('def sigmoid(x):', 'float'), ('def relu(x):', 'float'), ('def tanh(x):', 'float'),
    ('def log_base(x, base):', 'float'), ('def square_root(x):', 'float'),
    ('def cube_root(x):', 'float'), ('def lerp(a, b, t):', 'float'),
    ('def normalize(value, min_val, max_val):', 'float'), ('def bmi(weight, height):', 'float'),
    ('def compound_interest(principal, rate, years):', 'float'),
]

# Token categories for natural text experiments
PYTHON_KEYWORDS = {'def', 'return', 'if', 'else', 'elif', 'for', 'while', 'in', 'not',
                   'and', 'or', 'True', 'False', 'None', 'import', 'from', 'class',
                   'try', 'except', 'with', 'as', 'is', 'lambda', 'pass', 'break',
                   'continue', 'yield', 'raise', 'finally', 'del', 'assert', 'global'}

def categorize_token(token_str):
    """Categorize a token string into a semantic category."""
    s = token_str.strip()
    if not s or s.isspace():
        return "whitespace"
    if s in PYTHON_KEYWORDS:
        return "keyword"
    if s in {'(', ')', '[', ']', '{', '}', ':', ',', '.', ';', '=', '==', '!=',
             '+=', '-=', '*=', '/=', '<', '>', '<=', '>=', '+', '-', '*', '/', 
             '//', '%', '**', '&', '|', '^', '~', '<<', '>>', '@'}:
        return "operator"
    if s.replace('.','').replace('-','').isdigit():
        return "number"
    if s.startswith('"') or s.startswith("'") or s.startswith('"""') or s.startswith("'''"):
        return "string"
    if s.startswith('#'):
        return "comment"
    # Identifiers (variable names, function names)
    if s.replace('_','').isalnum():
        return "identifier"
    return "other"


def run_code_future_lens(model):
    """
    APPROACH A: Code domain Future Lens
    Generate code from signatures, predict token at position N+K.
    Use token CATEGORIES as classes (6-8 classes instead of 100+).
    """
    logger.info("=" * 70)
    logger.info("APPROACH A: CODE DOMAIN FUTURE LENS")
    logger.info("=" * 70)
    
    n_layers = model.cfg.n_layers
    layers = sorted(set([0, n_layers//6, n_layers//3, n_layers//2,
                         2*n_layers//3, 5*n_layers//6, n_layers-1]))
    
    N_GEN = 15  # generate 15 tokens
    N_SIGS = len(SIGNATURES)
    
    # Step 1: Generate code
    logger.info(f"  Generating {N_GEN} tokens for {N_SIGS} signatures...")
    all_gen_tokens = []  # [n_sigs, N_GEN] token ids
    all_gen_strs = []    # [n_sigs, N_GEN] token strings
    
    for i, (sig, ret) in enumerate(SIGNATURES):
        prompt = sig + "\n    "
        tokens = model.to_tokens(prompt, prepend_bos=True)
        prompt_len = tokens.shape[1]
        gen_ids = []
        gen_strs = []
        
        with torch.no_grad():
            for step in range(N_GEN):
                logits = model(tokens)
                next_tok = logits[0, -1, :].argmax().item()
                gen_ids.append(next_tok)
                gen_strs.append(model.to_string(torch.tensor([next_tok])))
                tokens = torch.cat([tokens, torch.tensor([[next_tok]], device="cuda")], dim=1)
        
        all_gen_tokens.append(gen_ids)
        all_gen_strs.append(gen_strs)
        
        if i < 3:
            logger.info(f"    [{i}] {sig} → {''.join(gen_strs[:10])}")
        if (i + 1) % 50 == 0:
            logger.info(f"    Generated {i+1}/{N_SIGS}")
    
    # Step 2: For each K, predict token category at position N+K from position N
    results = {}
    
    for k in [1, 2, 3, 5]:
        logger.info(f"\n  === PREDICTING TOKEN AT gen_step+{k} ===")
        
        # Collect data: at each generation step N, predict category of step N+K
        all_activations = {l: [] for l in layers}
        all_labels = []
        all_context_bow = []
        all_context_window = []
        all_return_types = []
        
        for i, (sig, ret) in enumerate(SIGNATURES):
            prompt = sig + "\n    "
            tokens = model.to_tokens(prompt, prepend_bos=True)
            
            # Use multiple prediction points per signature
            for start_step in range(0, N_GEN - k, 2):  # every other step
                target_step = start_step + k
                if target_step >= N_GEN:
                    break
                
                target_tok_str = all_gen_strs[i][target_step]
                target_cat = categorize_token(target_tok_str)
                
                # Build context: prompt + generated tokens up to start_step
                ctx_tokens = tokens[0].cpu().tolist() + all_gen_tokens[i][:start_step + 1]
                full_input = torch.tensor([ctx_tokens], device="cuda")
                
                with torch.no_grad():
                    _, cache = model.run_with_cache(
                        full_input,
                        names_filter=[f"blocks.{l}.hook_resid_post" for l in layers]
                    )
                
                for layer in layers:
                    act = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].cpu().numpy()
                    all_activations[layer].append(act)
                
                all_labels.append(target_cat)
                all_return_types.append(ret)
                
                # Context BoW: count of each generated token category so far
                ctx_cats = [categorize_token(all_gen_strs[i][s]) for s in range(start_step + 1)]
                all_context_bow.append(ctx_cats)
                
                # Context window: last 3 token categories
                window_cats = ctx_cats[-3:] if len(ctx_cats) >= 3 else ctx_cats
                all_context_window.append(window_cats)
                
                del cache
                torch.cuda.empty_cache()
            
            if (i + 1) % 50 == 0:
                logger.info(f"    Processed {i+1}/{N_SIGS}")
        
        # Convert labels
        label_counts = Counter(all_labels)
        logger.info(f"  Label distribution: {dict(label_counts)}")
        
        # Filter to classes with enough examples
        valid_classes = {c for c, n in label_counts.items() if n >= 5}
        mask = [l in valid_classes for l in all_labels]
        
        labels_filtered = [l for l, m in zip(all_labels, mask) if m]
        ret_types_filtered = [r for r, m in zip(all_return_types, mask) if m]
        
        unique_labels = sorted(set(labels_filtered))
        l2i = {l: i for i, l in enumerate(unique_labels)}
        y = np.array([l2i[l] for l in labels_filtered])
        n_classes = len(unique_labels)
        n_examples = len(y)
        
        logger.info(f"  After filtering: {n_examples} examples, {n_classes} classes: {unique_labels}")
        
        if n_examples < 30 or n_classes < 2:
            logger.warning(f"  Skipping K={k}")
            continue
        
        min_class = min(Counter(y).values())
        n_splits = min(5, min_class)
        if n_splits < 2:
            logger.warning(f"  Min class {min_class} too small, skipping K={k}")
            continue
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        k_results = {"n_examples": n_examples, "n_classes": n_classes, 
                      "chance": float(1/n_classes), "classes": unique_labels}
        
        # --- BASELINE 1: Return type only ---
        ret_types_y = np.array([{'int':0,'str':1,'bool':2,'list':3,'float':4}[r] 
                                for r in ret_types_filtered])
        # One-hot
        X_ret = np.zeros((n_examples, 5))
        for j, r in enumerate(ret_types_y):
            X_ret[j, r] = 1.0
        
        ret_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
            X_ret, y, cv=cv, scoring="accuracy")
        ret_acc = ret_scores.mean()
        logger.info(f"  Return-type baseline: {ret_acc:.4f}")
        k_results["return_type_baseline"] = float(ret_acc)
        
        # --- BASELINE 2: Context BoW (category counts) ---
        cat_names = ["whitespace", "keyword", "operator", "number", "string", 
                      "comment", "identifier", "other"]
        c2i = {c: i for i, c in enumerate(cat_names)}
        
        X_bow = np.zeros((n_examples, len(cat_names)))
        bow_filtered = [b for b, m in zip(all_context_bow, mask) if m]
        for j, cats in enumerate(bow_filtered):
            for c in cats:
                if c in c2i:
                    X_bow[j, c2i[c]] += 1
        
        bow_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
            X_bow, y, cv=cv, scoring="accuracy")
        bow_acc = bow_scores.mean()
        logger.info(f"  Context BoW baseline: {bow_acc:.4f}")
        k_results["context_bow"] = float(bow_acc)
        
        # --- BASELINE 3: Return type + Context BoW ---
        X_combined = np.concatenate([X_ret, X_bow], axis=1)
        comb_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
            X_combined, y, cv=cv, scoring="accuracy")
        comb_acc = comb_scores.mean()
        logger.info(f"  Ret+BoW combined:    {comb_acc:.4f}")
        k_results["ret_plus_bow"] = float(comb_acc)
        
        # --- PROBE: Each layer ---
        logger.info(f"\n  Probe by layer:")
        probe_results = {}
        for layer in layers:
            X_probe = np.stack([all_activations[layer][j] 
                               for j, m in enumerate(mask) if m])
            
            scaler = StandardScaler()
            X_p = scaler.fit_transform(X_probe)
            if X_p.shape[1] > PCA_DIM:
                X_p = PCA(n_components=min(PCA_DIM, X_p.shape[0]-1),
                          random_state=42).fit_transform(X_p)
            
            p_scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
                X_p, y, cv=cv, scoring="accuracy")
            p_acc = p_scores.mean()
            
            gap_bow = p_acc - bow_acc
            gap_comb = p_acc - comb_acc
            
            logger.info(f"    L{layer:>3}: probe={p_acc:.4f} | "
                        f"gap_bow={gap_bow:+.4f} gap_comb={gap_comb:+.4f}")
            
            probe_results[str(layer)] = {
                "probe": float(p_acc),
                "gap_vs_bow": float(gap_bow),
                "gap_vs_combined": float(gap_comb),
            }
        
        k_results["probe_by_layer"] = probe_results
        
        # Summary
        best_layer = max(probe_results.items(), key=lambda x: x[1]["probe"])
        best_probe = best_layer[1]["probe"]
        
        logger.info(f"\n  === STAIRCASE SUMMARY (K={k}) ===")
        logger.info(f"  Chance:             {1/n_classes:.4f}")
        logger.info(f"  Return-type only:   {ret_acc:.4f}")
        logger.info(f"  Context BoW:        {bow_acc:.4f}")
        logger.info(f"  Ret+BoW combined:   {comb_acc:.4f}")
        logger.info(f"  Best Probe (L{best_layer[0]}):  {best_probe:.4f}")
        logger.info(f"  Gap (probe - comb): {best_probe - comb_acc:+.4f}")
        
        if best_probe - comb_acc < 0.03:
            logger.info(f"  >>> BASELINES EXPLAIN PROBE SIGNAL <<<")
        else:
            logger.info(f"  >>> PROBE EXCEEDS BASELINES BY {best_probe - comb_acc:+.4f} <<<")
        
        results[f"k{k}"] = k_results
    
    return results


def run_natural_text_future_lens(model):
    """
    APPROACH B: Natural text with token CATEGORIES
    Generate continuations from diverse prompts, predict next-token CATEGORY.
    """
    logger.info("\n" + "=" * 70)
    logger.info("APPROACH B: NATURAL TEXT WITH TOKEN CATEGORIES")
    logger.info("=" * 70)
    
    n_layers = model.cfg.n_layers
    layers = sorted(set([0, n_layers//6, n_layers//3, n_layers//2,
                         2*n_layers//3, 5*n_layers//6, n_layers-1]))
    
    PROMPTS = [
        "The theory of relativity was proposed by Albert Einstein in",
        "Machine learning is a subset of artificial intelligence that",
        "The stock market experienced significant volatility during",
        "Climate change continues to affect global temperatures and",
        "The human brain processes information through networks of",
        "Quantum computing promises to revolutionize fields such as",
        "The Renaissance period saw remarkable advances in art and",
        "Photosynthesis converts sunlight into chemical energy that",
        "The discovery of DNA structure by Watson and Crick led to",
        "Artificial neural networks are designed to mimic the way",
        "The Internet has transformed how people communicate and",
        "Evolution by natural selection explains the diversity of",
        "Black holes are formed when massive stars collapse under",
        "The Industrial Revolution fundamentally changed society by",
        "Vaccines have been instrumental in preventing the spread of",
        "The solar system contains eight planets orbiting around",
        "Democracy as a form of government originated in ancient",
        "The Great Wall of China was constructed over centuries to",
        "Nuclear energy is generated through the process of fission",
        "The Amazon rainforest plays a crucial role in regulating",
        "Blockchain technology provides a decentralized ledger that",
        "The speed of light is approximately three hundred thousand",
        "Gravity is one of the four fundamental forces governing",
        "The printing press invented by Gutenberg revolutionized",
        "Plate tectonics theory explains how continents move and",
        "The French Revolution of 1789 fundamentally altered the",
        "Superconductivity occurs when certain materials are cooled",
        "The United Nations was established after World War Two to",
        "CRISPR technology allows precise editing of genetic material",
        "The periodic table arranges elements by their atomic number",
    ]
    
    CONTEXT_LEN = 30
    N_GEN = 20
    
    logger.info(f"  {len(PROMPTS)} prompts, {N_GEN} gen tokens each")
    
    # Generate
    all_data = []
    for pi, prompt in enumerate(PROMPTS):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        prompt_len = tokens.shape[1]
        
        # Generate multiple continuations with different temperatures
        for temp in [0.0, 0.7, 1.0]:
            with torch.no_grad():
                if temp == 0.0:
                    gen = model.generate(tokens, max_new_tokens=N_GEN, temperature=0.0)
                else:
                    gen = model.generate(tokens, max_new_tokens=N_GEN, temperature=temp, top_k=50)
            
            full_ids = gen[0].cpu().tolist()
            gen_ids = full_ids[prompt_len:]
            gen_strs = [model.to_string(torch.tensor([t])) for t in gen_ids]
            
            all_data.append({
                "prompt": prompt,
                "prompt_len": prompt_len,
                "full_ids": full_ids,
                "gen_ids": gen_ids,
                "gen_strs": gen_strs,
            })
        
        if (pi + 1) % 10 == 0:
            logger.info(f"    Generated {pi+1}/{len(PROMPTS)}")
    
    logger.info(f"  Total continuations: {len(all_data)}")
    
    # For each K, predict token category
    results = {}
    
    for k in [1, 2, 3]:
        logger.info(f"\n  === NATURAL TEXT: PREDICTING CATEGORY AT N+{k} ===")
        
        all_activations = {l: [] for l in layers}
        all_labels = []
        all_bow_features = []
        
        for di, d in enumerate(all_data):
            for start_step in range(0, len(d["gen_ids"]) - k, 2):
                target_step = start_step + k
                if target_step >= len(d["gen_ids"]):
                    break
                
                target_str = d["gen_strs"][target_step]
                target_cat = categorize_token(target_str)
                
                # Build input
                ctx_ids = d["full_ids"][:d["prompt_len"] + start_step + 1]
                full_input = torch.tensor([ctx_ids], device="cuda")
                
                with torch.no_grad():
                    _, cache = model.run_with_cache(
                        full_input,
                        names_filter=[f"blocks.{l}.hook_resid_post" for l in layers]
                    )
                
                for layer in layers:
                    act = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].cpu().numpy()
                    all_activations[layer].append(act)
                
                all_labels.append(target_cat)
                
                # BoW of context categories
                ctx_gen_strs = d["gen_strs"][:start_step + 1]
                ctx_cats = [categorize_token(s) for s in ctx_gen_strs]
                all_bow_features.append(ctx_cats)
                
                del cache
                torch.cuda.empty_cache()
            
            if (di + 1) % 20 == 0:
                logger.info(f"    Processed {di+1}/{len(all_data)}")
        
        # Process labels
        label_counts = Counter(all_labels)
        logger.info(f"  Labels: {dict(label_counts)}")
        
        valid = {c for c, n in label_counts.items() if n >= 5}
        mask = [l in valid for l in all_labels]
        labels_f = [l for l, m in zip(all_labels, mask) if m]
        unique = sorted(set(labels_f))
        l2i = {l: i for i, l in enumerate(unique)}
        y = np.array([l2i[l] for l in labels_f])
        n_cls = len(unique)
        n_ex = len(y)
        
        logger.info(f"  Filtered: {n_ex} examples, {n_cls} classes")
        
        if n_ex < 30 or n_cls < 2:
            continue
        
        min_c = min(Counter(y).values())
        n_splits = min(5, min_c)
        if n_splits < 2:
            continue
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # BoW baseline
        cat_names = ["whitespace", "keyword", "operator", "number", "string",
                      "comment", "identifier", "other"]
        c2i_bow = {c: i for i, c in enumerate(cat_names)}
        
        X_bow = np.zeros((n_ex, len(cat_names)))
        bow_f = [b for b, m in zip(all_bow_features, mask) if m]
        for j, cats in enumerate(bow_f):
            for c in cats:
                if c in c2i_bow:
                    X_bow[j, c2i_bow[c]] += 1
        
        bow_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
            X_bow, y, cv=cv, scoring="accuracy")
        bow_acc = bow_scores.mean()
        logger.info(f"  BoW baseline: {bow_acc:.4f}")
        
        # Probe each layer
        k_results = {"n_examples": n_ex, "n_classes": n_cls, "chance": 1/n_cls,
                      "bow": float(bow_acc), "classes": unique}
        probe_results = {}
        
        for layer in layers:
            X = np.stack([all_activations[layer][j] for j, m in enumerate(mask) if m])
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            if X_s.shape[1] > PCA_DIM:
                X_s = PCA(n_components=min(PCA_DIM, X_s.shape[0]-1),
                          random_state=42).fit_transform(X_s)
            
            scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
                X_s, y, cv=cv, scoring="accuracy")
            acc = scores.mean()
            
            logger.info(f"    L{layer:>3}: {acc:.4f} (gap_bow={acc-bow_acc:+.4f})")
            probe_results[str(layer)] = {"probe": float(acc), "gap_bow": float(acc-bow_acc)}
        
        k_results["probe_by_layer"] = probe_results
        results[f"k{k}"] = k_results
    
    return results


def main():
    logger.info("=" * 70)
    logger.info("FUTURE LENS REPLICATION v2 — GPT-J-6B")
    logger.info("=" * 70)
    
    from transformer_lens import HookedTransformer
    
    logger.info("Loading GPT-J-6B...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda", dtype=torch.float16)
    model.eval()
    logger.info(f"  Loaded in {time.time()-t0:.1f}s, {model.cfg.n_layers} layers")
    
    all_results = {"model": MODEL_NAME}
    
    # Approach A: Code domain
    all_results["code_domain"] = run_code_future_lens(model)
    
    # Approach B: Natural text categories
    all_results["natural_text"] = run_natural_text_future_lens(model)
    
    # Save
    outfile = "results/lookahead/final/future_lens_staircase_v2.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved to {outfile}")
    
    logger.info("\n" + "=" * 70)
    logger.info("DONE — FUTURE LENS v2")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
