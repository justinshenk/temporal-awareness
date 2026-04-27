#!/usr/bin/env python3
"""
TRAINING DYNAMICS — FIXED (uses HF revision on deduped models)
===============================================================
TransformerLens checkpoint_index is broken. Instead:
  1. Load HF model with revision='stepN' 
  2. Pass to TransformerLens via hf_model parameter

PHASE 1: Code staircase across training (3 Pythia sizes × 17 checkpoints)
PHASE 2: Behavioral accuracy at 5 key checkpoints (3 models)  
PHASE 3: Future Lens K decay at 5 key checkpoints (Pythia-2.8B)

Resume support. Saves after every checkpoint.
Expected: ~14 hours on L40S.
"""

import json, os, sys, time, re, gc
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

PCA_DIM = 128
OUTDIR = "results/lookahead/final"
os.makedirs(OUTDIR, exist_ok=True)

# ================================================================
# CHECKPOINTS (log-spaced for smooth curve)
# ================================================================
ALL_STEPS = [
    "step0", "step1", "step8", "step64", "step128", "step256",
    "step512", "step1000", "step2000", "step4000", "step8000",
    "step16000", "step32000", "step64000", "step96000",
    "step128000", "step143000",
]

KEY_STEPS = ["step0", "step512", "step4000", "step32000", "step143000"]

# 3 Pythia-deduped models
MODELS = [
    ("EleutherAI/pythia-2.8b-deduped", "pythia-2.8b-deduped", "Pythia-2.8B"),
    ("EleutherAI/pythia-1b-deduped",   "pythia-1b-deduped",   "Pythia-1B"),
    ("EleutherAI/pythia-410m-deduped",  "pythia-410m-deduped",  "Pythia-410M"),
]

# ================================================================
# SIGNATURES
# ================================================================
SIGS = [
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

FL_PROMPTS = [
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
    "Artificial intelligence systems learn from data by identifying patterns",
    "The internet was originally developed as a military communication network",
    "Machine learning algorithms can be broadly classified into supervised and",
    "Cloud computing allows organizations to access computing resources over",
    "Neural networks are computational models inspired by the structure of",
    "Programming languages provide abstractions that allow developers to write",
    "Climate change is primarily driven by the emission of greenhouse",
    "Renewable energy sources include solar wind and hydroelectric power",
    "The largest ocean on Earth is the Pacific Ocean which",
    "Democracy as a form of government gives citizens the power",
    "Education plays a crucial role in the development of modern",
    "The stock market serves as a platform where investors can",
    "The human brain is the most complex organ in the",
    "In Python programming the def keyword is used to define",
    "A function that takes a list of numbers and returns",
    "The algorithm works by first sorting the input array and",
    "To implement a binary search tree you need to define",
    "The main difference between a list and a tuple in",
]


# ================================================================
# UTILITIES
# ================================================================
def parse_signature(sig):
    match = re.match(r'def\s+(\w+)\s*\((.*?)\)\s*:', sig)
    if not match: return [], []
    name_tokens = match.group(1).split('_')
    param_tokens = []
    if match.group(2).strip():
        for p in match.group(2).split(','):
            p = p.strip()
            if ':' in p: p = p.split(':')[0].strip()
            if '=' in p: p = p.split('=')[0].strip()
            if p: param_tokens.extend(p.split('_'))
    return name_tokens, param_tokens


def build_np_features(sigs):
    all_toks = set()
    parsed = []
    for sig, _ in sigs:
        nt, pt = parse_signature(sig)
        parsed.append((nt, pt))
        all_toks.update(nt); all_toks.update(pt)
    vocab = sorted(all_toks)
    t2i = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(sigs), len(vocab) * 2))
    for i, (nt, pt) in enumerate(parsed):
        for t in nt:
            if t in t2i: X[i, t2i[t]] = 1.0
        for t in pt:
            if t in t2i: X[i, len(vocab) + t2i[t]] = 1.0
    return X


def get_layers(n_layers):
    return sorted(set([0, n_layers//6, n_layers//3, n_layers//2,
                       2*n_layers//3, 5*n_layers//6, n_layers-1]))


def save(data, name):
    path = f"{OUTDIR}/dynamics_{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load(name):
    path = f"{OUTDIR}/dynamics_{name}.json"
    if os.path.exists(path):
        return json.load(open(path))
    return None


def load_model(hf_id, tl_id, step):
    """Load HF model at revision, convert to TransformerLens."""
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer
    
    hf = AutoModelForCausalLM.from_pretrained(
        hf_id, revision=step, torch_dtype=torch.float16,
        use_safetensors=True)
    
    tl = HookedTransformer.from_pretrained(
        tl_id, hf_model=hf, device="cuda", dtype=torch.float16)
    
    del hf; torch.cuda.empty_cache()
    tl.eval()
    return tl


# ================================================================
# PHASE 1: CODE STAIRCASE ACROSS TRAINING
# ================================================================
def phase1(hf_id, tl_id, label):
    logger.info(f"\n{'='*70}")
    logger.info(f"PHASE 1: CODE STAIRCASE — {label}")
    logger.info(f"{'='*70}")
    
    targets = sorted(set(r for _, r in SIGS))
    t2i = {t: i for i, t in enumerate(targets)}
    labels = np.array([t2i[r] for _, r in SIGS])
    
    X_np = build_np_features(SIGS)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    np_acc = cross_val_score(
        LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
        X_np, labels, cv=cv, scoring="accuracy").mean()
    
    logger.info(f"  N+P baseline: {np_acc:.4f}")
    
    safe_name = tl_id.replace('-', '_').replace('.', '_')
    save_key = f"p1_{safe_name}"
    existing = load(save_key)
    if existing and "checkpoints" in existing:
        results = existing
        done = {cp["step"] for cp in results["checkpoints"]}
        logger.info(f"  Resuming: {len(done)}/{len(ALL_STEPS)} done")
    else:
        results = {"model": hf_id, "label": label, "np_acc": float(np_acc), "checkpoints": []}
        done = set()
    
    # Print cached
    for cp in sorted(results["checkpoints"], key=lambda x: ALL_STEPS.index(x["step"]) if x["step"] in ALL_STEPS else 99):
        logger.info(f"  {cp['step']:>12} probe={cp['best_probe']:.4f} gap={cp['gap']:+.4f} L{cp['best_layer']} (cached)")
    
    for step in ALL_STEPS:
        if step in done:
            continue
        
        t0 = time.time()
        try:
            model = load_model(hf_id, tl_id, step)
        except Exception as e:
            logger.warning(f"  {step}: FAILED load — {e}")
            continue
        
        layers = get_layers(model.cfg.n_layers)
        
        all_acts = {l: [] for l in layers}
        for si, (sig, _) in enumerate(SIGS):
            tokens = model.to_tokens(sig + "\n    ", prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens,
                    names_filter=[f"blocks.{l}.hook_resid_post" for l in layers])
            for l in layers:
                all_acts[l].append(cache[f"blocks.{l}.hook_resid_post"][0, -1, :].cpu().numpy())
            del cache; torch.cuda.empty_cache()
        
        best_probe, best_layer = 0, 0
        layer_accs = {}
        for l in layers:
            X = np.stack(all_acts[l])
            X_s = StandardScaler().fit_transform(X)
            if X_s.shape[1] > PCA_DIM:
                X_s = PCA(n_components=min(PCA_DIM, X.shape[0]-1),
                          random_state=42).fit_transform(X_s)
            acc = cross_val_score(
                LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                X_s, labels, cv=cv, scoring="accuracy").mean()
            layer_accs[str(l)] = float(acc)
            if acc > best_probe:
                best_probe, best_layer = acc, l
        
        gap = best_probe - np_acc
        elapsed = time.time() - t0
        
        logger.info(f"  {step:>12} probe={best_probe:.4f} gap={gap:+.4f} "
                    f"L{best_layer} ({elapsed:.0f}s)")
        
        results["checkpoints"].append({
            "step": step, "best_probe": float(best_probe),
            "best_layer": int(best_layer), "gap": float(gap),
            "per_layer": layer_accs,
        })
        
        save(results, save_key)
        del model; torch.cuda.empty_cache(); gc.collect()
    
    # Sort by training step
    step_order = {s: i for i, s in enumerate(ALL_STEPS)}
    results["checkpoints"].sort(key=lambda x: step_order.get(x["step"], 99))
    save(results, save_key)
    return results


# ================================================================
# PHASE 2: BEHAVIORAL AT KEY CHECKPOINTS
# ================================================================
def phase2(hf_id, tl_id, label):
    logger.info(f"\n{'='*70}")
    logger.info(f"PHASE 2: BEHAVIORAL — {label}")
    logger.info(f"{'='*70}")
    
    targets = sorted(set(r for _, r in SIGS))
    
    safe_name = tl_id.replace('-', '_').replace('.', '_')
    save_key = f"p2_{safe_name}"
    existing = load(save_key)
    if existing and "checkpoints" in existing:
        results = existing
        done = {cp["step"] for cp in results["checkpoints"]}
        logger.info(f"  Resuming: {len(done)}/{len(KEY_STEPS)} done")
    else:
        results = {"model": hf_id, "label": label, "checkpoints": []}
        done = set()
    
    for step in KEY_STEPS:
        if step in done:
            logger.info(f"  {step}: cached")
            continue
        
        t0 = time.time()
        try:
            model = load_model(hf_id, tl_id, step)
        except Exception as e:
            logger.warning(f"  {step}: FAILED — {e}")
            continue
        
        correct, total = 0, 0
        type_correct = {t: 0 for t in targets}
        type_total = {t: 0 for t in targets}
        
        for si, (sig, expected) in enumerate(SIGS):
            prompt = sig + "\n    "
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                gen = model.generate(tokens, max_new_tokens=40, temperature=0.0)
            gen_text = model.to_string(gen[0, tokens.shape[1]:]).lower().strip()
            
            predicted = None
            if any(kw in gen_text for kw in ['return true', 'return false', 'return not ']):
                predicted = 'bool'
            elif any(kw in gen_text for kw in ['return []', 'return [', 'return list(', '.append(']):
                predicted = 'list'
            elif any(kw in gen_text for kw in ['return ""', "return ''", 'return f"', "return f'", 'return str(', '.join(']):
                predicted = 'str'
            elif any(kw in gen_text for kw in ['return 0.', 'return float(', 'math.']):
                predicted = 'float'
            elif any(kw in gen_text for kw in ['return 0', 'return len(', 'return int(', 'return n ', 'return count', 'return sum(', 'return max(', 'return min(']):
                predicted = 'int'
            
            if predicted == expected:
                correct += 1
                type_correct[expected] += 1
            type_total[expected] += 1
            total += 1
        
        behavioral = correct / total if total > 0 else 0
        elapsed = time.time() - t0
        logger.info(f"  {step}: behavioral={behavioral:.4f} ({correct}/{total}) ({elapsed:.0f}s)")
        
        results["checkpoints"].append({
            "step": step, "behavioral": float(behavioral),
            "correct": correct, "total": total,
        })
        
        save(results, save_key)
        del model; torch.cuda.empty_cache(); gc.collect()
    
    step_order = {s: i for i, s in enumerate(KEY_STEPS)}
    results["checkpoints"].sort(key=lambda x: step_order.get(x["step"], 99))
    save(results, save_key)
    return results


# ================================================================
# PHASE 3: FUTURE LENS K DECAY DURING TRAINING (2.8B only)
# ================================================================
def phase3():
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 3: FUTURE LENS K DECAY DURING TRAINING (Pythia-2.8B)")
    logger.info(f"{'='*70}")
    
    hf_id = "EleutherAI/pythia-2.8b-deduped"
    tl_id = "pythia-2.8b-deduped"
    K_VALUES = [1, 3, 5]
    N_GEN = 80
    MIN_TARGET = 10
    
    save_key = "p3_future_lens"
    existing = load(save_key)
    if existing and "checkpoints" in existing:
        results = existing
        done = {cp["step"] for cp in results["checkpoints"]}
        logger.info(f"  Resuming: {len(done)}/{len(KEY_STEPS)} done")
    else:
        results = {"model": hf_id, "checkpoints": []}
        done = set()
    
    for step in KEY_STEPS:
        if step in done:
            logger.info(f"  {step}: cached")
            continue
        
        t0 = time.time()
        try:
            model = load_model(hf_id, tl_id, step)
        except Exception as e:
            logger.warning(f"  {step}: FAILED — {e}")
            continue
        
        layers = get_layers(model.cfg.n_layers)
        W_E = model.W_E.detach()
        
        logger.info(f"  {step}: generating {len(FL_PROMPTS)} prompts...")
        all_seqs = []
        for prompt in FL_PROMPTS:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                gen = model.generate(tokens, max_new_tokens=N_GEN, temperature=0.0)
            all_seqs.append({"prompt_len": tokens.shape[1],
                             "full_ids": gen[0].cpu().tolist()})
        
        train_seqs = all_seqs[:15]
        test_seqs = all_seqs[15:]
        
        ckpt_result = {"step": step}
        
        for k in K_VALUES:
            test_tgts = []
            for seq in test_seqs:
                ids = seq["full_ids"]; pl = seq["prompt_len"]
                for n in range(pl, len(ids) - k):
                    test_tgts.append(ids[n + k])
            tc = Counter(test_tgts)
            frequent = {t for t, c in tc.items() if c >= MIN_TARGET}
            t2i = {t: i for i, t in enumerate(sorted(frequent))}
            n_cls = len(t2i)
            
            if n_cls < 3:
                logger.info(f"    K={k}: {n_cls} classes, skip")
                ckpt_result[f"k{k}"] = {"skip": True}
                continue
            
            activations = {l: [] for l in layers}
            ctx_embs, fl_labels = [], []
            
            for seq in test_seqs:
                ids = seq["full_ids"]; pl = seq["prompt_len"]
                inp = torch.tensor([ids], device="cuda")
                with torch.no_grad():
                    _, cache = model.run_with_cache(inp,
                        names_filter=[f"blocks.{l}.hook_resid_post" for l in layers])
                for n in range(pl, len(ids) - k):
                    tgt = ids[n + k]
                    if tgt not in t2i: continue
                    fl_labels.append(t2i[tgt])
                    for l in layers:
                        activations[l].append(
                            cache[f"blocks.{l}.hook_resid_post"][0, n, :].cpu().numpy())
                    ws = max(0, n - 4)
                    ctx = W_E[torch.tensor(ids[ws:n+1], device="cuda")].cpu().numpy()
                    ctx_embs.append(ctx.mean(axis=0))
                del cache; torch.cuda.empty_cache()
            
            fl_labels = np.array(fl_labels)
            n_ex = len(fl_labels)
            if n_ex < 20:
                ckpt_result[f"k{k}"] = {"skip": True, "n_ex": n_ex}
                continue
            
            min_c = min(Counter(fl_labels).values())
            n_splits = min(5, min_c)
            if n_splits < 2:
                ckpt_result[f"k{k}"] = {"skip": True}
                continue
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            X_ctx = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(
                StandardScaler().fit_transform(np.stack(ctx_embs)))
            ctx_acc = cross_val_score(
                LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                X_ctx, fl_labels, cv=cv, scoring="accuracy").mean()
            
            best_p, best_l = 0, 0
            for l in layers:
                X = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(
                    StandardScaler().fit_transform(np.stack(activations[l])))
                acc = cross_val_score(
                    LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                    X, fl_labels, cv=cv, scoring="accuracy").mean()
                if acc > best_p: best_p, best_l = acc, l
            
            gap = best_p - ctx_acc
            logger.info(f"    K={k}: {n_ex} ex | ctx={ctx_acc:.3f} "
                        f"probe(L{best_l})={best_p:.3f} gap={gap:+.3f}")
            
            ckpt_result[f"k{k}"] = {
                "n_examples": n_ex, "n_classes": n_cls,
                "context": float(ctx_acc), "probe": float(best_p),
                "gap": float(gap), "best_layer": int(best_l),
            }
        
        elapsed = time.time() - t0
        logger.info(f"    ({elapsed:.0f}s)")
        results["checkpoints"].append(ckpt_result)
        save(results, save_key)
        del model; torch.cuda.empty_cache(); gc.collect()
    
    step_order = {s: i for i, s in enumerate(KEY_STEPS)}
    results["checkpoints"].sort(key=lambda x: step_order.get(x["step"], 99))
    save(results, save_key)
    return results


# ================================================================
# ANALYSIS
# ================================================================
def analyze(all_results):
    logger.info(f"\n{'='*70}")
    logger.info("COMPREHENSIVE ANALYSIS")
    logger.info(f"{'='*70}")
    
    # Phase 1 trajectories
    logger.info(f"\n  PHASE 1: PROBE GAP TRAJECTORIES")
    logger.info(f"  {'Step':<12}", end="")
    for _, tl_id, label in MODELS:
        logger.info(f" {label:>12}", end="")
    logger.info("")
    
    for step in ALL_STEPS:
        logger.info(f"  {step:<12}", end="")
        for _, tl_id, label in MODELS:
            safe = tl_id.replace('-','_').replace('.','_')
            r = all_results.get(f"p1_{safe}", {})
            gap = None
            for cp in r.get("checkpoints", []):
                if cp["step"] == step:
                    gap = cp["gap"]
            logger.info(f" {gap:>+12.4f}" if gap is not None else f" {'---':>12}", end="")
        logger.info("")
    
    # Trajectory shapes
    for _, tl_id, label in MODELS:
        safe = tl_id.replace('-','_').replace('.','_')
        r = all_results.get(f"p1_{safe}", {})
        cps = r.get("checkpoints", [])
        if not cps: continue
        
        gaps = [cp["gap"] for cp in cps]
        probes = [cp["best_probe"] for cp in cps]
        max_gap = max(gaps)
        min_gap = min(gaps)
        max_idx = gaps.index(max_gap)
        final_gap = gaps[-1]
        
        if max_idx < len(gaps) - 2 and max_gap > final_gap + 0.03:
            shape = "RISE-FALL"
        elif max_gap - min_gap < 0.03:
            shape = "FLAT"
        else:
            shape = "MONOTONIC"
        
        logger.info(f"\n  {label}: {shape}")
        logger.info(f"    Probe: {probes[0]:.4f} → {probes[-1]:.4f}")
        logger.info(f"    Gap:   {gaps[0]:+.4f} → {gaps[-1]:+.4f} (peak {max_gap:+.4f} at {cps[max_idx]['step']})")
    
    # Phase 2: Probe vs Behavioral
    logger.info(f"\n  PHASE 2: PROBE vs BEHAVIORAL")
    for _, tl_id, label in MODELS:
        safe = tl_id.replace('-','_').replace('.','_')
        p1 = all_results.get(f"p1_{safe}", {})
        p2 = all_results.get(f"p2_{safe}", {})
        if not p2.get("checkpoints"): continue
        
        logger.info(f"\n  {label}:")
        logger.info(f"  {'Step':<12} {'Probe':>8} {'Behav':>8} {'Lead?':>14}")
        for cp2 in p2["checkpoints"]:
            step = cp2["step"]
            behav = cp2["behavioral"]
            probe = None
            for cp1 in p1.get("checkpoints", []):
                if cp1["step"] == step:
                    probe = cp1["best_probe"]
            if probe is not None:
                if probe > behav + 0.1: lead = "PROBE LEADS"
                elif behav > probe + 0.1: lead = "BEHAV LEADS"
                else: lead = "CO-DEVELOP"
                logger.info(f"  {step:<12} {probe:>8.4f} {behav:>8.4f} {lead:>14}")
    
    # Phase 3: Future Lens dynamics
    p3 = all_results.get("p3_fl", {})
    if p3.get("checkpoints"):
        logger.info(f"\n  PHASE 3: FUTURE LENS DURING TRAINING")
        logger.info(f"  {'Step':<12} {'K=1':>8} {'K=3':>8} {'K=5':>8}")
        for cp in p3["checkpoints"]:
            k1 = cp.get("k1", {}).get("gap")
            k3 = cp.get("k3", {}).get("gap")
            k5 = cp.get("k5", {}).get("gap")
            logger.info(f"  {cp['step']:<12} "
                        f"{k1:>+8.3f}" if k1 is not None else f"{'skip':>8}" 
                        f" {k3:>+8.3f}" if k3 is not None else f" {'skip':>8}"
                        f" {k5:>+8.3f}" if k5 is not None else f" {'skip':>8}")


# ================================================================
# MAIN
# ================================================================
def main():
    logger.info("=" * 70)
    logger.info("TRAINING DYNAMICS — FIXED (HF revision + deduped)")
    logger.info("3 Pythia × 17 checkpoints + behavioral + Future Lens")
    logger.info("=" * 70)
    t_start = time.time()
    
    all_results = {}
    
    # Phase 1: Code staircase (priority order: 2.8B first)
    for hf_id, tl_id, label in MODELS:
        safe = tl_id.replace('-','_').replace('.','_')
        all_results[f"p1_{safe}"] = phase1(hf_id, tl_id, label)
    
    # Phase 2: Behavioral
    for hf_id, tl_id, label in MODELS:
        safe = tl_id.replace('-','_').replace('.','_')
        all_results[f"p2_{safe}"] = phase2(hf_id, tl_id, label)
    
    # Phase 3: Future Lens
    all_results["p3_fl"] = phase3()
    
    # Analysis
    analyze(all_results)
    
    # Save all
    save(all_results, "all_complete")
    
    elapsed = (time.time() - t_start) / 3600
    logger.info(f"\n{'='*70}")
    logger.info(f"DONE — {elapsed:.1f} hours")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
