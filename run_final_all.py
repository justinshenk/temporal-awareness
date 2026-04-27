#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE SCRIPT — ALL REMAINING EXPERIMENTS
========================================================
Everything not yet completed, in one run. Saves after each section.

SECTION 1: Qwen-7B code staircase FIXED (proper name+params baseline)
SECTION 2: Qwen-7B behavioral code generation accuracy  
SECTION 3: Nonlinear probes (MLP) on code — GPT-J + Qwen-7B
           Both probe AND baseline get MLP treatment (fair comparison)
SECTION 4: Nonlinear probes on domain study (GPT-J, K=3)
SECTION 5: Pairwise domain significance tests (no GPU)
SECTION 6: Cross-model domain correlation (no GPU)

Expected: ~5-6 hours on L40S/RTX 3090.
"""

import json, os, sys, time, re, gc
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from scipy.stats import spearmanr

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

PCA_DIM = 128
OUTDIR = "results/lookahead/final"
os.makedirs(OUTDIR, exist_ok=True)

def save(data, name):
    path = f"{OUTDIR}/final_all_{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"  [SAVED] {path}")


# ================================================================
# SIGNATURE DATA
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

# Domain prompts for nonlinear comparison (20 each)
DOMAIN_PROMPTS = {
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
        "Question: What is 18 times 23? Step 1: Break it down.",
        "Problem: A car uses 8 gallons for 240 miles. What is the mpg? Solution:",
        "Calculate: What is 3/4 plus 2/3? First, find a common denominator:",
        "Question: If you save $50 per week, how much in 6 months?",
        "Problem: Find the hypotenuse of a right triangle with legs 3 and 4.",
        "Question: What is 15 squared? Step 1: multiply 15 by 15.",
        "Problem: Pizza 8 slices. 3 people eat 2 each. How many remain?",
        "Calculate: 20% tip on a $85 bill. Step 1: find 10% first:",
        "Question: How many minutes in 3 days? Let me break this down:",
        "Problem: Shirt costs $40 and is 25% off. Sale price?",
        "Calculate: perimeter of rectangle length 15 width 8.",
        "Question: What is 144 divided by 12? Let me solve:",
        "Problem: Train 2:15 PM to 5:45 PM. Trip duration?",
        "Calculate: compound interest on $1000 at 5% for 2 years.",
        "Question: Volume of a cube with side length 5? Step 1:",
        "Problem: 4 workers finish in 6 hours, how long for 3 workers?",
        "Calculate: area of circle with radius 7. Using pi = 3.14:",
        "Question: What is 2 to the power of 8? Step by step.",
        "Problem: Notebooks $3 each. How much for 15?",
        "Calculate: diagonal of rectangle sides 6 and 8.",
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
        "def max_element(arr):\n    best = arr[0]\n    for x in arr:",
        "def remove_duplicates(lst):\n    seen = set()\n    result = []",
        "def power(base, exp):\n    if exp == 0:\n        return 1",
        "def gcd(a, b):\n    while b != 0:\n        a, b = b,",
        "def matrix_multiply(A, B):\n    rows = len(A)\n    cols = len(",
        "def insertion_sort(arr):\n    for i in range(1, len(arr)):",
        "def two_sum(nums, target):\n    seen = {}\n    for i,",
        "def valid_parentheses(s):\n    stack = []\n    mapping = {')': '(',",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr",
        "def palindrome_check(s):\n    s = s.lower().replace(' ', '')\n    return",
        "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(",
        "def rotate_array(arr, k):\n    k = k % len(arr)\n    return",
        "def count_chars(text):\n    freq = {}\n    for ch in text:",
        "def find_median(arr):\n    arr.sort()\n    n = len(arr)\n    if",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot",
        "def bfs(graph, start):\n    visited = set()\n    queue = [start]",
        "def dijkstra(graph, source):\n    distances = {node: float('inf') for",
        "def knapsack(weights, values, capacity):\n    n = len(weights)\n    dp",
        "def coin_change(coins, amount):\n    dp = [float('inf')] * (amount",
        "def edit_distance(s1, s2):\n    m, n = len(s1), len(",
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
        "How do I love thee? Let me count the ways. I",
        "The road not taken is the one that leads to where",
        "In Xanadu did Kubla Khan a stately pleasure dome decree",
        "I think that I shall never see a poem lovely as",
        "The world is too much with us late and soon getting",
        "Tyger tyger burning bright in the forests of the night what",
        "O Captain my Captain our fearful trip is done the ship",
        "Hope is the thing with feathers that perches in the soul",
        "I hear America singing the varied carols I hear each one",
        "The love song of a wandering soul who seeks the distant",
        "Stop all the clocks cut off the telephone prevent the dog",
        "Whose woods these are I think I know his house is",
        "Let us go then you and I when the evening is",
        "I carry your heart with me I carry it in my",
        "Season of mists and mellow fruitfulness close bosom friend of",
        "My heart leaps up when I behold a rainbow in the",
        "Out of the night that covers me black as the pit",
        "Wild nights wild nights were I with thee wild nights should",
        "Gather ye rosebuds while ye may old time is still a",
        "Come live with me and be my love and we will",
    ],
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
        "The village was nestled in a valley surrounded by mountains",
        "Every morning she would wake up before the alarm and",
        "The letter arrived on a Tuesday and changed everything about",
        "He had never been the kind of person who enjoyed",
        "The garden behind the house was overgrown with wildflowers",
        "They met at a coffee shop on the corner of",
        "The wind howled through the empty corridors of the old",
        "She picked up the phone and dialed the number she",
        "The children played in the yard while their parents watched",
        "It was the kind of day when nothing seemed to go",
        "The train pulled into the station just as the clock",
        "He opened the door to find a package sitting on",
        "The ocean stretched out before them endless and blue and",
        "She closed her eyes and tried to remember the last",
        "The market was crowded with vendors selling fruits and vegetables",
        "He walked along the beach collecting shells and thinking about",
        "The snow began to fall softly covering the ground in",
        "She sat at her desk staring at the blank page",
        "The forest was dense and dark but he kept walking",
        "They drove for hours along the winding mountain road until",
    ],
}


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


def make_mlp_clf():
    """Standard nonlinear probe: 2-layer MLP."""
    return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                         random_state=42, early_stopping=True,
                         validation_fraction=0.15, alpha=0.01)


def make_linear_clf():
    return LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")


# ================================================================
# SECTION 1: Qwen-7B code staircase FIXED
# ================================================================
def section1_qwen_code():
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 1: QWEN-7B CODE STAIRCASE (PROPER BASELINES)")
    logger.info("=" * 70)
    
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-7B",
                                               device="cuda", dtype=torch.float16)
    model.eval()
    layers = get_layers(model.cfg.n_layers)
    
    targets = sorted(set(r for _, r in SIGS))
    t2i = {t: i for i, t in enumerate(targets)}
    labels = np.array([t2i[r] for _, r in SIGS])
    logger.info(f"  {len(SIGS)} sigs, {len(targets)} types, {len(layers)} layers")
    
    # Proper baselines
    X_np = build_np_features(SIGS)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    np_lin = cross_val_score(make_linear_clf(), X_np, labels, cv=cv, scoring="accuracy").mean()
    np_mlp = cross_val_score(make_mlp_clf(), X_np, labels, cv=cv, scoring="accuracy").mean()
    logger.info(f"  N+P linear: {np_lin:.4f}")
    logger.info(f"  N+P MLP:    {np_mlp:.4f}")
    
    # Extract activations
    all_acts = {l: [] for l in layers}
    for si, (sig, _) in enumerate(SIGS):
        tokens = model.to_tokens(sig + "\n    ", prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens,
                names_filter=[f"blocks.{l}.hook_resid_post" for l in layers])
        for l in layers:
            all_acts[l].append(cache[f"blocks.{l}.hook_resid_post"][0, -1, :].cpu().numpy())
        del cache; torch.cuda.empty_cache()
        if (si+1) % 25 == 0:
            logger.info(f"    Extracted {si+1}/{len(SIGS)}")
    
    logger.info(f"\n  {'Layer':>6} {'LinProbe':>9} {'MLPProbe':>9} {'N+P_Lin':>8} {'N+P_MLP':>8} {'LinGap':>8} {'MLPGap':>8}")
    
    results = {"model": "Qwen/Qwen2.5-7B", "n_sigs": len(SIGS),
               "np_linear": float(np_lin), "np_mlp": float(np_mlp)}
    layer_results = {}
    
    for l in layers:
        X = np.stack(all_acts[l])
        X_s = StandardScaler().fit_transform(X)
        if X_s.shape[1] > PCA_DIM:
            X_s = PCA(n_components=min(PCA_DIM, X.shape[0]-1), random_state=42).fit_transform(X_s)
        
        la = cross_val_score(make_linear_clf(), X_s, labels, cv=cv, scoring="accuracy").mean()
        ma = cross_val_score(make_mlp_clf(), X_s, labels, cv=cv, scoring="accuracy").mean()
        lg = la - np_lin
        mg = ma - np_mlp
        
        logger.info(f"  L{l:>4} {la:>9.4f} {ma:>9.4f} {np_lin:>8.4f} {np_mlp:>8.4f} {lg:>+8.4f} {mg:>+8.4f}")
        layer_results[str(l)] = {"lin_probe": float(la), "mlp_probe": float(ma),
                                  "lin_gap": float(lg), "mlp_gap": float(mg)}
    
    results["layers"] = layer_results
    bl = max(layer_results.items(), key=lambda x: x[1]["lin_probe"])
    bm = max(layer_results.items(), key=lambda x: x[1]["mlp_probe"])
    results["best_linear"] = {"layer": bl[0], "acc": bl[1]["lin_probe"], "gap": bl[1]["lin_gap"]}
    results["best_mlp"] = {"layer": bm[0], "acc": bm[1]["mlp_probe"], "gap": bm[1]["mlp_gap"]}
    
    del model; torch.cuda.empty_cache(); gc.collect()
    return results


# ================================================================
# SECTION 2: Qwen-7B behavioral
# ================================================================
def section2_qwen_behavioral():
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 2: QWEN-7B BEHAVIORAL CODE GENERATION")
    logger.info("=" * 70)
    
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-7B",
                                               device="cuda", dtype=torch.float16)
    model.eval()
    
    targets = sorted(set(r for _, r in SIGS))
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
        if 'return true' in gen_text or 'return false' in gen_text or 'return not ' in gen_text:
            predicted = 'bool'
        elif 'return []' in gen_text or 'return [' in gen_text or 'return list(' in gen_text or 'result.append' in gen_text:
            predicted = 'list'
        elif 'return ""' in gen_text or "return ''" in gen_text or 'return f"' in gen_text or "return f'" in gen_text or 'return str(' in gen_text or '.join(' in gen_text:
            predicted = 'str'
        elif 'return 0.0' in gen_text or 'return float(' in gen_text or 'math.' in gen_text or '/ ' in gen_text.split('\n')[0] if '\n' in gen_text else '':
            predicted = 'float'
        elif any(kw in gen_text for kw in ['return 0', 'return len(', 'return int(', 'return n', 'return count', 'return sum(']):
            predicted = 'int'
        
        # Fallback: check return type annotation if present
        if predicted is None and '->' in gen_text:
            for t in targets:
                if f'-> {t}' in gen_text:
                    predicted = t
                    break
        
        if predicted == expected:
            correct += 1
            type_correct[expected] += 1
        type_total[expected] += 1
        total += 1
        
        if (si+1) % 25 == 0:
            logger.info(f"    {si+1}/{len(SIGS)}: running acc = {correct/total:.3f}")
    
    behavioral = correct / total
    logger.info(f"\n  Overall behavioral: {behavioral:.4f} ({correct}/{total})")
    for t in targets:
        if type_total[t] > 0:
            logger.info(f"    {t}: {type_correct[t]}/{type_total[t]} = {type_correct[t]/type_total[t]:.3f}")
    
    results = {"model": "Qwen/Qwen2.5-7B", "behavioral": float(behavioral),
               "correct": correct, "total": total,
               "per_type": {t: {"correct": type_correct[t], "total": type_total[t],
                                "acc": float(type_correct[t]/type_total[t]) if type_total[t] > 0 else 0}
                            for t in targets}}
    
    del model; torch.cuda.empty_cache(); gc.collect()
    return results


# ================================================================
# SECTION 3: Nonlinear probes on code (GPT-J + Qwen-7B)
# ================================================================
def section3_nonlinear_code(model_name):
    logger.info(f"\n" + "=" * 70)
    logger.info(f"SECTION 3: NONLINEAR PROBES ON CODE — {model_name}")
    logger.info("=" * 70)
    
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=torch.float16)
    model.eval()
    layers = get_layers(model.cfg.n_layers)
    
    targets = sorted(set(r for _, r in SIGS))
    t2i = {t: i for i, t in enumerate(targets)}
    labels = np.array([t2i[r] for _, r in SIGS])
    
    X_np = build_np_features(SIGS)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    np_lin = cross_val_score(make_linear_clf(), X_np, labels, cv=cv, scoring="accuracy").mean()
    np_mlp = cross_val_score(make_mlp_clf(), X_np, labels, cv=cv, scoring="accuracy").mean()
    logger.info(f"  N+P linear: {np_lin:.4f}, N+P MLP: {np_mlp:.4f}")
    
    all_acts = {l: [] for l in layers}
    for si, (sig, _) in enumerate(SIGS):
        tokens = model.to_tokens(sig + "\n    ", prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens,
                names_filter=[f"blocks.{l}.hook_resid_post" for l in layers])
        for l in layers:
            all_acts[l].append(cache[f"blocks.{l}.hook_resid_post"][0, -1, :].cpu().numpy())
        del cache; torch.cuda.empty_cache()
        if (si+1) % 25 == 0:
            logger.info(f"    Extracted {si+1}/{len(SIGS)}")
    
    logger.info(f"\n  {'Layer':>6} {'LinP':>7} {'MLPP':>7} {'LinB':>7} {'MLPB':>7} {'LinG':>7} {'MLPG':>7}")
    
    results = {"model": model_name, "n_sigs": len(SIGS),
               "np_linear": float(np_lin), "np_mlp": float(np_mlp)}
    layer_results = {}
    
    for l in layers:
        X = np.stack(all_acts[l])
        X_s = StandardScaler().fit_transform(X)
        if X_s.shape[1] > PCA_DIM:
            X_s = PCA(n_components=min(PCA_DIM, X.shape[0]-1), random_state=42).fit_transform(X_s)
        
        la = cross_val_score(make_linear_clf(), X_s, labels, cv=cv, scoring="accuracy").mean()
        ma = cross_val_score(make_mlp_clf(), X_s, labels, cv=cv, scoring="accuracy").mean()
        
        logger.info(f"  L{l:>4} {la:>7.4f} {ma:>7.4f} {np_lin:>7.4f} {np_mlp:>7.4f} "
                    f"{la-np_lin:>+7.4f} {ma-np_mlp:>+7.4f}")
        layer_results[str(l)] = {"lin_probe": float(la), "mlp_probe": float(ma),
                                  "lin_gap": float(la - np_lin), "mlp_gap": float(ma - np_mlp)}
    
    results["layers"] = layer_results
    del model; torch.cuda.empty_cache(); gc.collect()
    return results


# ================================================================
# SECTION 4: Nonlinear probes on domains (GPT-J, K=3)
# ================================================================
def section4_nonlinear_domains():
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 4: NONLINEAR PROBES ON DOMAINS (GPT-J-6B, K=3)")
    logger.info("=" * 70)
    
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("EleutherAI/gpt-j-6b",
                                               device="cuda", dtype=torch.float16)
    model.eval()
    layers = get_layers(model.cfg.n_layers)
    W_E = model.W_E.detach()
    K = 3
    N_GEN = 60
    MIN_TARGET = 8
    
    results = {"model": "EleutherAI/gpt-j-6b", "k": K}
    
    for domain_name, prompts in DOMAIN_PROMPTS.items():
        logger.info(f"\n  Domain: {domain_name} ({len(prompts)} prompts)")
        
        all_seqs = []
        for pi, prompt in enumerate(prompts):
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                gen = model.generate(tokens, max_new_tokens=N_GEN, temperature=0.0)
            all_seqs.append({"prompt_len": tokens.shape[1], "full_ids": gen[0].cpu().tolist()})
        
        # 15 train / 15 test for 30-prompt domains
        split = len(prompts) // 2
        test_seqs = all_seqs[split:]
        
        test_tgts = []
        for seq in test_seqs:
            ids = seq["full_ids"]; pl = seq["prompt_len"]
            for n in range(pl, len(ids) - K):
                test_tgts.append(ids[n + K])
        tc = Counter(test_tgts)
        frequent = {t for t, c in tc.items() if c >= MIN_TARGET}
        t2i = {t: i for i, t in enumerate(sorted(frequent))}
        n_cls = len(t2i)
        if n_cls < 3:
            logger.info(f"    {n_cls} classes, skip")
            continue
        
        activations = {l: [] for l in layers}
        ctx_embs, labels_list = [], []
        
        for seq in test_seqs:
            ids = seq["full_ids"]; pl = seq["prompt_len"]
            inp = torch.tensor([ids], device="cuda")
            with torch.no_grad():
                _, cache = model.run_with_cache(inp,
                    names_filter=[f"blocks.{l}.hook_resid_post" for l in layers])
            for n in range(pl, len(ids) - K):
                tgt = ids[n + K]
                if tgt not in t2i: continue
                labels_list.append(t2i[tgt])
                for l in layers:
                    activations[l].append(cache[f"blocks.{l}.hook_resid_post"][0, n, :].cpu().numpy())
                ws = max(0, n - 4)
                ctx = W_E[torch.tensor(ids[ws:n+1], device="cuda")].cpu().numpy()
                ctx_embs.append(ctx.mean(axis=0))
            del cache; torch.cuda.empty_cache()
        
        labels = np.array(labels_list)
        n_ex = len(labels)
        if n_ex < 30: continue
        min_c = min(Counter(labels).values())
        n_splits = min(5, min_c)
        if n_splits < 2: continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Context baselines (linear + MLP)
        X_ctx = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(
            StandardScaler().fit_transform(np.stack(ctx_embs)))
        ctx_lin = cross_val_score(make_linear_clf(), X_ctx, labels, cv=cv, scoring="accuracy").mean()
        ctx_mlp = cross_val_score(make_mlp_clf(), X_ctx, labels, cv=cv, scoring="accuracy").mean()
        
        # Best probe (linear + MLP) across layers
        best_la, best_ma = 0, 0
        best_ll, best_ml = 0, 0
        
        for l in layers:
            X = PCA(n_components=min(PCA_DIM, n_ex-1), random_state=42).fit_transform(
                StandardScaler().fit_transform(np.stack(activations[l])))
            la = cross_val_score(make_linear_clf(), X, labels, cv=cv, scoring="accuracy").mean()
            ma = cross_val_score(make_mlp_clf(), X, labels, cv=cv, scoring="accuracy").mean()
            if la > best_la: best_la, best_ll = la, l
            if ma > best_ma: best_ma, best_ml = ma, l
        
        lin_gap = best_la - ctx_lin
        mlp_gap = best_ma - ctx_mlp
        
        logger.info(f"    {n_ex} ex, {n_cls} cls")
        logger.info(f"    Linear: ctx={ctx_lin:.3f} probe(L{best_ll})={best_la:.3f} gap={lin_gap:+.3f}")
        logger.info(f"    MLP:    ctx={ctx_mlp:.3f} probe(L{best_ml})={best_ma:.3f} gap={mlp_gap:+.3f}")
        logger.info(f"    Does MLP change conclusion? gap_diff = {mlp_gap - lin_gap:+.3f}")
        
        results[domain_name] = {
            "n_examples": n_ex, "n_classes": n_cls,
            "linear_ctx": float(ctx_lin), "linear_probe": float(best_la),
            "linear_gap": float(lin_gap), "linear_best_layer": int(best_ll),
            "mlp_ctx": float(ctx_mlp), "mlp_probe": float(best_ma),
            "mlp_gap": float(mlp_gap), "mlp_best_layer": int(best_ml),
            "gap_difference": float(mlp_gap - lin_gap),
        }
    
    del model; torch.cuda.empty_cache(); gc.collect()
    return results


# ================================================================
# SECTION 5: Pairwise domain significance (no GPU)
# ================================================================
def section5_pairwise():
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 5: PAIRWISE DOMAIN SIGNIFICANCE TESTS")
    logger.info("=" * 70)
    
    data = json.load(open(f"{OUTDIR}/overnight_phase1a_domains.json"))
    ds = data["gptj_domains_50"]
    
    domains = ["chain_of_thought", "chain_of_thought_scrambled", "chain_of_thought_nonmath",
               "free_prose", "structured_prose", "code", "poetry"]
    
    k3_data = {}
    for dom in domains:
        if dom in ds and "k3" in ds[dom]:
            k3_data[dom] = ds[dom]["k3"]
    
    logger.info(f"\n  K=3 gaps (sorted):")
    sorted_doms = sorted(k3_data.items(), key=lambda x: -x[1]["gap"])
    for dom, d in sorted_doms:
        logger.info(f"    {dom:<30} gap={d['gap']:+.3f} [{d.get('gap_ci_lo',0):+.3f}, {d.get('gap_ci_hi',0):+.3f}]")
    
    logger.info(f"\n  Pairwise CI overlap tests:")
    logger.info(f"  {'A':<25} {'B':<25} {'Diff':>7} {'Result':>15}")
    
    results = {"model": "GPT-J-6B", "k": 3, "comparisons": []}
    
    for i in range(len(sorted_doms)):
        for j in range(i+1, min(i+3, len(sorted_doms))):  # Compare adjacent pairs
            da, dd_a = sorted_doms[i]
            db, dd_b = sorted_doms[j]
            diff = dd_a["gap"] - dd_b["gap"]
            
            a_lo, a_hi = dd_a.get("gap_ci_lo", dd_a["gap"]), dd_a.get("gap_ci_hi", dd_a["gap"])
            b_lo, b_hi = dd_b.get("gap_ci_lo", dd_b["gap"]), dd_b.get("gap_ci_hi", dd_b["gap"])
            
            if a_lo > b_hi:
                result = "A > B (SIG)"
            elif b_lo > a_hi:
                result = "B > A (SIG)"
            else:
                result = "Overlap (ns)"
            
            logger.info(f"  {da[:23]:<25} {db[:23]:<25} {diff:>+7.3f} {result:>15}")
            results["comparisons"].append({
                "a": da, "b": db, "diff": float(diff), "result": result
            })
    
    return results


# ================================================================
# SECTION 6: Cross-model domain correlation (no GPU)
# ================================================================
def section6_cross_model():
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 6: CROSS-MODEL DOMAIN CORRELATION")
    logger.info("=" * 70)
    
    gptj = json.load(open(f"{OUTDIR}/overnight_phase1a_domains.json"))["gptj_domains_50"]
    qwen = json.load(open(f"{OUTDIR}/overnight_complete.json")).get("qwen7b_domains_50", {})
    
    domains = ["chain_of_thought", "chain_of_thought_scrambled", "chain_of_thought_nonmath",
               "free_prose", "structured_prose", "code", "poetry"]
    
    logger.info(f"\n  {'Domain':<30} {'GPT-J K=3':>10} {'Qwen K=3':>10}")
    gj_vals, qw_vals = [], []
    
    for dom in domains:
        g = gptj.get(dom, {}).get("k3", {}).get("gap")
        q = qwen.get(dom, {}).get("k3", {}).get("gap")
        if g is not None and q is not None:
            logger.info(f"  {dom:<30} {g:>+10.3f} {q:>+10.3f}")
            gj_vals.append(g)
            qw_vals.append(q)
    
    results = {"domains": domains}
    
    if len(gj_vals) >= 4:
        rho, p = spearmanr(gj_vals, qw_vals)
        logger.info(f"\n  Spearman rho = {rho:.3f}, p = {p:.3f}")
        logger.info(f"  {'CORRELATED' if p < 0.05 else 'NOT correlated'}: domain ordering {'does' if p < 0.05 else 'does not'} replicate across models")
        results["spearman_rho"] = float(rho)
        results["spearman_p"] = float(p)
    
    # Also K=1
    logger.info(f"\n  {'Domain':<30} {'GPT-J K=1':>10} {'Qwen K=1':>10}")
    gj1, qw1 = [], []
    for dom in domains:
        g = gptj.get(dom, {}).get("k1", {}).get("gap")
        q = qwen.get(dom, {}).get("k1", {}).get("gap")
        if g is not None and q is not None:
            logger.info(f"  {dom:<30} {g:>+10.3f} {q:>+10.3f}")
            gj1.append(g); qw1.append(q)
    
    if len(gj1) >= 4:
        rho1, p1 = spearmanr(gj1, qw1)
        logger.info(f"  K=1 Spearman rho = {rho1:.3f}, p = {p1:.3f}")
        results["spearman_k1_rho"] = float(rho1)
        results["spearman_k1_p"] = float(p1)
    
    return results


# ================================================================
# MAIN
# ================================================================
def main():
    logger.info("=" * 70)
    logger.info("FINAL COMPREHENSIVE SCRIPT — ALL REMAINING EXPERIMENTS")
    logger.info("=" * 70)
    t_start = time.time()
    
    all_results = {}
    
    # Section 1: Qwen-7B code staircase (proper)
    all_results["s1_qwen_code"] = section1_qwen_code()
    save(all_results, "s1")
    
    # Section 2: Qwen-7B behavioral
    all_results["s2_qwen_behavioral"] = section2_qwen_behavioral()
    save(all_results, "s2")
    
    # Section 3a: Nonlinear code on GPT-J
    all_results["s3a_gptj_nonlinear_code"] = section3_nonlinear_code("EleutherAI/gpt-j-6b")
    save(all_results, "s3a")
    
    # Section 3b: Nonlinear code on Qwen-7B (reuses activations concept from s1)
    # Actually s1 already has both linear and MLP for Qwen-7B, so skip duplicate
    logger.info("\n  [NOTE] Qwen-7B nonlinear code already done in Section 1")
    
    # Section 4: Nonlinear domains on GPT-J
    all_results["s4_gptj_nonlinear_domains"] = section4_nonlinear_domains()
    save(all_results, "s4")
    
    # Section 5: Pairwise tests (no GPU)
    all_results["s5_pairwise"] = section5_pairwise()
    save(all_results, "s5")
    
    # Section 6: Cross-model correlation (no GPU)
    all_results["s6_cross_model"] = section6_cross_model()
    save(all_results, "s6")
    
    # Final save
    save(all_results, "complete")
    
    elapsed = (time.time() - t_start) / 3600
    logger.info(f"\n{'='*70}")
    logger.info(f"ALL DONE — {elapsed:.1f} hours")
    logger.info(f"{'='*70}")
    
    # Summary
    s1 = all_results.get("s1_qwen_code", {})
    if "best_linear" in s1:
        logger.info(f"\n  QWEN-7B CODE:")
        logger.info(f"    N+P: linear={s1['np_linear']:.4f} MLP={s1['np_mlp']:.4f}")
        logger.info(f"    Best linear probe: {s1['best_linear']['acc']:.4f} gap={s1['best_linear']['gap']:+.4f}")
        logger.info(f"    Best MLP probe:    {s1['best_mlp']['acc']:.4f} gap={s1['best_mlp']['gap']:+.4f}")
    
    s2 = all_results.get("s2_qwen_behavioral", {})
    if "behavioral" in s2:
        logger.info(f"    Behavioral: {s2['behavioral']:.4f}")
    
    s3 = all_results.get("s3a_gptj_nonlinear_code", {})
    if "layers" in s3:
        bl = max(s3["layers"].items(), key=lambda x: x[1]["lin_probe"])
        bm = max(s3["layers"].items(), key=lambda x: x[1]["mlp_probe"])
        logger.info(f"\n  GPT-J CODE:")
        logger.info(f"    N+P: linear={s3['np_linear']:.4f} MLP={s3['np_mlp']:.4f}")
        logger.info(f"    Best linear: L{bl[0]} {bl[1]['lin_probe']:.4f} gap={bl[1]['lin_gap']:+.4f}")
        logger.info(f"    Best MLP:    L{bm[0]} {bm[1]['mlp_probe']:.4f} gap={bm[1]['mlp_gap']:+.4f}")
    
    s4 = all_results.get("s4_gptj_nonlinear_domains", {})
    if s4:
        logger.info(f"\n  GPT-J NONLINEAR DOMAINS (K=3):")
        logger.info(f"  {'Domain':<25} {'Lin_gap':>8} {'MLP_gap':>8} {'Change':>8}")
        for dom in ["chain_of_thought", "code", "poetry", "free_prose"]:
            if dom in s4 and isinstance(s4[dom], dict):
                d = s4[dom]
                logger.info(f"  {dom:<25} {d['linear_gap']:>+8.3f} {d['mlp_gap']:>+8.3f} {d['gap_difference']:>+8.3f}")


if __name__ == "__main__":
    main()
