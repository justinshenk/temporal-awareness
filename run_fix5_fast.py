import json, os, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()
from transformer_lens import HookedTransformer

SIGS = [
    ('def add(a, b):', 'int'), ('def subtract(a, b):', 'int'), ('def multiply(a, b):', 'int'),
    ('def divide_int(a, b):', 'int'), ('def modulo(a, b):', 'int'), ('def power(base, exp):', 'int'),
    ('def count_words(text):', 'int'), ('def count_chars(text):', 'int'), ('def count_lines(text):', 'int'),
    ('def factorial(n):', 'int'), ('def fibonacci(n):', 'int'), ('def find_max(numbers):', 'int'),
    ('def find_min(numbers):', 'int'), ('def sum_list(numbers):', 'int'), ('def product(numbers):', 'int'),
    ('def string_length(s):', 'int'), ('def index_of(items, target):', 'int'),
    ('def count_vowels(text):', 'int'), ('def hamming_distance(s1, s2):', 'int'),
    ('def num_digits(n):', 'int'),
    ('def greet(name):', 'str'), ('def farewell(name):', 'str'), ('def to_upper(text):', 'str'),
    ('def to_lower(text):', 'str'), ('def capitalize(text):', 'str'), ('def strip_whitespace(text):', 'str'),
    ('def reverse_string(s):', 'str'), ('def repeat_string(s, n):', 'str'),
    ('def join_words(words):', 'str'), ('def first_word(text):', 'str'), ('def last_word(text):', 'str'),
    ('def remove_spaces(s):', 'str'), ('def replace_char(s, old, new):', 'str'),
    ('def first_name(full_name):', 'str'), ('def last_name(full_name):', 'str'),
    ('def format_date(year, month, day):', 'str'), ('def format_time(hours, minutes):', 'str'),
    ('def to_binary(n):', 'str'), ('def to_hex(n):', 'str'), ('def slug(text):', 'str'),
    ('def is_even(n):', 'bool'), ('def is_odd(n):', 'bool'), ('def is_positive(x):', 'bool'),
    ('def is_negative(x):', 'bool'), ('def is_zero(x):', 'bool'), ('def is_prime(n):', 'bool'),
    ('def is_palindrome(s):', 'bool'), ('def is_empty(s):', 'bool'), ('def is_sorted(items):', 'bool'),
    ('def contains(items, target):', 'bool'), ('def starts_with(text, prefix):', 'bool'),
    ('def ends_with(text, suffix):', 'bool'), ('def is_alpha(text):', 'bool'),
    ('def is_digit(text):', 'bool'), ('def is_upper(text):', 'bool'), ('def is_lower(text):', 'bool'),
    ('def has_duplicates(items):', 'bool'), ('def all_positive(numbers):', 'bool'),
    ('def any_negative(numbers):', 'bool'), ('def is_valid_email(text):', 'bool'),
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
    ('def average(numbers):', 'float'), ('def median(numbers):', 'float'),
    ('def variance(numbers):', 'float'), ('def std_dev(numbers):', 'float'),
    ('def to_celsius(f):', 'float'), ('def to_fahrenheit(c):', 'float'),
    ('def percentage(part, total):', 'float'), ('def ratio(a, b):', 'float'),
    ('def distance(x1, y1, x2, y2):', 'float'), ('def magnitude(x, y, z):', 'float'),
    ('def dot_product(a, b):', 'float'), ('def cosine_similarity(a, b):', 'float'),
    ('def circle_area(radius):', 'float'), ('def sphere_volume(radius):', 'float'),
    ('def triangle_area(base, height):', 'float'), ('def hypotenuse(a, b):', 'float'),
    ('def sigmoid(x):', 'float'), ('def relu(x):', 'float'),
    ('def log_base(x, base):', 'float'), ('def square_root(x):', 'float'),
]

logger.info(f"Loading GPT-J-6B...")
model = HookedTransformer.from_pretrained("EleutherAI/gpt-j-6b", device="cuda", dtype=torch.float16)
model.eval()
n_layers = model.cfg.n_layers
mid_layer = n_layers // 2

targets = sorted(set(r for _, r in SIGS))
t2i = {t: i for i, t in enumerate(targets)}
labels = np.array([t2i[r] for _, r in SIGS])
logger.info(f"{len(SIGS)} sigs, {len(targets)} types, layer {mid_layer}")

FIXED_POS = [0, 1, 2, 3, 5, -1]  # -1 = last position
results = {}

for fp in FIXED_POS:
    acts = []
    for sig, ret in SIGS:
        tokens = model.to_tokens(sig + "\n    ", prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens,
                names_filter=[f"blocks.{mid_layer}.hook_resid_post"])
        pos = fp if fp >= 0 else tokens.shape[1] - 1
        if pos < tokens.shape[1]:
            acts.append(cache[f"blocks.{mid_layer}.hook_resid_post"][0, pos, :].cpu().numpy())
        else:
            acts.append(np.zeros(model.cfg.d_model))
        del cache; torch.cuda.empty_cache()

    X = StandardScaler().fit_transform(np.stack(acts))
    if X.shape[1] > 128:
        X = PCA(n_components=min(128, X.shape[0]-1), random_state=42).fit_transform(X)
    cv = StratifiedKFold(n_splits=min(5, min(Counter(labels).values())), shuffle=True, random_state=42)
    acc = cross_val_score(LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                          X, labels, cv=cv, scoring="accuracy").mean()

    pos_name = "last" if fp == -1 else str(fp)
    logger.info(f"  pos={pos_name}: {acc:.3f}")
    results[pos_name] = float(acc)

logger.info(f"\n=== 100-SIG FIXED POSITION (Layer {mid_layer}) ===")
for fp in FIXED_POS:
    pos_name = "last" if fp == -1 else str(fp)
    logger.info(f"  pos={pos_name}: {results[pos_name]:.3f}")

outfile = "results/lookahead/final/fix5_100sigs.json"
os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, "w") as f:
    json.dump({"n_sigs": len(SIGS), "layer": mid_layer, "results": results}, f, indent=2)
logger.info(f"\nSaved to {outfile}")
logger.info("DONE — FIX 5 100 SIGS")
