#!/usr/bin/env python3
"""
RQ4 WEAKNESS FIXES:
1. Bootstrap 1000 for SantaCoder + CodeLlama (stable p-values)
2. CodeLlama with PCA 256 (check if 128 crushed signal)
3. Pythia scaling: 410M, 1B, 1.4B (emergence curve)
"""

import json, logging, os, sys, hashlib, time, random
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()
sys.path.insert(0, os.getcwd())

from transformer_lens import HookedTransformer
from src.lookahead.utils.types import PlanningExample, TaskType
from src.lookahead.probing.activation_extraction import extract_activations_batch
from src.lookahead.probing.behavioral_validation import run_behavioral_validation, compute_behavioral_summary
from src.lookahead.datasets.rhyme import generate_rhyme_dataset

# ================================================================
# DATASET (same as main script)
# ================================================================
UNTYPED_LARGE = [
    ('def add(a, b):', 'int'), ('def subtract(a, b):', 'int'), ('def multiply(a, b):', 'int'),
    ('def divide_int(a, b):', 'int'), ('def modulo(a, b):', 'int'), ('def power(base, exp):', 'int'),
    ('def count_words(text):', 'int'), ('def count_chars(text):', 'int'), ('def count_lines(text):', 'int'),
    ('def factorial(n):', 'int'), ('def fibonacci(n):', 'int'), ('def find_max(numbers):', 'int'),
    ('def find_min(numbers):', 'int'), ('def sum_list(numbers):', 'int'), ('def product(numbers):', 'int'),
    ('def string_length(s):', 'int'), ('def index_of(items, target):', 'int'),
    ('def count_vowels(text):', 'int'), ('def hamming_distance(s1, s2):', 'int'),
    ('def num_digits(n):', 'int'), ('def gcd(a, b):', 'int'), ('def lcm(a, b):', 'int'),
    ('def abs_value(n):', 'int'), ('def sign(n):', 'int'), ('def clamp(val, lo, hi):', 'int'),
    ('def popcount(n):', 'int'), ('def manhattan_distance(x1, y1, x2, y2):', 'int'),
    ('def depth(tree):', 'int'), ('def height(node):', 'int'), ('def size(collection):', 'int'),
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
    ('def pad_right(s, width, char):', 'str'), ('def truncate(text, length):', 'str'),
    ('def extract_domain(email):', 'str'), ('def extract_extension(filename):', 'str'),
    ('def base64_encode(data):', 'str'),
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
    ('def is_vowel(char):', 'bool'), ('def is_consonant(char):', 'bool'),
    ('def file_exists(path):', 'bool'), ('def is_balanced(parens):', 'bool'),
    ('def is_symmetric(matrix):', 'bool'), ('def is_connected(graph):', 'bool'),
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
    ('def permutations(items):', 'list'), ('def combinations(items, k):', 'list'),
    ('def topk(items, k):', 'list'), ('def bottomk(items, k):', 'list'),
    ('def sliding_window(items, size):', 'list'), ('def rotate(items, k):', 'list'),
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
    ('def moving_average(values, window):', 'float'), ('def entropy(probs):', 'float'),
    ('def rmse(predicted, actual):', 'float'), ('def correlation(x, y):', 'float'),
]


def make_examples():
    examples = []
    for idx, (sig, ret) in enumerate(UNTYPED_LARGE):
        examples.append(PlanningExample(
            task_type=TaskType.CODE_RETURN, prompt=sig + "\n    ", target_value=ret,
            target_token_positions=[],
            example_id=hashlib.md5(f"large_{idx}".encode()).hexdigest()[:12],
            metadata={"signature": sig, "has_type_annotation": False,
                      "return_type": ret, "is_control": False},
        ))
    return examples


def make_nonsense_examples(examples):
    random.seed(42)
    names = ["xyzq", "blorpf", "qwmx", "fnrd", "ghtk", "zmvp", "krtl", "dwqn",
             "pxvs", "jhlm", "vbnq", "wrtx", "ycfg", "nmzk", "tplr", "sdhj",
             "bxcv", "lfgn", "mwqz", "rkpt", "hdsx", "jvnq", "cwfl", "nbtm",
             "gzpr", "xkdw", "fqms", "ylrv", "thcn", "pdwk"]
    out = []
    for ex in examples:
        sig = ex.metadata["signature"]
        params = sig[sig.index("("):]
        nn = random.choice(names)
        new_sig = f"def {nn}{params}"
        out.append(PlanningExample(
            task_type=TaskType.CODE_RETURN, prompt=new_sig + "\n    ",
            target_value=ex.target_value, target_token_positions=[],
            example_id=hashlib.md5(f"nonsense_{nn}_{params}".encode()).hexdigest()[:12],
            metadata={"signature": new_sig, "original": sig,
                      "has_type_annotation": False, "return_type": ex.target_value,
                      "is_control": False},
        ))
    return out


def run_probing_with_pca(model, examples, labels, targets, layers, pca_dim, n_boot):
    """Run probing with configurable PCA dim and bootstrap count."""
    chance = 1.0 / len(targets)
    results = {}
    caches = extract_activations_batch(model, model.tokenizer, examples, layers=layers, device="cuda")

    for layer in layers:
        min_seq = min(len(c.token_ids) for c in caches)
        best_acc, best_pos = 0, 0

        for pos in range(min_seq):
            X = np.stack([caches[i].activations[layer][pos] for i in range(len(examples))])
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            if X_s.shape[1] > pca_dim:
                X_s = PCA(n_components=min(pca_dim, X_s.shape[0]-1, X_s.shape[1]), random_state=42).fit_transform(X_s)
            probe = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(probe, X_s, labels, cv=cv, scoring="accuracy")
            if scores.mean() > best_acc:
                best_acc = scores.mean()
                best_pos = pos

        # Name-only baseline
        name_acts = np.stack([
            caches[i].activations[layer][min(2, len(caches[i].token_ids) - 1)]
            for i in range(len(examples))
        ])
        scaler_n = StandardScaler()
        name_s = scaler_n.fit_transform(name_acts)
        if name_s.shape[1] > pca_dim:
            name_s = PCA(n_components=min(pca_dim, name_s.shape[0]-1, name_s.shape[1]), random_state=42).fit_transform(name_s)
        name_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
            name_s, labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy",
        )
        best_name = name_scores.mean()

        # BoW baseline
        bow_dim = min(max(max(c.token_ids) for c in caches) + 1, 50257)
        X_bow = np.zeros((len(examples), bow_dim), dtype=np.float32)
        for row in range(len(examples)):
            for t in range(min(best_pos + 1, len(caches[row].token_ids))):
                tid = caches[row].token_ids[t]
                if tid < bow_dim:
                    X_bow[row, tid] = 1.0
        nz = X_bow.sum(axis=0) > 0
        X_bow_f = X_bow[:, nz]
        if X_bow_f.shape[1] > 0:
            scaler_b = StandardScaler()
            X_bow_s = scaler_b.fit_transform(X_bow_f)
            bow_scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                X_bow_s, labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="accuracy",
            )
            best_bow = bow_scores.mean()
        else:
            best_bow = chance

        # Bootstrap CI + probe vs name-only test
        X_best = np.stack([caches[i].activations[layer][best_pos] for i in range(len(examples))])
        scaler_bp = StandardScaler()
        X_best_s = scaler_bp.fit_transform(X_best)
        if X_best_s.shape[1] > pca_dim:
            X_best_s = PCA(n_components=min(pca_dim, X_best_s.shape[0]-1, X_best_s.shape[1]), random_state=42).fit_transform(X_best_s)

        rng = np.random.RandomState(42)
        boot_accs = []
        probe_wins = 0
        for _ in range(n_boot):
            idx = rng.choice(len(X_best_s), len(X_best_s), replace=True)
            oob = list(set(range(len(X_best_s))) - set(idx))
            if len(oob) < 5 or len(np.unique(labels[idx])) < len(targets):
                continue
            p1 = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            p1.fit(X_best_s[idx], labels[idx])
            a1 = p1.score(X_best_s[oob], labels[oob])
            boot_accs.append(a1)

            p2 = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            p2.fit(name_s[idx], labels[idx])
            a2 = p2.score(name_s[oob], labels[oob])
            if a1 > a2:
                probe_wins += 1

        ci_lo, ci_hi = np.percentile(boot_accs, [2.5, 97.5]) if boot_accs else (0, 0)
        p_value = 1.0 - (probe_wins / len(boot_accs)) if boot_accs else 1.0

        logger.info(
            f"  L{layer}: probe={best_acc:.3f} [{ci_lo:.3f},{ci_hi:.3f}] "
            f"BoW={best_bow:.3f} name={best_name:.3f} "
            f"gap_name={best_acc - best_name:.3f} "
            f"p={p_value:.4f} pos={best_pos} (pca={pca_dim}, boot={n_boot})"
        )

        results[f"layer_{layer}"] = {
            "probe": float(best_acc), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "bow": float(best_bow), "name_only": float(best_name),
            "gap_name": float(best_acc - best_name),
            "p_value_vs_name": float(p_value),
            "probe_wins": probe_wins, "n_boot_valid": len(boot_accs),
            "best_pos": int(best_pos), "pca_dim": pca_dim, "n_boot": n_boot,
        }

    return results, caches


def run_nonsense(model, examples, labels, targets, layers, pca_dim):
    """Run nonsense names control."""
    nonsense = make_nonsense_examples(examples)
    n_labels = np.array([{t: i for i, t in enumerate(targets)}[e.target_value] for e in nonsense])
    n_caches = extract_activations_batch(model, model.tokenizer, nonsense, layers=layers, device="cuda")
    chance = 1.0 / len(targets)
    results = {}
    for layer in layers:
        min_seq = min(len(c.token_ids) for c in n_caches)
        best_acc = 0
        for pos in range(min_seq):
            X = np.stack([n_caches[i].activations[layer][pos] for i in range(len(nonsense))])
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            if X_s.shape[1] > pca_dim:
                X_s = PCA(n_components=min(pca_dim, X_s.shape[0]-1, X_s.shape[1]), random_state=42).fit_transform(X_s)
            scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                X_s, n_labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="accuracy",
            )
            best_acc = max(best_acc, scores.mean())
        logger.info(f"  Nonsense L{layer}: {best_acc:.3f} (chance={chance:.3f})")
        results[f"nonsense_L{layer}"] = {"probe": float(best_acc)}
    return results


def run_model(model_name, dtype, pca_dim, n_boot):
    """Run full suite for one model with configurable PCA and bootstrap."""
    logger.info("=" * 70)
    logger.info(f"MODEL: {model_name} (pca={pca_dim}, boot={n_boot})")
    logger.info("=" * 70)

    model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=dtype)
    model.eval()
    n_layers = model.cfg.n_layers
    layers = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]))
    logger.info(f"  {n_layers} layers, d_model={model.cfg.d_model}, probing layers: {layers}")

    examples = make_examples()
    targets = sorted(set(e.target_value for e in examples))
    t2i = {t: i for i, t in enumerate(targets)}
    labels = np.array([t2i[e.target_value] for e in examples])

    result = {"model": model_name, "n_layers": n_layers, "d_model": model.cfg.d_model,
              "pca_dim": pca_dim, "n_boot": n_boot}

    # Behavioral
    logger.info("\n  === BEHAVIORAL ===")
    beh = run_behavioral_validation(model, examples, max_new_tokens=50)
    beh_sum = compute_behavioral_summary(beh)
    for t, s in beh_sum.items():
        logger.info(f"    {t}: {s['task_accuracy']:.1%}")
    result["behavioral"] = beh_sum

    type_correct = {}
    for r in beh:
        type_correct.setdefault(r.target_value, []).append(r.task_success)
    for t in sorted(type_correct):
        logger.info(f"    {t}: {np.mean(type_correct[t]):.1%}")
    result["behavioral_per_type"] = {t: float(np.mean(v)) for t, v in type_correct.items()}

    # Probing
    logger.info(f"\n  === PROBING (pca={pca_dim}, boot={n_boot}) ===")
    probe_results, caches = run_probing_with_pca(model, examples, labels, targets, layers, pca_dim, n_boot)
    result["probing"] = probe_results

    # Nonsense
    logger.info("\n  === NONSENSE ===")
    nonsense_results = run_nonsense(model, examples, labels, targets, layers, pca_dim)
    result.update(nonsense_results)

    # Rhyme behavioral
    logger.info("\n  === RHYME ===")
    rhyme_ex = generate_rhyme_dataset(n_per_rhyme_set=3, include_controls=False)
    beh_r = run_behavioral_validation(model, rhyme_ex, max_new_tokens=50)
    beh_r_sum = compute_behavioral_summary(beh_r)
    for t, s in beh_r_sum.items():
        logger.info(f"    {t}: {s['task_accuracy']:.1%}")
    result["rhyme_behavioral"] = beh_r_sum

    # Save incrementally
    outfile = f"results/lookahead/complete/fix_{model_name.replace('/', '_')}_pca{pca_dim}_boot{n_boot}.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"  Saved to {outfile}")

    del model, caches
    torch.cuda.empty_cache()
    return result


def main():
    logger.info("=" * 70)
    logger.info("RQ4 WEAKNESS FIXES")
    logger.info("=" * 70)

    all_results = {}

    # ============================================================
    # FIX 1: SantaCoder with 1000 bootstrap (was 100)
    # ============================================================
    logger.info("\n>>> FIX 1: SantaCoder bootstrap=1000")
    all_results["santacoder_boot1000"] = run_model(
        "bigcode/santacoder", torch.float16, pca_dim=128, n_boot=1000
    )

    # ============================================================
    # FIX 2a: CodeLlama with 1000 bootstrap + PCA 128 (same as before but more boots)
    # ============================================================
    logger.info("\n>>> FIX 2a: CodeLlama bootstrap=1000, PCA=128")
    all_results["codellama_pca128_boot1000"] = run_model(
        "codellama/CodeLlama-7b-Python-hf", torch.float16, pca_dim=128, n_boot=1000
    )

    # ============================================================
    # FIX 2b: CodeLlama with PCA 256 (check if 128 crushed signal)
    # ============================================================
    logger.info("\n>>> FIX 2b: CodeLlama PCA=256 (check if 128 crushed signal)")
    # Note: max PCA = min(n_samples-1, pca_dim) = min(149, 256) = 149
    # So effectively PCA=149 which captures more variance than 128
    all_results["codellama_pca256_boot1000"] = run_model(
        "codellama/CodeLlama-7b-Python-hf", torch.float16, pca_dim=256, n_boot=1000
    )

    # ============================================================
    # FIX 3: Pythia scaling curve (410M, 1B, 1.4B)
    # ============================================================
    logger.info("\n>>> FIX 3: Pythia scaling curve")

    for pythia_size in ["pythia-410m", "pythia-1b", "pythia-1.4b"]:
        logger.info(f"\n>>> Pythia: {pythia_size}")
        all_results[pythia_size] = run_model(
            pythia_size, torch.float32, pca_dim=128, n_boot=1000
        )

    # Save everything
    outfile = "results/lookahead/complete/weakness_fixes_all.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nAll fixes saved to {outfile}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for name, res in all_results.items():
        if "probing" in res:
            beh = res.get("behavioral", {})
            beh_acc = list(beh.values())[0].get("task_accuracy", "N/A") if beh else "N/A"
            logger.info(f"\n{name}: behavioral={beh_acc}")
            for lk, lv in res["probing"].items():
                logger.info(
                    f"  {lk}: probe={lv['probe']:.3f} [{lv['ci_lo']:.3f},{lv['ci_hi']:.3f}] "
                    f"name={lv['name_only']:.3f} gap={lv['gap_name']:.3f} p={lv['p_value_vs_name']:.4f}"
                )


if __name__ == "__main__":
    main()
