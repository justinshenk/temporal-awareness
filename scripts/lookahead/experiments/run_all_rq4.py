#!/usr/bin/env python3
"""
RQ4 COMPLETE EXPERIMENT SUITE
Runs everything, saves everything, addresses all weaknesses.
"""

import json, logging, os, sys, hashlib, time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()
sys.path.insert(0, os.getcwd())

from transformer_lens import HookedTransformer
from src.lookahead.utils.types import PlanningExample, TaskType
from src.lookahead.probing.activation_extraction import extract_activations_batch
from src.lookahead.probing.commitment_probes import (
    ProbeConfig, train_commitment_probes, compute_commitment_curves, find_commitment_points,
)
from src.lookahead.probing.comprehensive_baselines import bag_of_words_baseline
from src.lookahead.probing.behavioral_validation import run_behavioral_validation, compute_behavioral_summary
from src.lookahead.datasets.code_return import generate_code_return_dataset
from src.lookahead.datasets.rhyme import generate_rhyme_dataset

# ================================================================
# LARGE UNTYPED DATASET (150 examples, 30 per type)
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


def make_examples(dataset=UNTYPED_LARGE):
    examples = []
    for idx, (sig, ret) in enumerate(dataset):
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


def run_probing_suite(model, examples, labels, targets, layers, model_name, task_name):
    """Run full probing with LBFGS, bootstrap CIs, BoW, name-only, and nonsense."""
    chance = 1.0 / len(targets)
    t2i = {t: i for i, t in enumerate(targets)}
    results = {}

    caches = extract_activations_batch(model, model.tokenizer, examples, layers=layers, device="cuda")

    for layer in layers:
        min_seq = min(len(c.token_ids) for c in caches)
        best_acc, best_pos = 0, 0

        for pos in range(min_seq):
            X = np.stack([caches[i].activations[layer][pos] for i in range(len(examples))])
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            if X_s.shape[1] > 256:
                X_s = PCA(n_components=min(128, X_s.shape[0]-1, X_s.shape[1]), random_state=42).fit_transform(X_s)
            probe = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(probe, X_s, labels, cv=cv, scoring="accuracy")
            if scores.mean() > best_acc:
                best_acc = scores.mean()
                best_pos = pos

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
                LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                X_bow_s, labels,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="accuracy",
            )
            best_bow = bow_scores.mean()
        else:
            best_bow = chance

        # Name-only baseline (position 2 = function name token)
        name_acts = np.stack([
            caches[i].activations[layer][min(2, len(caches[i].token_ids) - 1)]
            for i in range(len(examples))
        ])
        scaler_n = StandardScaler()
        name_s = scaler_n.fit_transform(name_acts)
        if name_s.shape[1] > 256:
            name_s = PCA(n_components=min(128, X_s.shape[0]-1, X_s.shape[1]), random_state=42).fit_transform(name_s)
        name_scores = cross_val_score(
            LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
            name_s, labels,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy",
        )
        best_name = name_scores.mean()

        # Bootstrap CI
        X_best = np.stack([caches[i].activations[layer][best_pos] for i in range(len(examples))])
        scaler_bp = StandardScaler()
        X_best_s = scaler_bp.fit_transform(X_best)
        if X_best_s.shape[1] > 256:
            pca_bp = PCA(n_components=min(128, X_s.shape[0]-1, X_s.shape[1]), random_state=42)
            X_best_s = pca_bp.fit_transform(X_best_s)
            pass  # name_s already PCA'd
        rng = np.random.RandomState(42)
        boot_accs = []
        for _ in range(100):
            idx = rng.choice(len(X_best_s), len(X_best_s), replace=True)
            oob = list(set(range(len(X_best_s))) - set(idx))
            if len(oob) < 5 or len(np.unique(labels[idx])) < len(targets):
                continue
            p = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
            p.fit(X_best_s[idx], labels[idx])
            boot_accs.append(p.score(X_best_s[oob], labels[oob]))
        ci_lo, ci_hi = np.percentile(boot_accs, [2.5, 97.5]) if boot_accs else (0, 0)

        # Probe vs name-only paired bootstrap
        probe_wins = 0
        n_boot = 100
        for _ in range(n_boot):
            idx = rng.choice(len(examples), len(examples), replace=True)
            oob = list(set(range(len(examples))) - set(idx))
            if len(oob) < 5 or len(np.unique(labels[idx])) < len(targets):
                continue
            p1 = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
            p1.fit(X_best_s[idx], labels[idx])
            a1 = p1.score(X_best_s[oob], labels[oob])
            p2 = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
            p2.fit(name_s[idx], labels[idx])
            a2 = p2.score(name_s[oob], labels[oob])
            if a1 > a2:
                probe_wins += 1
        p_value = 1.0 - (probe_wins / n_boot)

        logger.info(
            f"  L{layer}: probe={best_acc:.3f} [{ci_lo:.3f},{ci_hi:.3f}] "
            f"BoW={best_bow:.3f} name={best_name:.3f} "
            f"gap_bow={best_acc - best_bow:.3f} gap_name={best_acc - best_name:.3f} "
            f"p={p_value:.4f} pos={best_pos}"
        )

        results[f"layer_{layer}"] = {
            "probe": float(best_acc), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "bow": float(best_bow), "name_only": float(best_name),
            "gap_bow": float(best_acc - best_bow),
            "gap_name": float(best_acc - best_name),
            "p_value_vs_name": float(p_value),
            "probe_wins": probe_wins, "best_pos": int(best_pos),
        }

    return results, caches


def run_steering(model, examples, caches, layers, targets):
    """Steering with alpha scaling on untyped functions."""
    t2i = {t: i for i, t in enumerate(targets)}
    type_examples = {}
    for e, c in zip(examples, caches):
        type_examples.setdefault(e.target_value, []).append((e, c))

    results = []
    for alpha in [1.0, 3.0, 5.0, 10.0]:
        if "int" not in type_examples or "str" not in type_examples:
            continue
        exs_int = type_examples["int"][:3]
        exs_str = type_examples["str"][:3]

        for layer in layers:
            mean_int = np.mean([c.activations[layer].mean(axis=0) for _, c in exs_int], axis=0)
            mean_str = np.mean([c.activations[layer].mean(axis=0) for _, c in exs_str], axis=0)
            sv = torch.tensor((mean_str - mean_int) * alpha, dtype=torch.float32, device="cuda")
            hook_name = f"blocks.{layer}.hook_resid_post"

            for ex, cache in exs_int[:2]:
                tokens = model.to_tokens(ex.prompt, prepend_bos=True)
                with torch.no_grad():
                    orig = model(tokens)
                op = torch.softmax(orig[0, -1, :], dim=-1)

                def make_hook(s):
                    def h(value, hook):
                        value[0, :, :] += s
                        return value
                    return h

                with torch.no_grad():
                    steered = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_hook(sv))])
                sp = torch.softmax(steered[0, -1, :], dim=-1)
                kl = torch.sum(op * (torch.log(op + 1e-10) - torch.log(sp + 1e-10))).item()

                ot5 = [model.to_string(torch.tensor([t])) for t in torch.topk(op, 5).indices.tolist()]
                st5 = [model.to_string(torch.tensor([t])) for t in torch.topk(sp, 5).indices.tolist()]

                logger.info(f"    alpha={alpha:.0f} L{layer}: KL={kl:.3f} orig={ot5} steered={st5}")
                results.append({"alpha": alpha, "layer": layer, "kl": kl,
                                "sig": ex.metadata["signature"],
                                "orig_top5": ot5, "steered_top5": st5})
    return results


def run_causal_patching(model, device="cuda"):
    """Phase 3: Causal patching on typed contrastive pairs."""
    examples = generate_code_return_dataset(include_untyped=False, include_contrastive=True)
    contrastive = [e for e in examples if e.metadata.get("is_contrastive")]
    pairs_by_id = {}
    for e in contrastive:
        pairs_by_id.setdefault(e.metadata["contrastive_pair"], []).append(e)

    layers = [0, 4, 8, 10, 11]
    caches = extract_activations_batch(model, model.tokenizer, contrastive, layers=layers, device=device)
    cache_by_id = {c.example_id: c for c in caches}

    results = []
    for pair_id, pair_ex in pairs_by_id.items():
        if len(pair_ex) < 2:
            continue
        ex_a, ex_b = pair_ex[0], pair_ex[1]
        ca, cb = cache_by_id.get(ex_a.example_id), cache_by_id.get(ex_b.example_id)
        if ca is None or cb is None:
            continue

        for layer in layers:
            ml = min(ca.activations[layer].shape[0], cb.activations[layer].shape[0])
            diffs = [np.linalg.norm(cb.activations[layer][p] - ca.activations[layer][p]) for p in range(ml)]
            mdp = int(np.argmax(diffs))

            tokens_a = model.to_tokens(ex_a.prompt, prepend_bos=True)
            src = torch.tensor(cb.activations[layer][mdp], dtype=torch.float32, device=device)
            hook = f"blocks.{layer}.hook_resid_post"

            with torch.no_grad():
                ol = model(tokens_a)
            op = torch.softmax(ol[0, -1, :], dim=-1)

            def make_patch(pos, s):
                def h(act, hook):
                    if pos < act.shape[1]:
                        act[0, pos, :] = s
                    return act
                return h

            with torch.no_grad():
                pl = model.run_with_hooks(tokens_a, fwd_hooks=[(hook, make_patch(mdp, src))])
            pp = torch.softmax(pl[0, -1, :], dim=-1)
            nkl = torch.sum(op * (torch.log(op + 1e-10) - torch.log(pp + 1e-10))).item()

            sv = torch.tensor(cb.activations[layer][mdp] - ca.activations[layer][mdp],
                              dtype=torch.float32, device=device)

            def make_steer(s):
                def h(act, hook):
                    act[0, :, :] += s
                    return act
                return h

            with torch.no_grad():
                sl = model.run_with_hooks(tokens_a, fwd_hooks=[(hook, make_steer(sv))])
            sp = torch.softmax(sl[0, -1, :], dim=-1)
            skl = torch.sum(op * (torch.log(op + 1e-10) - torch.log(sp + 1e-10))).item()

            ot5 = [model.to_string(torch.tensor([t])) for t in torch.topk(op, 5).indices.tolist()]
            st5 = [model.to_string(torch.tensor([t])) for t in torch.topk(sp, 5).indices.tolist()]

            results.append({"pair": pair_id, "layer": layer, "necessity_kl": nkl,
                            "sufficiency_kl": skl, "target_a": ex_a.target_value,
                            "target_b": ex_b.target_value, "steered_top5": st5, "orig_top5": ot5})
            logger.info(f"  {pair_id} L{layer}: nec_KL={nkl:.4f} suf_KL={skl:.4f}")

    return results


def run_model_suite(model_name, dtype=torch.float32):
    """Run complete suite for one model."""
    logger.info("=" * 70)
    logger.info(f"MODEL: {model_name}")
    logger.info("=" * 70)

    model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=dtype)
    model.eval()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    logger.info(f"  {n_layers} layers, d_model={d_model}")

    # Select layers
    layers = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]))
    logger.info(f"  Probing layers: {layers}")

    examples = make_examples()
    targets = sorted(set(e.target_value for e in examples))
    t2i = {t: i for i, t in enumerate(targets)}
    labels = np.array([t2i[e.target_value] for e in examples])
    chance = 1.0 / len(targets)

    all_results = {
        "model": model_name, "n_layers": n_layers, "d_model": d_model,
        "n_examples": len(examples), "chance": chance, "targets": targets,
    }

    # --- Behavioral validation ---
    logger.info("\n  === BEHAVIORAL VALIDATION ===")
    beh = run_behavioral_validation(model, examples, max_new_tokens=50)
    beh_sum = compute_behavioral_summary(beh)
    for t, s in beh_sum.items():
        logger.info(f"    {t}: {s['task_accuracy']:.1%} (n={s['n']})")
    all_results["behavioral"] = beh_sum

    type_correct = {}
    for r in beh:
        type_correct.setdefault(r.target_value, []).append(r.task_success)
    for t in sorted(type_correct):
        logger.info(f"    {t}: {np.mean(type_correct[t]):.1%}")
    all_results["behavioral_per_type"] = {t: float(np.mean(v)) for t, v in type_correct.items()}

    # --- Probing suite ---
    logger.info("\n  === PROBING (LBFGS + CIs + baselines) ===")
    probe_results, caches = run_probing_suite(model, examples, labels, targets, layers, model_name, "code_untyped")
    all_results["probing"] = probe_results

    # --- Nonsense names ---
    logger.info("\n  === NONSENSE FUNCTION NAMES ===")
    nonsense = make_nonsense_examples(examples)
    n_labels = np.array([t2i[e.target_value] for e in nonsense])
    n_caches = extract_activations_batch(model, model.tokenizer, nonsense, layers=layers, device="cuda")
    for layer in layers:
        min_seq = min(len(c.token_ids) for c in n_caches)
        best_acc = 0
        for pos in range(min_seq):
            X = np.stack([n_caches[i].activations[layer][pos] for i in range(len(nonsense))])
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            if X_s.shape[1] > 256:
                X_s = PCA(n_components=min(128, X_s.shape[0]-1, X_s.shape[1]), random_state=42).fit_transform(X_s)
            scores = cross_val_score(
                LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
                X_s, n_labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="accuracy",
            )
            best_acc = max(best_acc, scores.mean())
        logger.info(f"  Nonsense L{layer}: {best_acc:.3f} (chance={chance:.3f})")
        all_results[f"nonsense_L{layer}"] = {"probe": float(best_acc)}

    # --- Steering ---
    logger.info("\n  === STEERING (alpha=1,3,5,10) ===")
    steer_layers = [layers[0], layers[len(layers)//2], layers[-1]]
    steer_layers = [l for l in steer_layers if l < n_layers]
    steer_results = run_steering(model, examples, caches, steer_layers, targets)
    all_results["steering"] = steer_results

    # --- Rhyme behavioral ---
    logger.info("\n  === RHYME BEHAVIORAL ===")
    rhyme_ex = generate_rhyme_dataset(n_per_rhyme_set=3, include_controls=False)
    beh_r = run_behavioral_validation(model, rhyme_ex, max_new_tokens=50)
    beh_r_sum = compute_behavioral_summary(beh_r)
    for t, s in beh_r_sum.items():
        logger.info(f"    {t}: {s['task_accuracy']:.1%}")
    all_results["rhyme_behavioral"] = beh_r_sum

    # --- Causal patching (only for gpt2 small to save time) ---
    if model_name == "gpt2":
        logger.info("\n  === CAUSAL PATCHING ===")
        patching = run_causal_patching(model, "cuda")
        all_results["causal_patching"] = patching

        # --- Commitment curves ---
        logger.info("\n  === COMMITMENT CURVES ===")
        config = ProbeConfig(n_folds=3, commitment_threshold=0.5, stability_window=2, random_state=42)
        curve_results = {}
        for layer in layers:
            probes = train_commitment_probes(caches, examples, layer, config)
            if not probes:
                continue
            curves = compute_commitment_curves(caches, examples, layer, probes, config)
            points = find_commitment_points(curves, threshold=0.5, stability_window=2)
            valid = [p for p in points if p.is_valid]
            if valid:
                positions = [p.position for p in valid]
                tokens_before = [p.tokens_before_target for p in valid]
                logger.info(f"  L{layer}: {len(valid)}/{len(points)} commitments, "
                            f"mean_pos={np.mean(positions):.1f}, "
                            f"mean_tokens_before={np.mean(tokens_before):.1f}")
            else:
                logger.info(f"  L{layer}: no valid commitments")
            curve_results[layer] = {
                "n_valid": len(valid), "n_total": len(points),
                "mean_tokens_before": float(np.mean([p.tokens_before_target for p in valid])) if valid else None,
            }
        all_results["commitment_curves"] = curve_results

    # Cleanup
    del model, caches, n_caches
    torch.cuda.empty_cache()

    return all_results


# ================================================================
# MAIN
# ================================================================
def main():
    logger.info("=" * 70)
    logger.info("RQ4 COMPLETE EXPERIMENT SUITE")
    logger.info("=" * 70)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Disk: {os.popen('df -h /workspace').read()}")

    all_results = {}

    # GPT-2 Small (124M, 12 layers)
    all_results["gpt2"] = run_model_suite("gpt2")

    # GPT-2 Medium (345M, 24 layers)
    all_results["gpt2-medium"] = run_model_suite("gpt2-medium")

    # GPT-2 XL (1.5B, 48 layers)
    all_results["gpt2-xl"] = run_model_suite("gpt2-xl")

    # Pythia-2.8B (32 layers) in float16
    try:
        all_results["pythia-2.8b"] = run_model_suite("pythia-2.8b", dtype=torch.float16)
    except Exception as e:
        logger.error(f"Pythia-2.8B failed: {e}")
        all_results["pythia-2.8b"] = {"error": str(e)}

    # Save everything
    os.makedirs("results/lookahead/complete", exist_ok=True)
    with open("results/lookahead/complete/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Also tar everything
    os.system("tar czf /workspace/rq4_complete_results.tar.gz results/lookahead/")

    logger.info("\n" + "=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("Saved to results/lookahead/complete/all_results.json")
    logger.info("Tar at /workspace/rq4_complete_results.tar.gz")
    logger.info("=" * 70)

    # Print summary
    for model_name, res in all_results.items():
        if isinstance(res, dict) and "probing" in res:
            logger.info(f"\n{model_name}:")
            beh = res.get("behavioral", {})
            for t, s in beh.items():
                logger.info(f"  Behavioral: {s.get('task_accuracy', 'N/A')}")
            for lk, lv in res["probing"].items():
                logger.info(
                    f"  {lk}: probe={lv['probe']:.3f} [{lv['ci_lo']:.3f},{lv['ci_hi']:.3f}] "
                    f"BoW={lv['bow']:.3f} name={lv['name_only']:.3f} "
                    f"gap_name={lv['gap_name']:.3f} p={lv['p_value_vs_name']:.4f}"
                )


if __name__ == "__main__":
    main()
