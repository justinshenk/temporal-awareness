#!/usr/bin/env python3
"""run_staircase_v2.py — unified position-baseline staircase pipeline.

One script, all 5 domains, any HF-supported model. Replaces the
workshop's per-domain ad-hoc scripts with a single configurable runner.

USAGE
-----
    python scripts/lookahead/experiments/run_staircase_v2.py \\
        --model EleutherAI/pythia-2.8b-deduped \\
        --domain rhyme \\
        --output_dir results/v2

OUTPUT
------
Each (model, domain) writes a single JSON file under output_dir/:
    {model_slug}__{domain}__staircase.json

containing:
    - meta: model, domain, n_examples, layers tested, predicted_gap_sign
    - per_position: layer → list of {position, cv_accuracy_mean, ...}
    - headlines: list of {layer, resolver, target_acc, max_earlier_acc,
                          headline_gap, n_examples, pre_registration_check}
    - baselines: BoW + mean-pool (workshop legacy) + chance level
    - ablation (if requested): {layer → {zero: {...}, mean: {...}}}

DESIGN
------
* Single forward pass per example extracts every layer's activations.
  Same caches drive linear probes, MLP probes, and ablation baselines.
* Per-position probing comes from src.lookahead.probing.commitment_probes.
* Target positions resolved via tokenizer-aware DomainSpec rules.
* Headline N+P baseline = max accuracy across positions strictly earlier
  than each example's target. Most reviewer-defensible baseline choice.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# CLI + logging setup
# ──────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Position-baseline staircase pipeline (EMNLP version)",
    )
    p.add_argument("--model", required=True,
                   help="HF model id, e.g. 'EleutherAI/pythia-2.8b-deduped' or 'google/gemma-2-9b-it'")
    p.add_argument("--domain", required=True,
                   choices=["code", "rhyme", "qa_suggestive", "qa_neutral", "trivia"],
                   help="Which staircase domain to evaluate")
    p.add_argument("--output_dir", default="results/v2",
                   help="Directory to write the JSON result file")
    p.add_argument("--split", default="test",
                   choices=["train", "test", "all"],
                   help="Dataset split to probe on (default test)")

    # Layer selection
    p.add_argument("--layer_mode", default="workshop_6",
                   choices=["workshop_6", "maar_range", "all", "custom"],
                   help="How to choose layers to probe")
    p.add_argument("--layers", default="",
                   help="Comma-separated layer indices when --layer_mode=custom")

    # Probe options
    p.add_argument("--probe_types", default="linear",
                   help="Comma-separated: linear,mlp")
    p.add_argument("--pca_dim", type=int, default=128,
                   help="PCA reduction before probing (workshop default 128)")
    p.add_argument("--n_folds", type=int, default=5,
                   help="StratifiedKFold splits")

    # Ablation
    p.add_argument("--ablation", default="",
                   help="Comma-separated subset of {zero, mean}; empty = skip ablation")
    p.add_argument("--ablation_layer", type=int, default=-1,
                   help="Layer at which to intervene (default: first sampled layer)")

    # Bootstrap
    p.add_argument("--n_boot", type=int, default=500,
                   help="Bootstrap iterations for headline CI (paired prompt-level resampling)")

    # Quantization for huge models
    p.add_argument("--quantization", default="bf16",
                   choices=["fp32", "bf16", "fp16", "int8", "int4"],
                   help="Load dtype/quantization for the model")
    p.add_argument("--device_map", default="auto",
                   help="HF device_map (auto, balanced, etc.)")

    # Maar data root (rhyme + qa)
    p.add_argument("--maar_data_root", default=None,
                   help="Override MAAR_DATA_ROOT env var if needed")

    # Quick-test option
    p.add_argument("--max_examples", type=int, default=0,
                   help="Cap N examples for quick testing (0 = use all)")

    # Resume
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if output file exists")

    # Seed
    p.add_argument("--seed", type=int, default=42)

    return p


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


logger = logging.getLogger("staircase_v2")


# ──────────────────────────────────────────────────────────────────────
# Dataset loading per domain
# ──────────────────────────────────────────────────────────────────────

def load_dataset_for_domain(domain: str, split: str, maar_root: Optional[str]):
    """Return (examples, n_classes_primary, label_extractor, group_fn).

    label_extractor: a callable(PlanningExample) → str label for the
    primary classification task in this domain.

    group_fn: optional callable(PlanningExample) → group_id. When provided,
    CV uses StratifiedGroupKFold to ensure all examples sharing the same
    group_id stay in the same fold. Needed for qa_neutral where pair
    members share question text — random CV puts opposite-labeled examples
    in train and test, making the task unlearnable.
    """
    from src.lookahead.utils.types import PlanningExample, TaskType

    if domain == "code":
        # Use the clean code_return module directly; the workshop's
        # run_rq4_final.py imports transformer_lens at module load.
        from src.lookahead.datasets.code_return import generate_code_return_dataset
        examples = generate_code_return_dataset(
            include_untyped=True, include_contrastive=True,
        )
        label_fn = lambda ex: ex.metadata.get("return_type", ex.target_value)
        n_classes = len(set(label_fn(e) for e in examples))
        return examples, n_classes, label_fn, None

    if domain == "trivia":
        from src.lookahead.datasets.trivia import load_trivia
        examples = load_trivia(split=split)
        label_fn = lambda ex: ex.metadata["category"]
        return examples, 5, label_fn, None

    if domain == "rhyme":
        from src.lookahead.datasets.maar_data import load_maar_rhyme
        if maar_root:
            os.environ["MAAR_DATA_ROOT"] = maar_root
        examples = load_maar_rhyme(split=split)
        label_fn = lambda ex: ex.metadata["rhyme_family"]
        return examples, 10, label_fn, None

    if domain == "qa_suggestive":
        from src.lookahead.datasets.maar_data import load_maar_qa_suggestive
        if maar_root:
            os.environ["MAAR_DATA_ROOT"] = maar_root
        examples = load_maar_qa_suggestive(split=split)
        label_fn = lambda ex: ex.metadata["article"]  # 2-class
        # Each suggestive question is noun-specific; random CV is appropriate.
        return examples, 2, label_fn, None

    if domain == "qa_neutral":
        from src.lookahead.datasets.maar_data import load_maar_qa_neutral
        if maar_root:
            os.environ["MAAR_DATA_ROOT"] = maar_root
        examples = load_maar_qa_neutral()
        label_fn = lambda ex: ex.metadata["article"]
        # CRITICAL: pair members share question text. Random CV puts the
        # same text with opposite labels in train/test, making the task
        # systematically unlearnable. Group by question text so each pair
        # stays in one fold; probe must then generalize across pairs.
        group_fn = lambda ex: ex.metadata["question"]
        return examples, 2, label_fn, group_fn

    raise ValueError(f"Unknown domain: {domain}")


# ──────────────────────────────────────────────────────────────────────
# Model loading with quantization options
# ──────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id: str, quantization: str, device_map: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_kwargs: dict = {"device_map": device_map, "trust_remote_code": True}
    if quantization == "bf16":
        quant_kwargs["torch_dtype"] = torch.bfloat16
    elif quantization == "fp16":
        quant_kwargs["torch_dtype"] = torch.float16
    elif quantization == "fp32":
        quant_kwargs["torch_dtype"] = torch.float32
    elif quantization in ("int8", "int4"):
        from transformers import BitsAndBytesConfig
        if quantization == "int8":
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

    logger.info(f"Loading model: {model_id}  (quantization={quantization})")
    model = AutoModelForCausalLM.from_pretrained(model_id, **quant_kwargs)
    model.eval()
    return model, tokenizer


def model_slug(model_id: str) -> str:
    """Filesystem-safe identifier from a model id."""
    return model_id.replace("/", "__").replace(":", "_")


# ──────────────────────────────────────────────────────────────────────
# Per-position probing wrapper (uses the workshop's machinery)
# ──────────────────────────────────────────────────────────────────────

def train_per_position(
    caches, examples, label_fn, layer: int,
    pca_dim: int, n_folds: int, seed: int,
    probe_type: str = "linear",
    groups=None,  # array of group_ids (e.g., question text); enables StratifiedGroupKFold
):
    """Probe at every token position (up to min_seq_len) at the given layer.

    Returns: dict[position → {cv_accuracy_mean, cv_accuracy_std, n_samples}]
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import (
        cross_val_score, StratifiedKFold, StratifiedGroupKFold,
    )

    if probe_type == "linear":
        from sklearn.linear_model import LogisticRegression
        make_clf = lambda: LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
    elif probe_type == "mlp":
        from src.lookahead.probing.mlp_probe import MLPProbe
        make_clf = lambda: MLPProbe(random_state=seed)
    else:
        raise ValueError(f"Unknown probe_type {probe_type!r}")

    labels_str = [label_fn(ex) for ex in examples]
    classes = sorted(set(labels_str))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    labels = np.array([cls_to_idx[s] for s in labels_str])

    if len(classes) < 2:
        logger.warning(f"Only {len(classes)} class(es) — skipping layer {layer}")
        return {}, classes

    min_seq = min(len(c.token_ids) for c in caches)

    # Pick the CV splitter
    if groups is not None:
        groups_arr = np.asarray(groups)
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        cv_kwargs = {"groups": groups_arr}
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        cv_kwargs = {}

    out: dict = {}
    for pos in range(min_seq):
        X = np.stack([caches[i].activations[layer][pos] for i in range(len(examples))])
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        if Xs.shape[1] > pca_dim:
            k = min(pca_dim, Xs.shape[0] - 1, Xs.shape[1])
            Xs = PCA(n_components=k, random_state=seed).fit_transform(Xs)
        try:
            scores = cross_val_score(make_clf(), Xs, labels, cv=cv, scoring="accuracy", **cv_kwargs)
        except Exception as e:
            logger.warning(f"  pos={pos} L{layer}: CV failed ({e}); skipping")
            continue
        out[pos] = {
            "cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_std": float(scores.std()),
            "n_samples": int(len(labels)),
        }
    return out, classes


# ──────────────────────────────────────────────────────────────────────
# Target-position accuracy per resolver
# ──────────────────────────────────────────────────────────────────────

def target_position_accuracies(
    caches, examples, label_fn,
    layer: int,
    resolved_positions,            # list[ResolvedPositions]
    pca_dim: int, n_folds: int, seed: int,
    probe_type: str = "linear",
    groups=None,
) -> dict[str, dict]:
    """For each target resolver, build features from each example's resolved
    target position and CV-score a probe on them.

    Returns: dict[resolver_name → {accuracy, n_used, mode_position}]
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import (
        cross_val_score, StratifiedKFold, StratifiedGroupKFold,
    )

    if probe_type == "linear":
        from sklearn.linear_model import LogisticRegression
        make_clf = lambda: LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
    else:
        from src.lookahead.probing.mlp_probe import MLPProbe
        make_clf = lambda: MLPProbe(random_state=seed)

    labels_str = [label_fn(ex) for ex in examples]
    classes = sorted(set(labels_str))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    labels_all = np.array([cls_to_idx[s] for s in labels_str])
    groups_all = np.asarray(groups) if groups is not None else None

    if not resolved_positions:
        return {}
    resolver_names = list(resolved_positions[0].targets_by_resolver.keys())

    if groups_all is not None:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    out: dict[str, dict] = {}
    for resolver_name in resolver_names:
        rows = []
        rlbls = []
        rgroups = []
        positions_used = []
        for i, rp in enumerate(resolved_positions):
            idx = rp.targets_by_resolver.get(resolver_name)
            if idx is None or idx >= len(caches[i].token_ids):
                continue
            rows.append(caches[i].activations[layer][idx])
            rlbls.append(labels_all[i])
            if groups_all is not None:
                rgroups.append(groups_all[i])
            positions_used.append(idx)

        if len(rows) < 10:
            continue
        X = np.stack(rows)
        y = np.array(rlbls)
        if len(np.unique(y)) < 2:
            continue

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        if Xs.shape[1] > pca_dim:
            k = min(pca_dim, Xs.shape[0] - 1, Xs.shape[1])
            Xs = PCA(n_components=k, random_state=seed).fit_transform(Xs)
        try:
            if groups_all is not None:
                scores = cross_val_score(make_clf(), Xs, y, cv=cv,
                                          groups=np.asarray(rgroups), scoring="accuracy")
            else:
                scores = cross_val_score(make_clf(), Xs, y, cv=cv, scoring="accuracy")
            acc = float(scores.mean())
        except Exception as e:
            logger.warning(f"  resolver={resolver_name} L{layer}: CV failed ({e})")
            continue

        from collections import Counter
        mode_pos = Counter(positions_used).most_common(1)[0][0] if positions_used else None

        out[resolver_name] = {
            "accuracy": acc,
            "n_used": len(rows),
            "mode_position": int(mode_pos) if mode_pos is not None else None,
        }
    return out


# ──────────────────────────────────────────────────────────────────────
# Bootstrap CI for the headline gap (paired prompt-level resampling)
# ──────────────────────────────────────────────────────────────────────

def bootstrap_headline_ci(
    caches, examples, label_fn,
    layer: int,
    resolved_positions,
    resolver_name: str,
    earlier_position_global: int,
    pca_dim: int,
    n_boot: int = 1000,
    seed: int = 42,
    probe_type: str = "linear",
) -> dict:
    """Paired bootstrap CI for the (target_accuracy − earlier_accuracy) gap.

    The same bootstrap-sample indices are used for both target-position
    features and earlier-position features, so the per-iteration gap is a
    meaningful paired statistic. Percentile-method 95% CI.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    if probe_type == "linear":
        from sklearn.linear_model import LogisticRegression
        make_clf = lambda: LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
    else:
        from src.lookahead.probing.mlp_probe import MLPProbe
        make_clf = lambda: MLPProbe(random_state=seed)

    labels_str = [label_fn(ex) for ex in examples]
    classes = sorted(set(labels_str))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    labels_all = np.array([cls_to_idx[s] for s in labels_str])

    # Build target-position feature matrix (skip examples where resolver fails)
    valid_indices = []
    X_target_rows = []
    for i, rp in enumerate(resolved_positions):
        idx = rp.targets_by_resolver.get(resolver_name)
        if idx is None or idx >= len(caches[i].token_ids):
            continue
        valid_indices.append(i)
        X_target_rows.append(caches[i].activations[layer][idx])

    if len(valid_indices) < 20:
        return {"available": False, "reason": "too few valid examples"}

    X_target = np.stack(X_target_rows)
    # Earlier-position features at the chosen global earlier position
    X_earlier_rows = []
    for i in valid_indices:
        seq_len = len(caches[i].token_ids)
        p = min(earlier_position_global, seq_len - 1)
        X_earlier_rows.append(caches[i].activations[layer][p])
    X_earlier = np.stack(X_earlier_rows)
    y = labels_all[valid_indices]

    # Fit one global PCA per feature matrix (computed ONCE outside the
    # bootstrap loop — same dim reduction applied to all bootstrap samples
    # is the standard practice and matches the workshop pipeline).
    def reduce(X):
        Xs = StandardScaler().fit_transform(X)
        if Xs.shape[1] > pca_dim:
            k = min(pca_dim, Xs.shape[0] - 1, Xs.shape[1])
            Xs = PCA(n_components=k, random_state=seed).fit_transform(Xs)
        return Xs

    X_tgt_red = reduce(X_target)
    X_ear_red = reduce(X_earlier)

    rng = np.random.RandomState(seed)
    n = len(y)
    boot_target, boot_earlier, boot_gap = [], [], []
    valid_boots = 0
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        oob = list(set(range(n)) - set(idx))
        if len(oob) < 5 or len(np.unique(y[idx])) < len(np.unique(y)):
            continue
        valid_boots += 1
        clf_t = make_clf().fit(X_tgt_red[idx], y[idx])
        a_t = clf_t.score(X_tgt_red[oob], y[oob])
        clf_e = make_clf().fit(X_ear_red[idx], y[idx])
        a_e = clf_e.score(X_ear_red[oob], y[oob])
        boot_target.append(a_t)
        boot_earlier.append(a_e)
        boot_gap.append(a_t - a_e)

    if valid_boots < 10:
        return {"available": False, "reason": f"only {valid_boots} valid bootstrap samples"}

    arr_g = np.array(boot_gap)
    lo, hi = np.percentile(arr_g, [2.5, 97.5])
    p_gap_positive = float(np.mean(arr_g > 0))
    return {
        "available": True,
        "valid_boots": valid_boots,
        "earlier_position_global": int(earlier_position_global),
        "target_mean": float(np.mean(boot_target)),
        "target_ci": [float(np.percentile(boot_target, 2.5)),
                      float(np.percentile(boot_target, 97.5))],
        "earlier_mean": float(np.mean(boot_earlier)),
        "earlier_ci": [float(np.percentile(boot_earlier, 2.5)),
                       float(np.percentile(boot_earlier, 97.5))],
        "gap_mean": float(np.mean(arr_g)),
        "gap_ci": [float(lo), float(hi)],
        "p_gap_positive": p_gap_positive,
    }


# ──────────────────────────────────────────────────────────────────────
# Bag-of-words baseline (simple, fast, lower-bound diagnostic)
# ──────────────────────────────────────────────────────────────────────

def bow_baseline_accuracy(caches, examples, label_fn, n_folds: int, seed: int) -> float:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    labels_str = [label_fn(ex) for ex in examples]
    classes = sorted(set(labels_str))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_to_idx[s] for s in labels_str])

    max_token_id = max(max(c.token_ids) for c in caches) + 1
    bow_dim = min(max_token_id, 200_000)  # generous cap

    X_bow = np.zeros((len(caches), bow_dim), dtype=np.float32)
    for r, c in enumerate(caches):
        for tid in c.token_ids:
            if tid < bow_dim:
                X_bow[r, tid] = 1.0
    nz = X_bow.sum(axis=0) > 0
    X_bow = X_bow[:, nz]
    if X_bow.shape[1] == 0:
        return float(1.0 / len(classes))

    Xs = StandardScaler(with_mean=False).fit_transform(X_bow)  # sparse-friendly
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(
        LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed),
        Xs.toarray() if hasattr(Xs, "toarray") else Xs,
        y, cv=cv, scoring="accuracy",
    )
    return float(scores.mean())


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def run(args) -> dict:
    # Late imports so --help is fast and doesn't need torch
    import torch
    from src.lookahead.probing.hf_activation_extraction import (
        extract_activations_batch, default_layer_sample, maar_layer_sample,
        find_transformer_blocks,
    )
    from src.lookahead.probing.staircase_headline import (
        resolve_positions_for_caches, compute_headlines, gap_sign_matches_prediction,
    )
    from src.lookahead.domains import get_domain

    spec = get_domain(args.domain)
    examples, n_classes, label_fn, group_fn = load_dataset_for_domain(
        args.domain, args.split, args.maar_data_root,
    )
    if args.max_examples and args.max_examples < len(examples):
        examples = examples[: args.max_examples]
    # Compute groups (if any) after example truncation so indexing matches.
    groups = [group_fn(ex) for ex in examples] if group_fn is not None else None
    logger.info(f"Domain={args.domain}  n_examples={len(examples)}  n_classes={n_classes}  "
                f"predicted_gap={spec.predicted_gap.value}")
    if groups is not None:
        n_groups = len(set(groups))
        logger.info(f"  Using StratifiedGroupKFold with {n_groups} unique groups "
                    f"(pair-stratified CV)")

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model,
        quantization=args.quantization,
        device_map=args.device_map,
    )

    # Decide which layers to probe
    _, blocks = find_transformer_blocks(model)
    n_layers = len(blocks)
    if args.layer_mode == "workshop_6":
        layers = default_layer_sample(n_layers, n_samples=6)
    elif args.layer_mode == "maar_range":
        layers = maar_layer_sample(n_layers)
    elif args.layer_mode == "all":
        layers = list(range(n_layers))
    else:  # custom
        layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
    logger.info(f"Model n_layers={n_layers}; probing layers={layers}")

    # Extract activations
    t0 = time.time()
    caches = extract_activations_batch(
        model=model, tokenizer=tokenizer, examples=examples,
        layers=layers, show_progress=True,
    )
    logger.info(f"Activation extraction: {time.time() - t0:.1f}s")

    # Resolve positions
    resolved = resolve_positions_for_caches(spec, caches, examples, tokenizer)

    # BoW baseline (chance lower bound)
    t0 = time.time()
    bow_acc = bow_baseline_accuracy(caches, examples, label_fn, args.n_folds, args.seed)
    logger.info(f"BoW baseline accuracy: {bow_acc:.3f}   (chance={1.0/n_classes:.3f})  "
                f"[{time.time()-t0:.1f}s]")

    # Probe every layer + position, for each requested probe type
    probe_types = [s.strip() for s in args.probe_types.split(",") if s.strip()]

    per_layer_results: dict[str, dict] = {}
    headlines_all: list[dict] = []

    for probe_type in probe_types:
        logger.info(f"=== Probe type: {probe_type} ===")
        for layer in layers:
            t_layer = time.time()
            per_pos, classes = train_per_position(
                caches, examples, label_fn, layer,
                pca_dim=args.pca_dim, n_folds=args.n_folds, seed=args.seed,
                probe_type=probe_type,
                groups=groups,
            )
            if not per_pos:
                continue

            target_accs = target_position_accuracies(
                caches, examples, label_fn,
                layer=layer, resolved_positions=resolved,
                pca_dim=args.pca_dim, n_folds=args.n_folds, seed=args.seed,
                probe_type=probe_type,
                groups=groups,
            )

            # Headline computation
            target_acc_simple = {k: v["accuracy"] for k, v in target_accs.items()}
            headlines = compute_headlines(
                spec=spec, layer=layer,
                per_position_results=per_pos,
                resolved=resolved,
                target_position_accuracies=target_acc_simple,
            )

            for h in headlines:
                check = gap_sign_matches_prediction(
                    observed_gap_pp=100.0 * h.headline_gap,
                    predicted=spec.predicted_gap,
                )
                row = h.to_dict()
                row["probe_type"] = probe_type
                row["pre_registration_check"] = check
                headlines_all.append(row)
                logger.info(
                    f"  L{layer:3d} [{probe_type:6s}] {h.resolver_name:30s}  "
                    f"target={h.target_accuracy:.3f}  max_earlier={h.max_earlier_accuracy:.3f}  "
                    f"gap={h.headline_gap*100:+.1f}pp  "
                    f"pred={spec.predicted_gap.value} obs={check['observed_sign']}  "
                    f"{'✓' if check['matches'] else '✗'}"
                )

            # Bootstrap CI for the strongest-headline (largest |gap|) at this layer.
            # Limiting to one bootstrap per (probe_type, layer) keeps total
            # runtime within ~10 min per (model, domain). The strongest
            # headline is the most informative for the paper table.
            if headlines:
                strongest = max(headlines, key=lambda h: abs(h.headline_gap))
                if strongest.max_earlier_position_mode is not None:
                    t_boot = time.time()
                    ci = bootstrap_headline_ci(
                        caches, examples, label_fn,
                        layer=layer, resolved_positions=resolved,
                        resolver_name=strongest.resolver_name,
                        earlier_position_global=strongest.max_earlier_position_mode,
                        pca_dim=args.pca_dim,
                        n_boot=args.n_boot,
                        seed=args.seed,
                        probe_type=probe_type,
                    )
                    # Attach CI to that row in headlines_all
                    for r in headlines_all:
                        if (r["layer"] == strongest.layer
                                and r["resolver"] == strongest.resolver_name
                                and r["probe_type"] == probe_type):
                            r["bootstrap_ci"] = ci
                            break
                    if ci.get("available"):
                        logger.info(
                            f"     ↳ bootstrap gap={ci['gap_mean']*100:+.1f}pp "
                            f"CI[{ci['gap_ci'][0]*100:+.1f}, {ci['gap_ci'][1]*100:+.1f}]pp  "
                            f"P(gap>0)={ci['p_gap_positive']:.3f}  "
                            f"[{time.time()-t_boot:.0f}s, {ci['valid_boots']} boots]"
                        )

            key = f"{probe_type}__L{layer}"
            per_layer_results[key] = {
                "probe_type": probe_type,
                "layer": layer,
                "per_position": per_pos,
                "target_resolver_accuracies": target_accs,
                "elapsed_seconds": time.time() - t_layer,
            }

    # ──────────────────────────────────────────────────────────────
    # Optional: causal ablation on the single best (layer, resolver)
    # ──────────────────────────────────────────────────────────────
    ablation_results: dict = {}
    ablation_modes = [m.strip() for m in args.ablation.split(",") if m.strip()]
    if ablation_modes and headlines_all:
        from src.lookahead.probing.np_ablation import run_np_ablation_experiment
        from src.lookahead.domains import get_earlier_positions
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        # Pick the strongest headline across all probe_types and layers.
        # That's the row most likely to make it into the paper, so it's
        # the one we want causal evidence for.
        best = max(headlines_all, key=lambda r: abs(r["headline_gap"]))
        best_layer = best["layer"]
        best_resolver = best["resolver"]
        intervention_layer = args.ablation_layer if args.ablation_layer >= 0 else layers[0]
        logger.info(
            f"=== Ablation: intervene@L{intervention_layer} "
            f"record@L{best_layer} resolver={best_resolver} "
            f"modes={ablation_modes} ==="
        )

        # Earlier positions per example for that resolver
        earlier_per_ex: list[list[int]] = []
        valid_mask: list[bool] = []
        for i, rp in enumerate(resolved):
            tgt = rp.targets_by_resolver.get(best_resolver)
            if tgt is None:
                earlier_per_ex.append([])
                valid_mask.append(False)
                continue
            ep = get_earlier_positions(
                spec=spec,
                target_position=tgt,
                n_tokens=rp.n_tokens,
                signature_end=rp.signature_end,
            )
            earlier_per_ex.append(ep)
            valid_mask.append(len(ep) > 0)

        n_valid = sum(valid_mask)
        if n_valid < 10:
            logger.warning("Too few valid examples for ablation; skipping.")
        else:
            # Filter to valid examples for both the call AND the probe
            valid_idx = [i for i, v in enumerate(valid_mask) if v]
            valid_examples = [examples[i] for i in valid_idx]
            valid_earlier = [earlier_per_ex[i] for i in valid_idx]

            try:
                t_abl = time.time()
                ablated = run_np_ablation_experiment(
                    model=model, tokenizer=tokenizer,
                    examples=valid_examples,
                    baseline_caches=[caches[i] for i in valid_idx],
                    earlier_positions_per_example=valid_earlier,
                    intervention_layer=intervention_layer,
                    record_layer=best_layer,
                    modes=tuple(ablation_modes),
                )
                logger.info(f"  Ablation forward passes: {time.time()-t_abl:.0f}s")

                # Train a probe on the ablated target-position activations
                # vs the BASELINE accuracy (already in target_resolver_accuracies).
                labels_str = [label_fn(ex) for ex in valid_examples]
                classes_local = sorted(set(labels_str))
                cls_to_idx = {c: i for i, c in enumerate(classes_local)}
                y_local = np.array([cls_to_idx[s] for s in labels_str])
                cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

                for mode in ablation_modes:
                    rows = []
                    for rec, ex in zip(ablated[mode], valid_examples):
                        # Find target position in the ablated cache (same prompt → same tokens)
                        tok_strs = rec["token_strings"]
                        # Re-resolve the target position in the ablated cache
                        from src.lookahead.domains import DOMAINS
                        sp = DOMAINS[args.domain]
                        res = {r.name: r.find(tok_strs, rec["token_ids"], tokenizer)
                               for r in sp.target_position_resolvers}
                        tgt = res.get(best_resolver)
                        if tgt is None or tgt >= len(tok_strs):
                            rows.append(None)
                            continue
                        rows.append(rec["layer_activation_after_ablation"][tgt])

                    kept = [(i, r) for i, r in enumerate(rows) if r is not None]
                    if len(kept) < 10:
                        continue
                    idxs, feats = zip(*kept)
                    X = np.stack(feats)
                    y_kept = y_local[list(idxs)]
                    Xs = StandardScaler().fit_transform(X)
                    if Xs.shape[1] > args.pca_dim:
                        k = min(args.pca_dim, Xs.shape[0] - 1, Xs.shape[1])
                        Xs = PCA(n_components=k, random_state=args.seed).fit_transform(Xs)
                    scores = cross_val_score(
                        LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                           random_state=args.seed),
                        Xs, y_kept, cv=cv, scoring="accuracy",
                    )
                    ablation_results[mode] = {
                        "intervention_layer": intervention_layer,
                        "record_layer": best_layer,
                        "resolver": best_resolver,
                        "ablated_accuracy": float(scores.mean()),
                        "ablated_accuracy_std": float(scores.std()),
                        "baseline_target_accuracy": float(best["target_accuracy"]),
                        "drop_pp": float(100.0 * (best["target_accuracy"] - scores.mean())),
                        "n_examples": int(len(kept)),
                    }
                    logger.info(
                        f"  ablation={mode:5s}: baseline={best['target_accuracy']:.3f}  "
                        f"ablated={scores.mean():.3f}  drop={(best['target_accuracy']-scores.mean())*100:+.1f}pp"
                    )
            except Exception as e:
                logger.warning(f"Ablation failed: {e}")
                traceback.print_exc()

    # Build result document
    out_doc = {
        "meta": {
            "model": args.model,
            "domain": args.domain,
            "split": args.split,
            "n_examples": len(examples),
            "n_classes": n_classes,
            "n_layers_model": n_layers,
            "layers_probed": layers,
            "probe_types": probe_types,
            "predicted_gap": spec.predicted_gap.value,
            "domain_notes": spec.notes,
            "quantization": args.quantization,
            "seed": args.seed,
        },
        "baselines": {
            "chance": 1.0 / n_classes,
            "bag_of_words_accuracy": bow_acc,
        },
        "per_layer": per_layer_results,
        "headlines": headlines_all,
        "ablation": ablation_results,
    }
    return out_doc


# ──────────────────────────────────────────────────────────────────────
# CLI entry
# ──────────────────────────────────────────────────────────────────────

def main():
    args = build_argparser().parse_args()
    setup_logging("INFO")
    sys.path.insert(0, os.getcwd())  # so `src.lookahead` and `scripts...` import

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_slug(args.model)}__{args.domain}__staircase.json"

    if out_path.exists() and not args.overwrite:
        logger.info(f"Output exists, skipping (use --overwrite): {out_path}")
        return 0

    t0 = time.time()
    try:
        doc = run(args)
    except Exception:
        traceback.print_exc()
        return 1
    doc["meta"]["total_seconds"] = round(time.time() - t0, 2)

    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2, default=str)
    logger.info(f"Wrote {out_path}  ({doc['meta']['total_seconds']}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
