#!/usr/bin/env python3
"""patch_meanpool_baseline.py — backfill workshop's mean-pool baseline onto existing JSONs.

For each results/v2/*__staircase.json:
  1. Load model + tokenizer + extract activations on the same examples
  2. Compute mean-pool baseline at each layer present in the JSON
  3. Compute group-aware BoW (if domain has groups, e.g., qa_neutral)
  4. Append fields:
       - baselines.mean_pool_accuracy[layer] = float
       - baselines.bag_of_words_accuracy_grouped = float (qa_neutral only)
       - each headlines[i] gets:
            target_vs_mean_pool_gap = float  (this is the workshop's gap)
            target_vs_max_earlier_gap = headline_gap  (already there)

This is much cheaper than a full rerun (no per-position probing) — ~2-5 min per JSON.

Usage:
    python3 patch_meanpool_baseline.py --results_dir results/v2
    python3 patch_meanpool_baseline.py --results_dir results/v2 --models gemma  # filter
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# CRITICAL: add cwd to sys.path so `from src.lookahead...` works.
# When this script runs via `python3 scripts/.../patch_meanpool_baseline.py`,
# Python adds the SCRIPT's directory to sys.path, NOT the cwd. The src/ package
# lives in cwd (/workspace/temporal-awareness), so we add it explicitly here,
# before any imports of src.* below.
sys.path.insert(0, os.getcwd())

import numpy as np
import torch

logger = logging.getLogger("backfill")


def setup_logging(level="INFO"):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ──────────────────────────────────────────────────────────────────────
# Domain & dataset reconstruction (mirror run_staircase_v2's loader)
# ──────────────────────────────────────────────────────────────────────
def load_dataset_for_domain(domain: str, split: str, maar_root: str | None):
    """Same as runner's loader — returns (examples, n_classes, label_fn, group_fn)."""
    import os
    if maar_root:
        os.environ["MAAR_DATA_ROOT"] = maar_root

    if domain == "code":
        from src.lookahead.datasets.code_untyped import load_code_untyped
        examples = load_code_untyped()
        label_fn = lambda ex: ex.metadata.get("return_type", ex.target_value)
        n = len(set(label_fn(e) for e in examples))
        return examples, n, label_fn, None

    if domain == "trivia":
        from src.lookahead.datasets.trivia import load_trivia
        examples = load_trivia(split=split)
        return examples, 5, (lambda ex: ex.metadata["category"]), None

    if domain == "rhyme":
        from src.lookahead.datasets.maar_data import load_maar_rhyme
        examples = load_maar_rhyme(split=split)
        return examples, 10, (lambda ex: ex.metadata["rhyme_family"]), None

    if domain == "qa_suggestive":
        from src.lookahead.datasets.maar_data import load_maar_qa_suggestive
        examples = load_maar_qa_suggestive(split=split)
        return examples, 2, (lambda ex: ex.metadata["article"]), None

    if domain == "qa_neutral":
        from src.lookahead.datasets.maar_data import load_maar_qa_neutral
        examples = load_maar_qa_neutral()
        return (
            examples, 2,
            (lambda ex: ex.metadata["article"]),
            (lambda ex: ex.metadata["question"]),  # group by question text
        )

    raise ValueError(f"Unknown domain: {domain}")


# ──────────────────────────────────────────────────────────────────────
# Mean-pool baseline (workshop-style)
# ──────────────────────────────────────────────────────────────────────
def mean_pool_baseline(
    caches, examples, label_fn,
    layer: int, pool_positions: int,
    pca_dim: int = 128, n_folds: int = 5, seed: int = 0,
    groups=None,
) -> float:
    """mean-pool first N tokens → standardize → PCA → LR → CV accuracy."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        cross_val_score, StratifiedKFold, StratifiedGroupKFold,
    )

    labels_str = [label_fn(ex) for ex in examples]
    classes = sorted(set(labels_str))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_to_idx[s] for s in labels_str])
    if len(classes) < 2:
        return float("nan")

    feats = []
    for c in caches:
        n_pool = min(pool_positions, len(c.token_ids))
        feats.append(c.activations[layer][:n_pool].mean(axis=0))
    X = np.stack(feats)

    Xs = StandardScaler().fit_transform(X)
    if Xs.shape[1] > pca_dim:
        k = min(pca_dim, Xs.shape[0] - 1, Xs.shape[1])
        Xs = PCA(n_components=k, random_state=seed).fit_transform(Xs)

    make_clf = lambda: LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(make_clf(), Xs, y, cv=cv, groups=np.asarray(groups), scoring="accuracy")
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(make_clf(), Xs, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


# ──────────────────────────────────────────────────────────────────────
# Grouped BoW
# ──────────────────────────────────────────────────────────────────────
def bow_grouped(examples, label_fn, groups, seed=0, n_folds=5) -> float:
    """BoW baseline using StratifiedGroupKFold (for qa_neutral)."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedGroupKFold

    texts = [ex.prompt for ex in examples]
    labels_str = [label_fn(ex) for ex in examples]
    classes = sorted(set(labels_str))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_to_idx[s] for s in labels_str])
    if len(classes) < 2:
        return float("nan")

    X = CountVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(texts)
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, groups=np.asarray(groups), scoring="accuracy")
    return float(scores.mean())


# ──────────────────────────────────────────────────────────────────────
# Main backfill loop
# ──────────────────────────────────────────────────────────────────────
def backfill_one(json_path: Path, maar_root: str, force: bool) -> bool:
    """Patch one JSON. Returns True if successful."""
    doc = json.load(open(json_path))
    if not force and "mean_pool_accuracy" in doc.get("baselines", {}):
        logger.info(f"  SKIP (already patched): {json_path.name}")
        return True

    meta = doc.get("meta", {})
    model_id = meta.get("model")
    domain = meta.get("domain")
    if not (model_id and domain):
        logger.warning(f"  {json_path.name}: missing meta.model or meta.domain")
        return False

    # Layers present in the JSON's headlines
    layers = sorted({h["layer"] for h in doc.get("headlines", [])})
    if not layers:
        logger.warning(f"  {json_path.name}: no headlines, skipping")
        return False

    logger.info(f"  {model_id} × {domain}  ({len(layers)} layers)")

    # Reconstruct examples
    examples, n_classes, label_fn, group_fn = load_dataset_for_domain(
        domain, split=meta.get("split", "all"), maar_root=maar_root,
    )
    if meta.get("max_examples") and meta["max_examples"] < len(examples):
        examples = examples[: meta["max_examples"]]

    groups = [group_fn(ex) for ex in examples] if group_fn is not None else None

    # Load model + tokenizer, extract activations
    sys.path.insert(0, ".")
    from src.lookahead.probing.hf_activation_extraction import (
        load_model_for_extraction, extract_activations_hf,
    )

    t0 = time.time()
    tokenizer, model = load_model_for_extraction(model_id, quantization="bf16")
    caches = extract_activations_hf(
        model=model, tokenizer=tokenizer, examples=examples,
        layers=layers, include_attention=False,
    )
    logger.info(f"    extract: {time.time()-t0:.0f}s")

    # Pool positions: 10 for code (signature length), or shorter for short prompts
    pool_positions = 10
    min_seq = min(len(c.token_ids) for c in caches)
    pool_positions = min(pool_positions, min_seq)

    # Compute mean-pool baseline at each layer
    t1 = time.time()
    mp_results = {}
    for layer in layers:
        acc = mean_pool_baseline(
            caches, examples, label_fn,
            layer=layer, pool_positions=pool_positions,
            pca_dim=128, n_folds=5, seed=0,
            groups=groups,
        )
        mp_results[str(layer)] = float(acc)
    logger.info(f"    mean-pool: {time.time()-t1:.0f}s  range=[{min(mp_results.values()):.3f}, {max(mp_results.values()):.3f}]")

    # Write back
    doc.setdefault("baselines", {})
    doc["baselines"]["mean_pool_accuracy"] = mp_results
    doc["baselines"]["mean_pool_positions"] = pool_positions

    # Grouped BoW (qa_neutral only)
    if groups is not None:
        try:
            bow_g = bow_grouped(examples, label_fn, groups)
            doc["baselines"]["bag_of_words_accuracy_grouped"] = float(bow_g)
            logger.info(f"    grouped BoW: {bow_g:.3f}")
        except Exception as e:
            logger.warning(f"    grouped BoW failed: {e}")

    # Add target_vs_mean_pool_gap on each headline
    for h in doc["headlines"]:
        mp = mp_results.get(str(h["layer"]))
        if mp is not None:
            h["target_vs_mean_pool_gap"] = float(h["target_accuracy"] - mp)

    # Write back (atomic via temp file)
    tmp = json_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2, default=str)
    tmp.replace(json_path)
    logger.info(f"    ✓ wrote back to {json_path.name}")

    # Free GPU
    del model, tokenizer, caches
    torch.cuda.empty_cache()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/v2")
    ap.add_argument("--maar_data_root", default="data/maar_supplementary_material")
    ap.add_argument("--models", default=None, help="optional substring filter (e.g. 'gemma')")
    ap.add_argument("--domains", default=None, help="optional comma-separated list (e.g. 'rhyme,qa_neutral')")
    ap.add_argument("--force", action="store_true", help="re-patch even if mean_pool already present")
    args = ap.parse_args()

    setup_logging("INFO")
    results = sorted(Path(args.results_dir).glob("*__staircase.json"))

    if args.models:
        results = [r for r in results if args.models in r.name]
    if args.domains:
        wanted = set(args.domains.split(","))
        results = [r for r in results
                   if any(f"__{d}__" in r.name for d in wanted)]

    logger.info(f"Backfilling {len(results)} JSON(s)")
    n_ok, n_fail = 0, 0
    t0 = time.time()
    for r in results:
        try:
            ok = backfill_one(r, args.maar_data_root, args.force)
            n_ok += int(ok); n_fail += int(not ok)
        except Exception as e:
            logger.error(f"  {r.name}: {e}", exc_info=True)
            n_fail += 1
    logger.info(f"Done in {time.time()-t0:.0f}s.  OK={n_ok}  FAIL={n_fail}")


if __name__ == "__main__":
    main()
