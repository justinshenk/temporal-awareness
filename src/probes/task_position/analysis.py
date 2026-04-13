"""Analyses A1 / A2 / A4 for task-position probes.

A1: orthogonality between task-position probe directions and a raw-token-position
    probe direction at the same layer. Tests whether task-position probes encode
    a new signal vs. just renaming positional encoding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.probes.task_position.probes import RidgeProbe


def _stack_activations(traces: list, layer: int, trace_ids: set) -> np.ndarray:
    parts = []
    for t in traces:
        if t["trace_id"] not in trace_ids:
            continue
        parts.append(t["activations"][layer].numpy().astype(np.float32))
    return np.vstack(parts)


def _stack_target(traces: list, target: str, trace_ids: set) -> np.ndarray:
    parts = []
    for t in traces:
        if t["trace_id"] not in trace_ids:
            continue
        labels = t["labels"][target]
        if target == "tokens_until_boundary":
            y = np.log1p(np.asarray(labels, dtype=np.float64))
        else:
            y = np.asarray(labels, dtype=np.float64)
        parts.append(y)
    return np.concatenate(parts)


def _stack_raw_position(traces: list, layer: int, trace_ids: set) -> np.ndarray:
    parts = []
    for t in traces:
        if t["trace_id"] not in trace_ids:
            continue
        n = t["activations"][layer].shape[0]
        parts.append(np.arange(n, dtype=np.float64))
    return np.concatenate(parts)


def analysis_a1_orthogonality(
    traces: list,
    layers: list[int],
    train_ids: set,
    targets: list[str],
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Cosine similarity between task-position and raw-position probe directions.

    For each layer:
      1. Fit a ridge probe on the layer's activations to predict raw_token_position.
      2. For each target, fit a ridge probe to predict the target.
      3. Compute the cosine similarity between the two unit-normalized direction
         vectors.

    Returns one row per (layer, target).
    """
    rows = []
    for layer in layers:
        X_train = _stack_activations(traces, layer, train_ids)
        y_raw = _stack_raw_position(traces, layer, train_ids)

        raw_probe = RidgeProbe(alpha=alpha).fit(X_train, y_raw)
        raw_dir = raw_probe.direction()
        raw_norm = float(np.linalg.norm(raw_dir))
        raw_unit = raw_dir / (raw_norm + 1e-12)

        for target in targets:
            y_target = _stack_target(traces, target, train_ids)
            probe = RidgeProbe(alpha=alpha).fit(X_train, y_target)
            d = probe.direction()
            d_norm = float(np.linalg.norm(d))
            d_unit = d / (d_norm + 1e-12)
            cos = float(np.dot(d_unit, raw_unit))
            rows.append(
                {
                    "layer": layer,
                    "target": target,
                    "cosine_with_raw_pos": cos,
                    "raw_dir_norm": raw_norm,
                    "target_dir_norm": d_norm,
                }
            )
    return pd.DataFrame(rows)


def _prediction_site(case_index: int, case_boundaries: list[int], n_tokens: int) -> int:
    """Token index of the last token of the given case's prompt."""
    if case_index + 1 < len(case_boundaries):
        return case_boundaries[case_index + 1] - 1
    return n_tokens - 1


def analysis_a2_failure_prediction(
    traces: list,
    layers: list[int],
    train_ids: set,
    test_ids: set,
    correctness_path: Path | str,
) -> pd.DataFrame:
    """Predict upcoming case correctness from residuals + task-position features.

    Baseline: residual stream activation alone (logistic regression).
    Treatment: residual stream + [task_index, within_task_fraction,
    log1p(tokens_until_boundary)] at the case's prediction site.

    Returns one row per layer with baseline_auc, with_task_position_auc, delta.
    """
    if not Path(correctness_path).exists():
        return pd.DataFrame(
            [{"status": "correctness file missing — rerun extraction with --eval-correctness"}]
        )

    with open(correctness_path) as f:
        correctness_by_trace = json.load(f)

    def gather(trace_ids: set, layer: int):
        Xs, ys, tps = [], [], []
        for t in traces:
            if t["trace_id"] not in trace_ids:
                continue
            cr = correctness_by_trace.get(str(t["trace_id"]), [])
            if not cr:
                continue
            boundaries = t["case_boundaries"]
            act = t["activations"][layer].numpy().astype(np.float32)
            labels = t["labels"]
            n_tokens = act.shape[0]
            for case in cr:
                ci = case["case_index"]
                pos = _prediction_site(ci, boundaries, n_tokens)
                Xs.append(act[pos])
                ys.append(1 if case["correct"] else 0)
                tps.append(
                    [
                        labels["task_index"][pos],
                        labels["within_task_fraction"][pos],
                        float(np.log1p(labels["tokens_until_boundary"][pos])),
                    ]
                )
        if not Xs:
            return None
        return np.array(Xs), np.array(ys), np.array(tps)

    rows = []
    for layer in layers:
        train = gather(train_ids, layer)
        test = gather(test_ids, layer)
        if train is None or test is None:
            rows.append({"layer": layer, "status": "insufficient data"})
            continue
        X_tr, y_tr, tp_tr = train
        X_te, y_te, tp_te = test
        if len(np.unique(y_te)) < 2:
            rows.append(
                {"layer": layer, "status": "test set has only one class"}
            )
            continue

        base = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
        base_auc = float(
            roc_auc_score(y_te, base.predict_proba(X_te)[:, 1])
        )

        X_tr_aug = np.hstack([X_tr, tp_tr])
        X_te_aug = np.hstack([X_te, tp_te])
        aug = LogisticRegression(max_iter=2000).fit(X_tr_aug, y_tr)
        aug_auc = float(
            roc_auc_score(y_te, aug.predict_proba(X_te_aug)[:, 1])
        )

        rows.append(
            {
                "layer": layer,
                "baseline_auc": base_auc,
                "with_task_position_auc": aug_auc,
                "delta": aug_auc - base_auc,
                "n_train": int(X_tr.shape[0]),
                "n_test": int(X_te.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def analysis_a4_calibration_gap(
    traces: list,
    layer: int,
    train_ids: set,
    test_ids: set,
    correctness_path: Path | str,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Calibration gap binned by within_task_fraction readout vs raw token position.

    Trains a within_task_fraction probe on the chosen layer using TRAIN traces,
    then for each test-trace case computes:
      - readout: probe applied to the prediction-site activation
      - confidence: option_probs[pred] from correctness.json
      - accuracy: 1 if correct else 0
      - raw_position: the prediction-site token index

    Bins cases by readout (qcut, equal count) and by raw_position. Returns a
    long-format DataFrame with both binnings stacked.
    """
    if not Path(correctness_path).exists():
        return pd.DataFrame([{"status": "correctness file missing"}])

    with open(correctness_path) as f:
        correctness_by_trace = json.load(f)

    X_train = _stack_activations(traces, layer, train_ids)
    y_train = _stack_target(traces, "within_task_fraction", train_ids)
    probe = RidgeProbe(alpha=1.0).fit(X_train, y_train)

    readouts: list[float] = []
    confidences: list[float] = []
    corrects: list[int] = []
    raw_positions: list[int] = []

    for t in traces:
        if t["trace_id"] not in test_ids:
            continue
        cr = correctness_by_trace.get(str(t["trace_id"]), [])
        if not cr:
            continue
        act = t["activations"][layer].numpy().astype(np.float32)
        boundaries = t["case_boundaries"]
        n_tokens = act.shape[0]
        for case in cr:
            ci = case["case_index"]
            pos = _prediction_site(ci, boundaries, n_tokens)
            readouts.append(float(probe.predict(act[pos : pos + 1])[0]))
            pred = case["pred"]
            confidences.append(float(case["option_probs"].get(pred, 0.0)))
            corrects.append(1 if case["correct"] else 0)
            raw_positions.append(pos)

    if not readouts:
        return pd.DataFrame([{"status": "no test cases with correctness data"}])

    df = pd.DataFrame(
        {
            "readout": readouts,
            "confidence": confidences,
            "correct": corrects,
            "raw_position": raw_positions,
        }
    )
    df["readout_bin"] = pd.qcut(df["readout"], q=n_bins, labels=False, duplicates="drop")
    df["raw_pos_bin"] = pd.qcut(
        df["raw_position"], q=n_bins, labels=False, duplicates="drop"
    )

    grouped_readout = (
        df.groupby("readout_bin")
        .agg(
            confidence=("confidence", "mean"),
            accuracy=("correct", "mean"),
            n=("correct", "size"),
        )
        .reset_index()
        .rename(columns={"readout_bin": "bin"})
        .assign(binning="within_task_fraction_readout")
    )
    grouped_raw = (
        df.groupby("raw_pos_bin")
        .agg(
            confidence=("confidence", "mean"),
            accuracy=("correct", "mean"),
            n=("correct", "size"),
        )
        .reset_index()
        .rename(columns={"raw_pos_bin": "bin"})
        .assign(binning="raw_position")
    )
    combined = pd.concat([grouped_readout, grouped_raw], ignore_index=True)
    combined["gap"] = combined["confidence"] - combined["accuracy"]
    return combined[["binning", "bin", "confidence", "accuracy", "n", "gap"]]
