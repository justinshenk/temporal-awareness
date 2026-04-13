"""Analyses A1 / A2 / A4 for task-position probes.

A1: orthogonality between task-position probe directions and a raw-token-position
    probe direction at the same layer. Tests whether task-position probes encode
    a new signal vs. just renaming positional encoding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
