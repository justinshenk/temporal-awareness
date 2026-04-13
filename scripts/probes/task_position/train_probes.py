"""Train ridge probes for each (target, layer) on saved activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from src.probes.task_position.probes import RidgeProbe

TARGETS = ["task_index", "within_task_fraction", "tokens_until_boundary"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--activations",
        default="results/probes/task_position/gemma-9b-it/activations.pt",
    )
    p.add_argument(
        "--out-dir",
        default="results/probes/task_position/gemma-9b-it/probes",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-frac", type=float, default=0.2)
    return p.parse_args()


def split_traces(n_traces: int, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_traces)
    n_test = max(1, int(round(n_traces * test_frac)))
    test_ids = set(int(x) for x in perm[:n_test])
    train_ids = set(int(x) for x in perm[n_test:])
    return train_ids, test_ids


def build_matrix(traces: list, layer: int, target: str, trace_ids: set):
    """Stack per-token activations and labels for the given layer and target."""
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for t in traces:
        if t["trace_id"] not in trace_ids:
            continue
        act = t["activations"][layer].numpy().astype(np.float32)
        labels = t["labels"][target]
        if target == "tokens_until_boundary":
            y = np.log1p(np.asarray(labels, dtype=np.float64))
        else:
            y = np.asarray(labels, dtype=np.float64)
        Xs.append(act)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def primary_metric(target: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if target == "task_index":
        return float(spearmanr(y_true, y_pred).correlation)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def raw_position_features(traces: list, trace_ids: set) -> np.ndarray:
    """Concatenate per-token raw token position across the given traces."""
    parts = []
    for t in traces:
        if t["trace_id"] not in trace_ids:
            continue
        n = t["activations"][next(iter(t["activations"]))].shape[0]
        parts.append(np.arange(n, dtype=np.float64).reshape(-1, 1))
    return np.vstack(parts)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.activations}...")
    blob = torch.load(args.activations, weights_only=False)
    layers = blob["layers"]
    traces = blob["traces"]
    n_traces = len(traces)
    train_ids, test_ids = split_traces(n_traces, args.test_frac, args.seed)
    print(f"n_traces={n_traces} train={sorted(train_ids)} test={sorted(test_ids)}")

    raw_pos_train = raw_position_features(traces, train_ids)
    raw_pos_test = raw_position_features(traces, test_ids)

    results = []
    for target in TARGETS:
        for layer in layers:
            X_train, y_train = build_matrix(traces, layer, target, train_ids)
            X_test, y_test = build_matrix(traces, layer, target, test_ids)

            probe = RidgeProbe(alpha=args.alpha)
            probe.fit(X_train, y_train)
            metric = primary_metric(target, y_test, probe.predict(X_test))

            baseline = Ridge(alpha=args.alpha).fit(raw_pos_train, y_train)
            baseline_metric = primary_metric(
                target, y_test, baseline.predict(raw_pos_test)
            )

            probe_path = out_dir / f"{target}_L{layer}.pkl"
            probe.save(probe_path)

            results.append(
                {
                    "target": target,
                    "layer": layer,
                    "metric": metric,
                    "baseline_raw_pos": baseline_metric,
                    "delta": metric - baseline_metric,
                    "n_train_tokens": int(X_train.shape[0]),
                    "n_test_tokens": int(X_test.shape[0]),
                }
            )
            print(
                f"  {target:25s} L{layer:<3d} metric={metric:.4f} "
                f"baseline={baseline_metric:.4f} delta={metric - baseline_metric:+.4f}"
            )

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "split.json", "w") as f:
        json.dump(
            {"train_ids": sorted(train_ids), "test_ids": sorted(test_ids)}, f
        )
    print(f"\nSaved {len(results)} probes and metrics to {out_dir}")


if __name__ == "__main__":
    main()
