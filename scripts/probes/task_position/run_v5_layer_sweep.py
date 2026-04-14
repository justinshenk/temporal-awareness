"""v5: dense early-layer probe sweep.

Trains within_task_fraction / tokens_until_boundary / task_index ridge probes
at every layer in the v5 extraction blob and reports per-layer test metrics
and A1 orthogonality. The goal is to localize the earliest layer at which
the position-belief signal is linearly decodable from the residual stream.

Helper functions build_matrix, primary_metric, and raw_position_features are
duplicated from scripts/probes/task_position/train_probes.py because that
module lives in scripts/ (not a proper installable package) and importing
across script modules is brittle. The logic is byte-identical to v1's training
code so v5 results are directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from src.probes.task_position.analysis import analysis_a1_orthogonality
from src.probes.task_position.probes import RidgeProbe

TARGETS = ["task_index", "within_task_fraction", "tokens_until_boundary"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--activations",
        default="results/probes/task_position/gemma-9b-it-v5/activations.pt",
    )
    p.add_argument(
        "--v1-split",
        default="results/probes/task_position/gemma-9b-it/probes/split.json",
    )
    p.add_argument(
        "--out-dir",
        default="results/probes/task_position/gemma-9b-it-v5/probes",
    )
    p.add_argument(
        "--out-md",
        default="results/probes/task_position/2026-04-13-v5-layer-sweep.md",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--model-label", default="Gemma-9B-IT")
    return p.parse_args()


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
    print(f"layers={layers} n_traces={len(traces)}")

    with open(args.v1_split) as f:
        split = json.load(f)
    train_ids = set(int(x) for x in split["train_ids"])
    test_ids = set(int(x) for x in split["test_ids"])
    print(f"train_ids={sorted(train_ids)} test_ids={sorted(test_ids)}")

    raw_pos_train = raw_position_features(traces, train_ids)
    raw_pos_test = raw_position_features(traces, test_ids)

    print("\nPhase 1: training probes at every layer × target")
    metrics_rows = []
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

            probe.save(out_dir / f"{target}_L{layer}.pkl")
            metrics_rows.append(
                {
                    "target": target,
                    "layer": layer,
                    "metric": metric,
                    "baseline_raw_pos": baseline_metric,
                    "delta": metric - baseline_metric,
                }
            )
            print(
                f"  {target:25s} L{layer:<3d} metric={metric:.4f} "
                f"baseline={baseline_metric:.4f} delta={metric - baseline_metric:+.4f}"
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    print("\nPhase 2: A1 orthogonality at every layer")
    a1_df = analysis_a1_orthogonality(traces, layers, train_ids, TARGETS, alpha=args.alpha)
    print(a1_df.to_string(index=False))
    a1_df.to_csv(out_dir / "a1_orthogonality.csv", index=False)

    print("\nPhase 3: per-target peak layer")
    peak_rows = []
    for target in TARGETS:
        sub = metrics_df[metrics_df["target"] == target]
        best_idx = sub["metric"].idxmax()
        peak_rows.append(
            {
                "target": target,
                "peak_layer": int(sub.loc[best_idx, "layer"]),
                "peak_metric": float(sub.loc[best_idx, "metric"]),
                "peak_delta": float(sub.loc[best_idx, "delta"]),
            }
        )
    peak_df = pd.DataFrame(peak_rows)
    print(peak_df.to_string(index=False))

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# v5: dense early-layer probe sweep ({args.model_label}, DDXPlus)\n\n")
    lines.append(
        f"Trained ridge probes at layers {layers} using the same 20-trace "
        "extraction and the same train/test split (16/4, seed 42) as v1. "
        "Goal: localize the earliest layer at which the within_task_fraction "
        "and tokens_until_boundary signals become linearly decodable.\n\n"
    )
    lines.append("## Peak layer per target\n\n")
    lines.append(peak_df.to_markdown(index=False) + "\n\n")
    lines.append("## Per-layer probe metrics\n\n")
    lines.append(metrics_df.to_markdown(index=False) + "\n\n")
    lines.append("## Per-layer A1 orthogonality (cosine with raw-position direction)\n\n")
    lines.append(a1_df.to_markdown(index=False) + "\n\n")
    lines.append(
        "## Interpretation\n\n"
        "Read the per-layer R² curve for `within_task_fraction` and "
        "`tokens_until_boundary`. Three regimes to distinguish:\n\n"
        "- **Early saturation (R² ≥ 0.8 at L2 or L4):** The signal is present "
        "almost immediately after token embeddings. The mechanism is upstream "
        "of almost every interventional knob — essentially baked into the "
        "embedding + first few attention operations. Any steering or ablation "
        "at L10+ is intervening on a downstream copy.\n"
        "- **Gradual climb (R² climbs from ~0 at L2 to ~0.9 at L10):** The "
        "signal is *constructed* across the early stack. The layer at which "
        "R² first exceeds ~0.5 is the earliest layer where an intervention "
        "could plausibly move the downstream calibration gap.\n"
        "- **Late emergence (R² still near baseline at L8):** The signal is "
        "built in a narrow window around L10. Our v2/v3/v4 interventions at "
        "L10+ were in the right place but the wrong modality (residual "
        "perturbation can't match the QK attention pattern that constructs "
        "the signal).\n\n"
        "The A1 orthogonality table tells a complementary story: at which "
        "layer does the probe direction first become orthogonal to raw "
        "position? Early orthogonality means the signal is *not* positional "
        "encoding at that layer.\n"
    )
    out_md.write_text("".join(lines))
    print(f"\nWrote {out_md}")


if __name__ == "__main__":
    main()
