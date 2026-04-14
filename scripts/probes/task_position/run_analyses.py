"""Run analyses A1/A2/A4 on Gemma-9B-IT activations and write results markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.probes.task_position.analysis import (
    analysis_a1_orthogonality,
    analysis_a2_failure_prediction,
    analysis_a4_calibration_gap,
)

TARGETS = ["task_index", "within_task_fraction", "tokens_until_boundary"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--activations",
        default="results/probes/task_position/gemma-9b-it/activations.pt",
    )
    p.add_argument(
        "--correctness",
        default="results/probes/task_position/gemma-9b-it/correctness.json",
    )
    p.add_argument(
        "--split",
        default="results/probes/task_position/gemma-9b-it/probes/split.json",
    )
    p.add_argument(
        "--metrics",
        default="results/probes/task_position/gemma-9b-it/probes/metrics.csv",
    )
    p.add_argument(
        "--out",
        default="results/probes/task_position/2026-04-13-v1-results.md",
    )
    p.add_argument("--calibration-layer", type=int, default=10)
    p.add_argument("--model-label", default="Gemma-9B-IT")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.activations}...")
    blob = torch.load(args.activations, weights_only=False)
    layers = blob["layers"]
    traces = blob["traces"]

    with open(args.split) as f:
        split = json.load(f)
    train_ids = set(int(x) for x in split["train_ids"])
    test_ids = set(int(x) for x in split["test_ids"])

    metrics_df = pd.read_csv(args.metrics)

    print("\nA1: orthogonality...")
    a1 = analysis_a1_orthogonality(traces, layers, train_ids, TARGETS)
    print(a1.to_string(index=False))

    print("\nA2: failure prediction...")
    a2 = analysis_a2_failure_prediction(
        traces, layers, train_ids, test_ids, args.correctness
    )
    print(a2.to_string(index=False))

    print(f"\nA4: calibration gap (layer {args.calibration_layer})...")
    a4 = analysis_a4_calibration_gap(
        traces, args.calibration_layer, train_ids, test_ids, args.correctness
    )
    print(a4.to_string(index=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    write_results_markdown(
        out=out,
        layers=layers,
        train_ids=train_ids,
        test_ids=test_ids,
        metrics_df=metrics_df,
        a1=a1,
        a2=a2,
        a4=a4,
        calibration_layer=args.calibration_layer,
        model_label=args.model_label,
    )
    print(f"\nWrote {out}")


def write_results_markdown(
    out: Path,
    layers: list[int],
    train_ids: set,
    test_ids: set,
    metrics_df: pd.DataFrame,
    a1: pd.DataFrame,
    a2: pd.DataFrame,
    a4: pd.DataFrame,
    calibration_layer: int,
    model_label: str = "Gemma-9B-IT",
) -> None:
    best = (
        metrics_df.loc[metrics_df.groupby("target")["metric"].idxmax()]
        [["target", "layer", "metric", "baseline_raw_pos", "delta"]]
        .reset_index(drop=True)
    )

    a2_has_data = "baseline_auc" in a2.columns
    a4_has_data = "binning" in a4.columns

    criterion_1 = bool((metrics_df["delta"] > 0).any())

    a1_taskidx = a1[a1["target"] == "task_index"]["cosine_with_raw_pos"].abs().mean()
    a1_wf = a1[a1["target"] == "within_task_fraction"]["cosine_with_raw_pos"].abs().mean()
    a1_tu = a1[a1["target"] == "tokens_until_boundary"]["cosine_with_raw_pos"].abs().mean()
    criterion_2 = bool(a1_taskidx > 0.5 and a1_wf < 0.3 and a1_tu < 0.3)

    criterion_3 = bool(a4_has_data and len(a4) >= 2)

    a2_strong_success = False
    if a2_has_data:
        a2_strong_success = bool((a2["delta"] > 0.05).any())

    lines = []
    lines.append(f"# Task-Position Probes v1 Results ({model_label}, DDXPlus)\n")
    lines.append(
        f"Multi-case DDXPlus traces accumulated to ~90% of {model_label}'s 8k context. "
        "Per-token residual streams captured at layers "
        f"{layers}. Twenty traces split 80/20 by trace id "
        f"(train={sorted(train_ids)}, test={sorted(test_ids)}, seed=42).\n"
    )
    lines.append("## Headline numbers (best layer per target, from training run)\n")
    lines.append(best.to_markdown(index=False) + "\n")
    lines.append(
        "Metric is Spearman ρ for `task_index` (ordinal target) and R² for the other "
        "two (the latter trained in `log1p` space). The baseline is a single-feature "
        "Ridge on raw token position; if a residual-stream probe doesn't beat it, the "
        "probe is just reading positional encoding.\n"
    )

    lines.append("## A1: Orthogonality between task-position and raw-position directions\n")
    lines.append(
        "For each layer we fit a ridge probe to predict raw token position from the "
        "residual stream, then compute the cosine similarity between that direction "
        "and each target probe's direction. Strong alignment = 'this target is just "
        "raw position'. Near-zero = 'this target encodes a new signal'.\n"
    )
    lines.append(a1.to_markdown(index=False) + "\n")
    lines.append(
        f"**Mean |cosine| with raw position** — `task_index` {a1_taskidx:.3f}, "
        f"`within_task_fraction` {a1_wf:.3f}, `tokens_until_boundary` {a1_tu:.3f}.\n"
    )

    lines.append("## A2: Upcoming failure prediction\n")
    lines.append(
        "Logistic regression trained on TRAIN traces, scored on TEST. "
        "Baseline: residual stream only. Treatment: residual stream + "
        "`[task_index, within_task_fraction, log1p(tokens_until_boundary)]` at the "
        "case's last prompt token. Target to beat: 0.67–0.68 (Qwen-7B baseline from "
        "context-fatigue writeup Section 4.1).\n"
    )
    if a2_has_data:
        lines.append(a2.to_markdown(index=False) + "\n")
    else:
        lines.append("_correctness file missing — analysis skipped_\n")

    lines.append(
        f"## A4: Calibration gap vs within_task_fraction readout (L{calibration_layer})\n"
    )
    lines.append(
        "We train a within_task_fraction probe on TRAIN activations and apply it to "
        "the prediction-site activation of each TEST case. We then bin cases two ways: "
        "(a) by the probe readout (the model's *subjective* sense of task-lateness), "
        "and (b) by raw token position (the *objective* depth into the trace). For "
        "each bin we report mean confidence (`option_probs[pred]`), mean accuracy "
        "(0/1), and the calibration gap = confidence - accuracy. The hypothesis is "
        "that calibration degrades more sharply with subjective lateness than with "
        "raw position.\n"
    )
    if a4_has_data:
        lines.append(a4.to_markdown(index=False) + "\n")
    else:
        lines.append("_correctness file missing — analysis skipped_\n")

    lines.append("## Verdict against v1 success criteria\n")
    lines.append(
        "**Criterion 1** (signal exists): at least one task-position probe beats both "
        "baselines on test split for at least one target.\n"
    )
    lines.append(f"  - Status: {'**MET**' if criterion_1 else '**NOT MET**'}\n")
    lines.append(
        "**Criterion 2** (interpretable orthogonality picture): A1 produces a clear "
        "alignment story.\n"
    )
    lines.append(f"  - Status: {'**MET**' if criterion_2 else '**NOT MET**'}\n")
    lines.append(
        "**Criterion 3** (calibration figure exists): A4 produces a binned table "
        "either supporting or refuting the calibration-gap hypothesis.\n"
    )
    lines.append(f"  - Status: {'**MET**' if criterion_3 else '**NOT MET**'}\n")

    if a2_strong_success:
        lines.append(
            "**Strong success** also: A2 beats the failure-prediction baseline by "
            ">= +0.05 AUC at at least one layer.\n"
        )

    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
