#!/usr/bin/env python3
"""Summarize RQ47 probe, intervention, and oversight results for presentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "results/tables/rq47"
FIGURE_DIR = ROOT / "results/figures/rq47"
INTERVENTION_DIR = ROOT / "results/temporal_interventions"
OVERSIGHT_DIR = ROOT / "results/temporal_oversight"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_interventions() -> pd.DataFrame:
    frames = []
    for path in INTERVENTION_DIR.glob("*/*/*.csv"):
        if "layers-0_" in path.name:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_file"] = str(path.relative_to(ROOT))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_oversight() -> pd.DataFrame:
    frames = []
    for path in OVERSIGHT_DIR.glob("*/*summary.csv"):
        if "layers-0_" in path.name:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_file"] = str(path.relative_to(ROOT))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_probe_results() -> pd.DataFrame:
    rows = []
    for method in ["lr", "dmm", "attn"]:
        for path in sorted((ROOT / "research/results" / method).glob("*.csv")):
            df = pd.read_csv(path)
            best = df.sort_values(["test_accuracy", "cv_accuracy_mean"], ascending=False).iloc[0]
            rows.append(
                {
                    "method": method,
                    "model_tag": path.name.split("_temporal_probe_")[0],
                    "best_layer": int(best.layer),
                    "best_test_acc": float(best.test_accuracy),
                    "mean_test_acc": float(df.test_accuracy.mean()),
                    "n_layers": len(df),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "probe_best_layers.csv", index=False)
    return out


def summarize_steering(df: pd.DataFrame) -> pd.DataFrame:
    steering = df[df.experiment == "steering"].copy()
    if steering.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["model_alias", "layer_source_method", "layer_selection", "component"]
    for key, group in steering.groupby(group_cols):
        means = group.groupby("strength").intervention_value.mean().sort_index()
        zero = means.get(0.0, pd.NA)
        plus = means.get(3.0, pd.NA)
        minus = means.get(-3.0, pd.NA)
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "mean_at_minus3": minus,
                "mean_at_zero": zero,
                "mean_at_plus3": plus,
                "plus3_minus_zero": plus - zero if pd.notna(plus) and pd.notna(zero) else pd.NA,
                "zero_minus_minus3": zero - minus if pd.notna(zero) and pd.notna(minus) else pd.NA,
                "n_rows": len(group),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "steering_dose_response_summary.csv", index=False)
    return out


def summarize_effects(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    subset = df[df.experiment == experiment].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["effect_clipped"] = subset.normalized_effect.clip(-2, 2)
    out = (
        subset.groupby(["model_alias", "layer_source_method", "layer_selection", "component"])
        .agg(
            median_effect=("normalized_effect", "median"),
            mean_clipped_effect=("effect_clipped", "mean"),
            q25_effect=("normalized_effect", lambda s: s.quantile(0.25)),
            q75_effect=("normalized_effect", lambda s: s.quantile(0.75)),
            n_rows=("normalized_effect", "count"),
        )
        .reset_index()
    )
    out.to_csv(TABLE_DIR / f"{experiment}_summary.csv", index=False)
    return out


def summarize_ablation(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df.experiment == "ablation"].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["drop_from_clean"] = 1 - subset.normalized_effect
    subset["drop_clipped"] = subset.drop_from_clean.clip(-2, 2)
    out = (
        subset.groupby(["model_alias", "layer_source_method", "layer_selection", "component"])
        .agg(
            median_drop=("drop_from_clean", "median"),
            mean_clipped_drop=("drop_clipped", "mean"),
            q25_drop=("drop_from_clean", lambda s: s.quantile(0.25)),
            q75_drop=("drop_from_clean", lambda s: s.quantile(0.75)),
            n_rows=("drop_from_clean", "count"),
        )
        .reset_index()
    )
    out.to_csv(TABLE_DIR / "ablation_summary.csv", index=False)
    return out


def summarize_oversight(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby(["model_alias", "layer_source_method"])
        .agg(
            median_initial=("initial_score", "median"),
            median_final=("final_score", "median"),
            median_pre_event_delta=("delta_before_event", "median"),
            event_rate=("event_detected", "mean"),
            n_rows=("prompt_id", "count"),
        )
        .reset_index()
    )
    out.to_csv(TABLE_DIR / "temporal_oversight_summary.csv", index=False)

    top = df.sort_values("delta_before_event", ascending=False).head(20)
    top.to_csv(TABLE_DIR / "temporal_oversight_top_pre_event_drifts.csv", index=False)
    return out


def plot_heatmap(table: pd.DataFrame, value: str, title: str, filename: str) -> None:
    if table.empty:
        return
    pivot = table.pivot_table(
        index="model_alias",
        columns="layer_source_method",
        values=value,
        aggfunc="median",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=160)
    plt.close(fig)


def plot_summary_figures(
    steering: pd.DataFrame,
    activation: pd.DataFrame,
    attribution: pd.DataFrame,
    ablation: pd.DataFrame,
    oversight: pd.DataFrame,
) -> None:
    best_resid_steering = steering[
        (steering.layer_selection == "best") & (steering.component == "resid")
    ]
    plot_heatmap(
        best_resid_steering,
        "plus3_minus_zero",
        "Steering Effect (+3 vs 0), Best Residual Layer",
        "steering_best_resid_heatmap.png",
    )

    for name, table, value, title in [
        ("activation_patching_best_resid_heatmap.png", activation, "median_effect", "Activation Patching Median Recovery"),
        ("attribution_patching_best_resid_heatmap.png", attribution, "median_effect", "Attribution Patching Median Estimate"),
        ("ablation_best_resid_heatmap.png", ablation, "median_drop", "Ablation Median Drop From Clean"),
    ]:
        subset = table[(table.layer_selection == "best") & (table.component == "resid")]
        plot_heatmap(subset, value, title, name)

    plot_heatmap(
        oversight,
        "median_pre_event_delta",
        "Temporal Probe Pre-Event Drift",
        "temporal_oversight_pre_event_delta_heatmap.png",
    )


def main() -> None:
    ensure_dirs()
    probe = summarize_probe_results()
    interventions = load_interventions()
    oversight_raw = load_oversight()

    steering = summarize_steering(interventions)
    activation = summarize_effects(interventions, "activation_patching")
    attribution = summarize_effects(interventions, "attribution_patching")
    ablation = summarize_ablation(interventions)
    oversight = summarize_oversight(oversight_raw)

    plot_summary_figures(steering, activation, attribution, ablation, oversight)

    print(f"Wrote tables to {TABLE_DIR.relative_to(ROOT)}")
    print(f"Wrote figures to {FIGURE_DIR.relative_to(ROOT)}")
    print(f"Probe rows: {len(probe)}")
    print(f"Intervention rows: {len(interventions)}")
    print(f"Oversight rows: {len(oversight_raw)}")


if __name__ == "__main__":
    main()
