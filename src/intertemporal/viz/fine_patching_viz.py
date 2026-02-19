"""Visualization for fine-grained activation patching results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...activation_patching import ActPatchAggregatedResult
from ...viz.plot_helpers import finalize_plot as _finalize_plot
from ...viz.patching_heatmaps import (
    PatchingHeatmapConfig,
    plot_patching_heatmap,
    visualize_activation_patching_result,
)
from ...viz.viz_act_patch import visualize_activation_patching


def visualize_fine_patching(
    result: ActPatchAggregatedResult | None,
    output_dir: Path,
    position_labels: list[str] | None = None,
    section_markers: dict[str, int] | None = None,
) -> None:
    """Visualize fine-grained activation patching results.

    Creates:
    - Recovery by layer bar chart
    - Layer x position heatmap (if available)
    - Per-sample summary

    Args:
        result: ActPatchAggregatedResult to visualize
        output_dir: Directory to save plots
        position_labels: Optional labels for positions
        section_markers: Optional section markers for prompt structure
    """
    if result is None or not result.by_sample:
        print("[viz] No fine patching results to visualize")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recovery by layer bar chart
    visualize_activation_patching(
        result,
        title="Fine Activation Patching: Recovery by Layer",
        save_path=output_dir / "fine_patching_by_layer.png",
    )

    # Layer x position heatmap
    visualize_activation_patching_result(
        result,
        position_labels=position_labels,
        section_markers=section_markers,
        save_path=output_dir / "fine_patching_heatmap.png",
    )

    # Per-sample breakdown
    _plot_per_sample_summary(result, output_dir)

    # Statistics summary
    _plot_statistics_summary(result, output_dir)

    print(f"[viz] Fine patching plots saved to {output_dir}")


def _plot_per_sample_summary(
    result: ActPatchAggregatedResult,
    output_dir: Path,
) -> None:
    """Plot per-sample recovery summary."""
    if result.n_samples == 0:
        return

    sample_ids = sorted(result.by_sample.keys())
    recoveries = [result.by_sample[sid].mean_recovery for sid in sample_ids]

    fig, ax = plt.subplots(figsize=(max(8, len(sample_ids) * 0.8), 5))

    x = np.arange(len(sample_ids))
    bars = ax.bar(x, recoveries, color="steelblue", alpha=0.8)

    # Color by recovery value
    max_recovery = max(recoveries) if recoveries else 1.0
    for bar, recovery in zip(bars, recoveries):
        intensity = recovery / max_recovery if max_recovery > 0 else 0
        bar.set_color(plt.cm.RdYlGn(0.3 + 0.6 * intensity))

    ax.set_xlabel("Sample ID", fontsize=11)
    ax.set_ylabel("Mean Recovery", fontsize=11)
    ax.set_title("Fine Patching: Per-Sample Recovery", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(sid) for sid in sample_ids], fontsize=9)
    ax.set_ylim(bottom=0)

    # Add mean line
    mean_val = result.mean_recovery
    ax.axhline(
        y=mean_val,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {mean_val:.3f}",
    )
    ax.legend(fontsize=9)

    _finalize_plot(output_dir / "fine_patching_per_sample.png")


def _plot_statistics_summary(
    result: ActPatchAggregatedResult,
    output_dir: Path,
) -> None:
    """Plot summary statistics for fine patching."""
    recovery_by_layer = result.get_recovery_by_layer()

    # Filter out None keys
    layer_data = {l: r for l, r in recovery_by_layer.items() if l is not None}

    if not layer_data:
        return

    layers = sorted(layer_data.keys())
    recoveries = [layer_data[l] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Recovery distribution histogram
    ax1 = axes[0]
    all_recoveries = []
    for pair in result.by_sample.values():
        for target_result in pair.by_target.values():
            all_recoveries.append(target_result.recovery)

    if all_recoveries:
        ax1.hist(
            all_recoveries, bins=20, color="steelblue", alpha=0.7, edgecolor="black"
        )
        ax1.axvline(
            x=np.mean(all_recoveries),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(all_recoveries):.3f}",
        )
        ax1.set_xlabel("Recovery", fontsize=11)
        ax1.set_ylabel("Count", fontsize=11)
        ax1.set_title("Recovery Distribution", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=9)

    # Recovery by layer heatmap (compact)
    ax2 = axes[1]
    if recoveries:
        sorted_layers = sorted(layers, key=lambda l: layer_data[l], reverse=True)
        sorted_recoveries = [layer_data[l] for l in sorted_layers]

        y = np.arange(len(sorted_layers))
        bars = ax2.barh(y, sorted_recoveries, color="steelblue", alpha=0.8)

        max_recovery = max(sorted_recoveries) if sorted_recoveries else 1.0
        for bar, recovery in zip(bars, sorted_recoveries):
            intensity = recovery / max_recovery if max_recovery > 0 else 0
            bar.set_color(plt.cm.RdYlGn(0.3 + 0.6 * intensity))

        ax2.set_yticks(y)
        ax2.set_yticklabels([f"L{l}" for l in sorted_layers], fontsize=9)
        ax2.set_xlabel("Recovery", fontsize=11)
        ax2.set_ylabel("Layer", fontsize=11)
        ax2.set_title("Recovery by Layer (Sorted)", fontsize=12, fontweight="bold")
        ax2.set_xlim(left=0)

    plt.tight_layout()
    _finalize_plot(output_dir / "fine_patching_statistics.png")
