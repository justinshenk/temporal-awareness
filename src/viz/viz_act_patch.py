"""Visualization for activation patching results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..activation_patching import ActPatchAggregatedResult, ActPatchPairResult
from .layer_position_heatmaps import HeatmapConfig
from .plot_helpers import finalize_plot as _finalize_plot


def visualize_activation_patching(
    result: ActPatchAggregatedResult,
    title: str = "Activation Patching Results",
    save_path: Path | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """Visualize activation patching results.

    Shows recovery by layer as a bar chart with statistics.

    Args:
        result: ActPatchAggregatedResult to visualize
        title: Plot title
        save_path: If provided, save to file; otherwise show on screen
        figsize: Figure size (width, height)
    """
    recovery_by_layer = result.get_recovery_by_layer()

    if not recovery_by_layer:
        print("No layer data to visualize")
        return

    # Sort layers (handle None for "all layers" case)
    layers = sorted([l for l in recovery_by_layer.keys() if l is not None])
    has_all_layers = None in recovery_by_layer

    if not layers and has_all_layers:
        _plot_single_bar(result, title, save_path)
        return

    recoveries = np.array([recovery_by_layer[l] for l in layers])

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(layers))
    bars = ax.bar(x, recoveries, color="steelblue", alpha=0.8)

    # Color bars by recovery value
    max_recovery = float(np.max(recoveries)) if len(recoveries) > 0 else 1.0
    for bar, recovery in zip(bars, recoveries):
        intensity = recovery / max_recovery if max_recovery > 0 else 0
        bar.set_color(plt.cm.RdYlGn(0.3 + 0.6 * intensity))

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean Recovery", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)

    # Add horizontal line at mean
    mean_recovery = result.mean_recovery
    ax.axhline(
        y=mean_recovery,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {mean_recovery:.3f}",
    )

    # Highlight best layer
    best_layer, best_recovery = result.get_best_layer()
    if best_layer is not None and best_layer in layers:
        best_idx = layers.index(best_layer)
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)

    # Add stats text
    best_single = float(np.max(recoveries)) if len(recoveries) > 0 else 0.0
    stats_text = f"Samples: {result.n_samples}\nBest layer recovery: {best_single:.3f}"
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(bottom=0)

    _finalize_plot(save_path)


def _plot_single_bar(
    result: ActPatchAggregatedResult,
    title: str,
    save_path: Path | None,
) -> None:
    """Plot detailed visualization for single-intervention result."""
    # Collect all individual recoveries
    recoveries = []
    for pair in result.by_sample.values():
        for target_result in pair.by_target.values():
            recoveries.append(target_result.recovery)

    if not recoveries:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No results", ha="center", va="center", fontsize=14)
        _finalize_plot(save_path)
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(recoveries))
    bars = ax.bar(x, recoveries, color="steelblue", alpha=0.8)

    # Color by recovery value
    max_recovery = max(recoveries) if recoveries else 1.0
    for bar, recovery in zip(bars, recoveries):
        intensity = recovery / max_recovery if max_recovery > 0 else 0
        bar.set_color(plt.cm.RdYlGn(0.3 + 0.6 * intensity))

    ax.set_xlabel("Pair-Target", fontsize=11)
    ax.set_ylabel("Recovery", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add mean line
    mean_val = sum(recoveries) / len(recoveries)
    ax.axhline(
        y=mean_val,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {mean_val:.3f}",
    )

    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(bottom=0)

    _finalize_plot(save_path)


def visualize_single_result(
    result: ActPatchPairResult,
    title: str = "Activation Patching",
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> None:
    """Visualize a single ActPatchPairResult.

    Shows recovery for each target as a bar chart.

    Args:
        result: ActPatchPairResult to visualize
        title: Plot title
        save_path: If provided, save to file; otherwise show on screen
        figsize: Figure size (width, height)
    """
    if not result.by_target:
        print("No results to visualize")
        return

    fig, ax = plt.subplots(figsize=figsize)

    targets = list(result.by_target.keys())
    recoveries = [result.by_target[t].recovery for t in targets]
    flip_counts = [result.by_target[t].flip_count for t in targets]

    x = np.arange(len(recoveries))
    colors = ["forestgreen" if fc > 0 else "steelblue" for fc in flip_counts]

    bars = ax.bar(x, recoveries, color=colors, alpha=0.8)

    ax.set_xlabel("Target", fontsize=11)
    ax.set_ylabel("Recovery", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.8, label="No flip"),
        Patch(facecolor="forestgreen", alpha=0.8, label="Flipped"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.set_ylim(bottom=0)

    _finalize_plot(save_path)
