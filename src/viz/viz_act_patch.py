"""Visualization for activation patching results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..activation_patching import AggregatedActivationPatchingResult, ActivationPatchingResult
from .heatmaps import HeatmapConfig, plot_layer_sweep, _finalize_plot


def visualize_activation_patching(
    result: AggregatedActivationPatchingResult,
    title: str = "Activation Patching Results",
    save_path: Path | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """Visualize activation patching results.

    Shows recovery by layer as a bar chart with statistics.

    Args:
        result: AggregatedActivationPatchingResult to visualize
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

    # Use heatmaps plot_layer_sweep with custom config
    config = HeatmapConfig(
        title=title,
        cbar_label="Mean Recovery",
        figsize=figsize,
    )

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
        y=mean_recovery, color="red", linestyle="--", alpha=0.7,
        label=f"Mean: {mean_recovery:.3f}"
    )

    # Highlight best layer
    best_layer, best_recovery = result.get_best_layer()
    if best_layer is not None and best_layer in layers:
        best_idx = layers.index(best_layer)
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)

    # Add stats text
    best_single = float(np.max(recoveries)) if len(recoveries) > 0 else 0.0
    stats_text = (
        f"Runs: {result.n_runs}\n"
        f"Flip rate: {result.flip_rate:.1%}\n"
        f"Best layer recovery: {best_single:.3f}"
    )
    ax.text(
        0.98, 0.98, stats_text,
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
    result: AggregatedActivationPatchingResult,
    title: str,
    save_path: Path | None,
) -> None:
    """Plot detailed visualization for single-intervention result."""
    # Collect all individual results
    all_results = [r for pr in result.results for r in pr.results]

    if not all_results:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No results", ha="center", va="center", fontsize=14)
        _finalize_plot(save_path)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Logprob difference before/after
    ax1 = axes[0]
    n_results = len(all_results)
    x = np.arange(n_results)
    width = 0.35

    orig_diffs = [r.original_logprob_diff for r in all_results]
    intv_diffs = [r.consistent_logprob_diff for r in all_results]

    bars1 = ax1.bar(x - width/2, orig_diffs, width, label="Original", color="steelblue", alpha=0.8)
    bars2 = ax1.bar(x + width/2, intv_diffs, width, label="Intervened", color="coral", alpha=0.8)

    ax1.set_xlabel("Pair", fontsize=11)
    ax1.set_ylabel("Logprob Difference", fontsize=11)
    ax1.set_title("Logprob Diff: Original vs Intervened", fontsize=12, fontweight="bold")
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.legend(fontsize=9)

    if n_results <= 10:
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"P{i+1}" for i in range(n_results)], fontsize=9)

    # Right panel: Recovery with details
    ax2 = axes[1]
    recoveries = [r.recovery for r in all_results]
    flipped = [r.choice_flipped for r in all_results]
    colors = ["forestgreen" if f else "steelblue" for f in flipped]

    bars = ax2.bar(x, recoveries, color=colors, alpha=0.8)

    # Mark degenerate flips
    for i, r in enumerate(all_results):
        if r.decoding_mismatch is True:
            bars[i].set_hatch("//")
            bars[i].set_edgecolor("red")

    ax2.set_xlabel("Pair", fontsize=11)
    ax2.set_ylabel("Recovery", fontsize=11)
    ax2.set_title("Recovery per Pair", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(1.0, max(recoveries) * 1.1) if recoveries else 1.0)

    if n_results <= 10:
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"P{i+1}" for i in range(n_results)], fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.8, label="No flip"),
        Patch(facecolor="forestgreen", alpha=0.8, label="Flipped"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Add summary stats
    first_result = all_results[0]

    # Get layer info from aggregated result
    if result.patched_layers:
        n_layers = len(result.patched_layers)
        if n_layers <= 5:
            layer_info = f"Layers {result.patched_layers}"
        else:
            layer_info = f"{n_layers} layers"
    elif hasattr(first_result, 'layer') and first_result.layer is not None:
        layer_info = f"Layer {first_result.layer}"
    else:
        layer_info = "All layers"

    # Get position info
    pos_mode = result.position_mode or "unknown"
    if pos_mode == "all":
        pos_info = "all positions"
    elif pos_mode == "explicit" and first_result.target.positions:
        n_pos = len(first_result.target.positions)
        pos_info = f"{n_pos} positions"
    else:
        pos_info = pos_mode

    stats_text = (
        f"Runs: {result.n_runs} | Mode: {first_result.mode}\n"
        f"Target: {layer_info}, {pos_info}\n"
        f"Mean recovery: {result.mean_recovery:.3f} | Flip rate: {result.flip_rate:.1%}"
    )

    fig.text(
        0.5, 0.02, stats_text,
        ha="center", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    _finalize_plot(save_path)


def visualize_single_result(
    result: ActivationPatchingResult,
    title: str = "Activation Patching",
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> None:
    """Visualize a single ActivationPatchingResult.

    Shows recovery for each intervention as a bar chart.

    Args:
        result: ActivationPatchingResult to visualize
        title: Plot title
        save_path: If provided, save to file; otherwise show on screen
        figsize: Figure size (width, height)
    """
    if not result.results:
        print("No results to visualize")
        return

    fig, ax = plt.subplots(figsize=figsize)

    recoveries = [r.recovery for r in result.results]
    flipped = [r.choice_flipped for r in result.results]

    x = np.arange(len(recoveries))
    colors = ["forestgreen" if f else "steelblue" for f in flipped]

    bars = ax.bar(x, recoveries, color=colors, alpha=0.8)

    # Mark degenerate flips with hatching
    for i, r in enumerate(result.results):
        if r.decoding_mismatch is True:
            bars[i].set_hatch("//")
            bars[i].set_edgecolor("red")

    ax.set_xlabel("Intervention", fontsize=11)
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
