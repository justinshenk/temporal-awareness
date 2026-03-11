"""Aggregated coarse patching visualization.

Shows mean scores across multiple samples for layer and position sweeps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ....activation_patching.coarse import CoarseActPatchAggregatedResults
from ....viz.token_coloring import PairTokenColoring
from .helpers import finalize_plot, setup_grid


def plot_aggregated(
    result: CoarseActPatchAggregatedResults,
    output_dir: Path,
    coloring: PairTokenColoring | None = None,
) -> None:
    """Visualize aggregated coarse patching results.

    Creates one plot per step size showing mean layer and position scores.

    Args:
        result: Aggregated results across multiple samples
        output_dir: Directory to save output
        coloring: Token coloring (unused, kept for API compatibility)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create plot for each step size
    all_step_sizes = set(result.layer_step_sizes) | set(result.position_step_sizes)

    for step_size in sorted(all_step_sizes):
        _plot_step_size(result, output_dir, step_size)

    print(f"[viz] Aggregated coarse patching plots saved to {output_dir}")


def _plot_step_size(
    result: CoarseActPatchAggregatedResults,
    output_dir: Path,
    step_size: int,
) -> None:
    """Create aggregated plot for a specific step size."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")
    fig.suptitle(
        f"Aggregated Coarse Patching (n={result.n_samples} samples, step={step_size})",
        fontsize=16,
        fontweight="bold",
    )

    # Layer scores
    _plot_layer_scores(axes[0], result, step_size)

    # Position scores
    _plot_position_scores(axes[1], result, step_size)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = output_dir / f"coarse_patching_agg_{step_size}.png"
    finalize_plot(fig, save_path)


def _plot_layer_scores(
    ax: plt.Axes,
    result: CoarseActPatchAggregatedResults,
    step_size: int,
) -> None:
    """Plot mean layer scores."""
    ax.set_facecolor("white")

    layer_scores = result.get_mean_layer_scores(step_size=step_size)
    if not layer_scores:
        ax.text(0.5, 0.5, "No layer data", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return

    layers = sorted(layer_scores.keys())
    scores = [layer_scores[l] for l in layers]

    ax.bar(layers, scores, color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Score", fontsize=12, fontweight="bold")
    ax.set_title("Mean Layer Score", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    setup_grid(ax)

    # Highlight best layers
    best_layers = result.best_layers(n_top=3, step_size=step_size)
    for layer in best_layers:
        if layer in layers:
            idx = layers.index(layer)
            ax.bar([layer], [scores[idx]], color="#DD5544", edgecolor="black", linewidth=1)

    # Add legend
    ax.text(
        0.98,
        0.98,
        f"Best: {best_layers}",
        transform=ax.transAxes,
        fontsize=10,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8),
    )


def _plot_position_scores(
    ax: plt.Axes,
    result: CoarseActPatchAggregatedResults,
    step_size: int,
) -> None:
    """Plot mean position scores."""
    ax.set_facecolor("white")

    pos_scores = result.get_mean_position_scores(step_size=step_size)
    if not pos_scores:
        ax.text(0.5, 0.5, "No position data", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return

    positions = sorted(pos_scores.keys())
    scores = [pos_scores[p] for p in positions]

    ax.bar(positions, scores, color="#55A868", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Position", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Score", fontsize=12, fontweight="bold")
    ax.set_title("Mean Position Score", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    setup_grid(ax)

    # Add threshold line
    ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.7, linewidth=2, label="80% threshold")
    ax.legend(loc="upper right", fontsize=10)
