"""Denoising vs noising comparison scatter plots.

Shows how recovery (denoising) correlates with disruption (noising)
across layers and positions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ....activation_patching.coarse import SweepStepResults
from ....viz.token_coloring import PairTokenColoring
from .helpers import get_tick_color, setup_grid


def plot_comparison(
    layer_data: SweepStepResults | None,
    position_data: SweepStepResults | None,
    output_dir: Path,
    coloring: PairTokenColoring | None = None,
    step_size: int = 8,
) -> None:
    """Plot denoising vs noising comparison scatter plots.

    Creates a 1x2 figure with layer comparison (left) and position comparison (right).

    Args:
        layer_data: Layer sweep results
        position_data: Position sweep results
        output_dir: Directory to save output
        coloring: Token coloring for position colors
        step_size: Step size used in the sweep
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="white")
    fig.suptitle(
        f"Activation Patching: Denoising vs Noising Recovery (step size={step_size})",
        fontsize=20,
        fontweight="bold",
    )

    # Layer comparison
    _plot_layer_comparison(axes[0], layer_data)

    # Position comparison
    _plot_position_comparison(axes[1], position_data, coloring)

    # Save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = output_dir / f"denoising_vs_noising_{step_size}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def _plot_layer_comparison(
    ax: plt.Axes,
    layer_data: SweepStepResults | None,
) -> None:
    """Plot layer sweep comparison scatter plot."""
    ax.set_facecolor("white")

    if not layer_data:
        ax.text(0.5, 0.5, "No layer results", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return

    layers = sorted(layer_data.keys())
    recoveries = [layer_data[l].recovery for l in layers]
    disruptions = [layer_data[l].disruption for l in layers]

    # Viridis colormap for layers
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))

    # Find extreme points to label
    label_indices = _get_extreme_indices(recoveries, disruptions, n_top=5)
    label_indices.add(0)  # First layer
    label_indices.add(len(layers) - 1)  # Last layer

    # Plot points
    for i, (rec, dis, layer) in enumerate(zip(recoveries, disruptions, layers)):
        ax.scatter(
            rec,
            dis,
            c=[colors[i]],
            s=200,
            edgecolors="black",
            linewidth=1.5,
            alpha=0.85,
            zorder=3,
        )
        if i in label_indices:
            _add_point_label(ax, rec, dis, f"L{layer}")

    _setup_comparison_axes(ax, "Layer Sweep: Patch Effect Comparison")


def _plot_position_comparison(
    ax: plt.Axes,
    position_data: SweepStepResults | None,
    coloring: PairTokenColoring | None,
) -> None:
    """Plot position sweep comparison scatter plot."""
    ax.set_facecolor("white")

    if not position_data:
        ax.text(0.5, 0.5, "No position results", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return

    positions = sorted(position_data.keys())
    recoveries = [position_data[p].recovery for p in positions]
    disruptions = [position_data[p].disruption for p in positions]
    point_colors = [get_tick_color(p, coloring) for p in positions]

    # Find extreme points to label (top 3 to reduce clutter)
    label_indices = _get_extreme_indices(recoveries, disruptions, n_top=3)

    # Plot points
    for i, (rec, dis, pos) in enumerate(zip(recoveries, disruptions, positions)):
        ax.scatter(
            rec,
            dis,
            c=[point_colors[i]],
            s=200,
            edgecolors="black",
            linewidth=1.5,
            alpha=0.75,
            zorder=3,
        )
        if i in label_indices:
            _add_point_label(ax, rec, dis, f"P{pos}")

    _setup_comparison_axes(ax, "Position Sweep: Patch Effect Comparison")


def _get_extreme_indices(
    recoveries: list[float],
    disruptions: list[float],
    n_top: int = 5,
) -> set[int]:
    """Get indices of extreme points in recovery and disruption."""
    rec_arr = np.array(recoveries)
    dis_arr = np.array(disruptions)
    rec_sorted_idx = np.argsort(rec_arr)[::-1][:n_top]
    dis_sorted_idx = np.argsort(dis_arr)[::-1][:n_top]
    return set(rec_sorted_idx) | set(dis_sorted_idx)


def _add_point_label(ax: plt.Axes, x: float, y: float, label: str) -> None:
    """Add annotation label to a scatter point."""
    ax.annotate(
        label,
        (x, y),
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="bottom",
        xytext=(0, 10),
        textcoords="offset points",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="gray",
            alpha=0.8,
        ),
    )


def _setup_comparison_axes(ax: plt.Axes, title: str) -> None:
    """Configure axes for comparison scatter plot."""
    ax.plot([0, 1], [0, 1], "k--", alpha=0.7, linewidth=2.5, label="Equal effect (y=x)")
    ax.set_xlabel("Recovery (Denoising)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Disruption (Noising)", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    setup_grid(ax)
    ax.axhline(y=0.5, color="#888888", linestyle=":", alpha=0.8, linewidth=2)
    ax.axvline(x=0.5, color="#888888", linestyle=":", alpha=0.8, linewidth=2)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(
        loc="lower right",
        fontsize=11,
        frameon=True,
        fancybox=True,
        title="Reference Lines",
        title_fontsize=10,
    )
