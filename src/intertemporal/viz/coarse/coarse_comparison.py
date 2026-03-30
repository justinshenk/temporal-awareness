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
from .coarse_helpers import get_tick_color, setup_grid


def plot_comparison(
    layer_data: SweepStepResults | None,
    position_data: SweepStepResults | None,
    output_dir: Path,
    coloring: PairTokenColoring | None = None,
    step_size: int = 8,
    component: str = "resid_post",
) -> None:
    """Plot denoising vs noising comparison scatter plots.

    Creates single-panel or two-panel figure depending on available data.

    Args:
        layer_data: Layer sweep results
        position_data: Position sweep results
        output_dir: Directory to save output
        coloring: Token coloring for position colors
        step_size: Step size used in the sweep
        component: Component being patched (for plot title)
    """
    has_layer = bool(layer_data)
    has_position = bool(position_data)

    if not has_layer and not has_position:
        return

    # Determine layout based on available data
    if has_layer and has_position:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="white")
        _plot_layer_comparison(axes[0], layer_data)
        _plot_position_comparison(axes[1], position_data, coloring)
        title_suffix = f"(step={step_size})"
    elif has_layer:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
        _plot_layer_comparison(ax, layer_data)
        title_suffix = f"Layer Sweep (step={step_size})"
    else:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
        _plot_position_comparison(ax, position_data, coloring)
        title_suffix = f"Position Sweep (step={step_size})"

    fig.suptitle(
        f"Activation Patching [{component}]: Denoising vs Noising {title_suffix}",
        fontsize=18,
        fontweight="bold",
    )

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
    recoveries = [layer_data[lyr].recovery for lyr in layers]
    disruptions = [layer_data[lyr].disruption for lyr in layers]

    # Viridis colormap for layers
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))

    # Build label indices: extreme points + evenly spaced
    label_indices = _get_extreme_indices(recoveries, disruptions, n_top=5)
    label_indices.add(0)  # First layer
    label_indices.add(len(layers) - 1)  # Last layer
    label_indices |= _get_evenly_spaced_indices(len(layers), n_labels=8)

    # Plot points (skip if either value is None)
    for i, (rec, dis, layer) in enumerate(zip(recoveries, disruptions, layers)):
        if rec is None or dis is None:
            continue
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

    # Build label indices: extreme points + evenly spaced
    label_indices = _get_extreme_indices(recoveries, disruptions, n_top=5)
    label_indices.add(0)  # First position
    label_indices.add(len(positions) - 1)  # Last position
    label_indices |= _get_evenly_spaced_indices(len(positions), n_labels=10)

    # Plot points (skip if either value is None)
    for i, (rec, dis, pos) in enumerate(zip(recoveries, disruptions, positions)):
        if rec is None or dis is None:
            continue
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
    recoveries: list[float | None],
    disruptions: list[float | None],
    n_top: int = 5,
) -> set[int]:
    """Get indices of extreme points in recovery and disruption.

    Handles None values by treating them as -inf for sorting purposes.
    """
    # Replace None with -inf so they sort to the end (we want highest values)
    rec_arr = np.array([r if r is not None else -np.inf for r in recoveries])
    dis_arr = np.array([d if d is not None else -np.inf for d in disruptions])
    rec_sorted_idx = np.argsort(rec_arr)[::-1][:n_top]
    dis_sorted_idx = np.argsort(dis_arr)[::-1][:n_top]
    return set(rec_sorted_idx) | set(dis_sorted_idx)


def _get_evenly_spaced_indices(n_total: int, n_labels: int = 8) -> set[int]:
    """Get evenly spaced indices for labeling.

    Distributes labels evenly across the index range.
    """
    if n_total <= n_labels:
        return set(range(n_total))
    step = (n_total - 1) / (n_labels - 1)
    return {int(round(i * step)) for i in range(n_labels)}


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
    # Add faint background region labels (before other elements)
    # "AND" above x=y line: both denoising and noising have strong effect
    ax.text(
        0.25, 0.75, "AND",
        fontsize=48, fontweight="bold", color="gray", alpha=0.15,
        ha="center", va="center", zorder=0,
    )
    # "OR" below x=y line: one or the other has effect
    ax.text(
        0.75, 0.25, "OR",
        fontsize=48, fontweight="bold", color="gray", alpha=0.15,
        ha="center", va="center", zorder=0,
    )

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
