"""Redundancy map visualization.

Shows Disruption - Recovery across layers and positions.
- Positive: AND behavior (component critical when corrupted)
- Negative: OR behavior (component helpful when restored)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ....activation_patching.coarse import SweepStepResults
from ....viz.token_coloring import PairTokenColoring
from .helpers import get_tick_color, setup_grid


def plot_redundancy(
    layer_data: SweepStepResults | None,
    position_data: SweepStepResults | None,
    output_dir: Path,
    step_size: int = 1,
    coloring: PairTokenColoring | None = None,
    component: str = "resid_post",
) -> None:
    """Plot redundancy maps (Disruption - Recovery) for layer and position sweeps.

    Creates a two-panel figure showing redundancy metric:
    - Left: Layer sweep redundancy
    - Right: Position sweep redundancy

    Args:
        layer_data: Layer sweep results
        position_data: Position sweep results
        output_dir: Directory to save output
        step_size: Step size used in the sweep
        coloring: Token coloring for position tick colors
        component: Component being patched (for plot title)
    """
    has_layer = bool(layer_data)
    has_position = bool(position_data)

    if not has_layer and not has_position:
        return

    # Determine layout based on available data
    if has_layer and has_position:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")
        _plot_layer_redundancy(axes[0], layer_data)
        _plot_position_redundancy(axes[1], position_data, coloring)
    elif has_layer:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
        _plot_layer_redundancy(ax, layer_data)
    else:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
        _plot_position_redundancy(ax, position_data, coloring)

    fig.suptitle(
        f"Redundancy Map [{component}]: Disruption - Recovery (step={step_size})",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = output_dir / f"redundancy_{step_size}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def _plot_layer_redundancy(
    ax: plt.Axes,
    layer_data: SweepStepResults | None,
) -> None:
    """Plot layer sweep redundancy as bar chart."""
    ax.set_facecolor("white")

    if not layer_data:
        ax.text(0.5, 0.5, "No layer results", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return

    layers = sorted(layer_data.keys())
    redundancies = []
    for layer in layers:
        rec = layer_data[layer].recovery
        dis = layer_data[layer].disruption
        if rec is not None and dis is not None:
            redundancies.append(dis - rec)
        else:
            redundancies.append(None)

    # Create bar colors based on sign (red for AND/positive, blue for OR/negative)
    colors = []
    for r in redundancies:
        if r is None:
            colors.append("gray")
        elif r > 0:
            colors.append("#d62728")  # Red for AND
        else:
            colors.append("#1f77b4")  # Blue for OR

    # Filter valid data for plotting
    valid_layers = []
    valid_redundancies = []
    valid_colors = []
    for layer, red, col in zip(layers, redundancies, colors):
        if red is not None:
            valid_layers.append(layer)
            valid_redundancies.append(red)
            valid_colors.append(col)

    if valid_layers:
        ax.bar(valid_layers, valid_redundancies, color=valid_colors, alpha=0.8, edgecolor="black")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Disruption - Recovery", fontsize=12, fontweight="bold")
    ax.set_title("Layer Sweep Redundancy", fontsize=14, fontweight="bold")
    setup_grid(ax)

    # Add region labels
    ax.text(
        0.02, 0.98, "AND (+)",
        transform=ax.transAxes, fontsize=10, fontweight="bold",
        color="#d62728", alpha=0.7, ha="left", va="top",
    )
    ax.text(
        0.02, 0.02, "OR (-)",
        transform=ax.transAxes, fontsize=10, fontweight="bold",
        color="#1f77b4", alpha=0.7, ha="left", va="bottom",
    )


def _plot_position_redundancy(
    ax: plt.Axes,
    position_data: SweepStepResults | None,
    coloring: PairTokenColoring | None = None,
) -> None:
    """Plot position sweep redundancy as bar chart."""
    ax.set_facecolor("white")

    if not position_data:
        ax.text(0.5, 0.5, "No position results", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return

    positions = sorted(position_data.keys())
    redundancies = []
    for pos in positions:
        rec = position_data[pos].recovery
        dis = position_data[pos].disruption
        if rec is not None and dis is not None:
            redundancies.append(dis - rec)
        else:
            redundancies.append(None)

    # Create bar colors based on sign
    colors = []
    for r in redundancies:
        if r is None:
            colors.append("gray")
        elif r > 0:
            colors.append("#d62728")  # Red for AND
        else:
            colors.append("#1f77b4")  # Blue for OR

    # Filter valid data for plotting
    valid_positions = []
    valid_redundancies = []
    valid_colors = []
    for pos, red, col in zip(positions, redundancies, colors):
        if red is not None:
            valid_positions.append(pos)
            valid_redundancies.append(red)
            valid_colors.append(col)

    if valid_positions:
        ax.bar(valid_positions, valid_redundancies, color=valid_colors, alpha=0.8, edgecolor="black")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
    ax.set_xlabel("Position", fontsize=12, fontweight="bold")
    ax.set_ylabel("Disruption - Recovery", fontsize=12, fontweight="bold")
    ax.set_title("Position Sweep Redundancy", fontsize=14, fontweight="bold")
    setup_grid(ax)

    # Color x-axis tick labels by token type if coloring available
    if coloring:
        for tick_label, pos in zip(ax.get_xticklabels(), valid_positions):
            tick_label.set_color(get_tick_color(pos, coloring))

    # Add region labels
    ax.text(
        0.02, 0.98, "AND (+)",
        transform=ax.transAxes, fontsize=10, fontweight="bold",
        color="#d62728", alpha=0.7, ha="left", va="top",
    )
    ax.text(
        0.02, 0.02, "OR (-)",
        transform=ax.transAxes, fontsize=10, fontweight="bold",
        color="#1f77b4", alpha=0.7, ha="left", va="bottom",
    )
