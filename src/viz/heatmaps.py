"""Heatmap visualizations for layer x position analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.markers import SECTION_COLORS

# Pastel background colors for section labels
SECTION_BG_COLORS = {
    "before_choices": "#FF8C00",       # Dark orange
    "before_time_horizon": "#00CED1",  # Dark cyan
    "before_choice_output": "#32CD32",  # Lime green
}


def plot_layer_position_heatmap(
    matrix: np.ndarray,
    layers: list[int],
    position_labels: list[str],
    save_path: Path,
    title: str = "Activation Patching",
    subtitle: Optional[str] = None,
    cbar_label: str = "Effect",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdYlGn",
    annotate: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    section_markers: Optional[dict[str, int]] = None,
) -> None:
    """Plot heatmap of values across layers and token positions."""
    n_layers, n_positions = matrix.shape

    # Auto-disable annotation for large matrices
    if annotate and (n_layers * n_positions > 100):
        annotate = False

    # Calculate figure size
    if figsize is None:
        fig_height = max(6, n_layers * 0.5 + 2)
        fig_width = max(10, n_positions * 0.8)
        fig_width = min(fig_width, 24)
        fig_height = min(fig_height, 16)
    else:
        fig_width, fig_height = figsize

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Auto-detect vmin/vmax
    if vmin is None:
        vmin = np.nanmin(matrix)
    if vmax is None:
        vmax = np.nanmax(matrix)

    # Center diverging colormaps at 0
    if "RdBu" in cmap or "coolwarm" in cmap or "bwr" in cmap:
        if vmin < 0 < vmax:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max

    # Use masked array to handle NaN values - they will be shown in gray
    masked_matrix = np.ma.masked_invalid(matrix)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgray")
    im = ax.imshow(masked_matrix, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax, origin="lower")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=10)

    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=9)

    # X-axis: show every Nth label if too many positions
    max_labels = 40
    if n_positions > max_labels:
        step = (n_positions + max_labels - 1) // max_labels
        tick_positions = list(range(0, n_positions, step))
        tick_labels = [position_labels[i] for i in tick_positions]
    else:
        tick_positions = list(range(n_positions))
        tick_labels = position_labels

    ax.set_xticks(tick_positions)
    x_fontsize = max(6, min(10, 400 // len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=x_fontsize)

    ax.set_xlabel("Token Position", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)

    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    else:
        ax.set_title(title, fontsize=12)

    # Add section markers: vertical lines with labels just above x-axis
    if section_markers:
        for name, pos in section_markers.items():
            if 0 <= pos < n_positions:
                color = SECTION_COLORS.get(name, "gray")
                # Line at RIGHT edge of bin (+0.5)
                x_pos = pos + 0.5
                ax.axvline(x=x_pos, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
                # Label just above x-axis (y=-0.5 in data coords, below layer 0)
                label = name.replace("before_", "").replace("_", " ")
                ax.annotate(label, xy=(x_pos, -0.5), xytext=(x_pos, -1.5),
                            fontsize=7, color="white", ha="center", va="top",
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor=color,
                                      edgecolor="none", alpha=0.95),
                            annotation_clip=False)

    if annotate:
        for i in range(n_layers):
            for j in range(n_positions):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < (vmin + vmax) / 2 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color=text_color, fontsize=7, fontweight="bold")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")


def plot_position_sweep(
    values: np.ndarray,
    position_labels: list[str],
    save_path: Path,
    title: str = "Position Sweep",
    cbar_label: str = "Recovery",
    vmin: float = 0.0,
    vmax: float = 1.0,
    section_markers: Optional[dict[str, int]] = None,
) -> None:
    """Plot position sweep as single-row heatmap (same style as layer x position)."""
    n_positions = len(values)

    # Same width as other heatmaps
    fig_width = max(10, n_positions * 0.15)
    fig_width = min(fig_width, 24)
    fig_height = 3

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Reshape to single row for imshow
    matrix = values.reshape(1, -1)

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, orientation="vertical")
    cbar.set_label(cbar_label, fontsize=9)

    # X-axis: positions
    max_labels = 40
    if n_positions > max_labels:
        step = (n_positions + max_labels - 1) // max_labels
        tick_positions = list(range(0, n_positions, step))
        tick_labels = [position_labels[i] for i in tick_positions]
    else:
        tick_positions = list(range(n_positions))
        tick_labels = position_labels

    ax.set_xticks(tick_positions)
    x_fontsize = max(6, min(9, 400 // len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=x_fontsize)
    ax.set_yticks([0])
    ax.set_yticklabels(["All Layers"], fontsize=9)

    ax.set_xlabel("Token Position", fontsize=10)
    ax.set_title(title, fontsize=11)

    # Section markers as vertical lines with labels just above x-axis
    if section_markers:
        for name, pos in section_markers.items():
            if 0 <= pos < n_positions:
                color = SECTION_COLORS.get(name, "gray")
                x_pos = pos + 0.5
                ax.axvline(x=x_pos, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
                label = name.replace("before_", "").replace("_", " ")
                ax.annotate(label, xy=(x_pos, -0.5), xytext=(x_pos, -1.2),
                            fontsize=7, color="white", ha="center", va="top",
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor=color,
                                      edgecolor="none", alpha=0.95),
                            annotation_clip=False)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")
