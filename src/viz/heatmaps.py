"""Heatmap visualizations for layer x position analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..intertemporal.analysis.section_markers import SECTION_COLORS


@dataclass
class HeatmapConfig:
    """Configuration for heatmap plots."""

    title: str = "Heatmap"
    subtitle: str | None = None
    cbar_label: str = "Value"
    cmap: str = "RdYlGn"
    vmin: float | None = None
    vmax: float | None = None
    annotate: bool = True
    figsize: tuple[float, float] | None = None
    max_labels: int = 40


def plot_layer_position_heatmap(
    matrix: np.ndarray,
    layers: list[int],
    position_labels: list[str],
    save_path: Path | None = None,
    config: HeatmapConfig | None = None,
    section_markers: dict[str, int] | None = None,
) -> None:
    """Plot heatmap of values across layers and token positions.

    Args:
        matrix: 2D array of shape (n_layers, n_positions)
        layers: Layer indices for y-axis labels
        position_labels: Labels for x-axis
        save_path: If provided, save to file; otherwise show on screen
        config: HeatmapConfig with styling options
        section_markers: Dict of {section_name: position} for vertical markers
    """
    if config is None:
        config = HeatmapConfig()

    n_layers, n_positions = matrix.shape

    # Auto-disable annotation for large matrices
    annotate = config.annotate and (n_layers * n_positions <= 100)

    fig, ax = plt.subplots(figsize=_compute_figsize(config.figsize, n_layers, n_positions))

    vmin, vmax = _compute_range(matrix, config.vmin, config.vmax, config.cmap)
    im = _draw_heatmap(ax, matrix, config.cmap, vmin, vmax)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(config.cbar_label, rotation=270, labelpad=20, fontsize=10)

    # Y-axis: layers
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=9)

    # X-axis: positions
    _setup_position_axis(ax, position_labels, config.max_labels)

    ax.set_xlabel("Token Position", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    _set_title(ax, config.title, config.subtitle)

    if section_markers:
        _draw_section_markers(ax, section_markers, n_positions)

    if annotate:
        _annotate_cells(ax, matrix, vmin, vmax)

    _finalize_plot(save_path)


def plot_position_sweep(
    values: np.ndarray,
    position_labels: list[str],
    save_path: Path | None = None,
    config: HeatmapConfig | None = None,
    section_markers: dict[str, int] | None = None,
) -> None:
    """Plot position sweep as single-row heatmap.

    Args:
        values: 1D array of values per position
        position_labels: Labels for x-axis
        save_path: If provided, save to file; otherwise show on screen
        config: HeatmapConfig with styling options
        section_markers: Dict of {section_name: position} for vertical markers
    """
    if config is None:
        config = HeatmapConfig(vmin=0.0, vmax=1.0)

    n_positions = len(values)

    # Compact figure for single row
    fig_width = min(24, max(10, n_positions * 0.15))
    fig, ax = plt.subplots(figsize=(fig_width, 3))

    matrix = values.reshape(1, -1)
    vmin = config.vmin if config.vmin is not None else 0.0
    vmax = config.vmax if config.vmax is not None else 1.0

    im = _draw_heatmap(ax, matrix, config.cmap, vmin, vmax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(config.cbar_label, fontsize=9)

    _setup_position_axis(ax, position_labels, config.max_labels)
    ax.set_yticks([0])
    ax.set_yticklabels(["All Layers"], fontsize=9)

    ax.set_xlabel("Token Position", fontsize=10)
    ax.set_title(config.title, fontsize=11)

    if section_markers:
        _draw_section_markers(ax, section_markers, n_positions, y_offset=-1.2)

    _finalize_plot(save_path)


def plot_layer_sweep(
    values: np.ndarray,
    layers: list[int],
    save_path: Path | None = None,
    config: HeatmapConfig | None = None,
) -> None:
    """Plot layer sweep as bar chart.

    Args:
        values: 1D array of values per layer
        layers: Layer indices
        save_path: If provided, save to file; otherwise show on screen
        config: HeatmapConfig with styling options
    """
    if config is None:
        config = HeatmapConfig(title="Layer Sweep", cbar_label="Recovery")

    fig, ax = plt.subplots(figsize=config.figsize or (12, 6))

    x = np.arange(len(layers))
    bars = ax.bar(x, values, color="steelblue", alpha=0.8)

    # Color bars by value
    max_val = max(values) if len(values) > 0 else 1.0
    for bar, val in zip(bars, values):
        intensity = val / max_val if max_val > 0 else 0
        bar.set_color(plt.cm.RdYlGn(0.3 + 0.6 * intensity))

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel(config.cbar_label, fontsize=11)
    ax.set_title(config.title, fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.set_ylim(bottom=0)

    _finalize_plot(save_path)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _compute_figsize(
    figsize: tuple[float, float] | None,
    n_layers: int,
    n_positions: int,
) -> tuple[float, float]:
    """Compute appropriate figure size for heatmap."""
    if figsize is not None:
        return figsize
    fig_height = min(16, max(6, n_layers * 0.5 + 2))
    fig_width = min(24, max(10, n_positions * 0.8))
    return (fig_width, fig_height)


def _compute_range(
    matrix: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    cmap: str,
) -> tuple[float, float]:
    """Compute value range, centering diverging colormaps at 0."""
    if vmin is None:
        vmin = float(np.nanmin(matrix))
    if vmax is None:
        vmax = float(np.nanmax(matrix))

    # Center diverging colormaps at 0
    if any(c in cmap for c in ("RdBu", "coolwarm", "bwr")):
        if vmin < 0 < vmax:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max

    return vmin, vmax


def _draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
):
    """Draw heatmap with NaN handling."""
    masked_matrix = np.ma.masked_invalid(matrix)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgray")
    return ax.imshow(
        masked_matrix, aspect="auto", cmap=cmap_obj,
        vmin=vmin, vmax=vmax, origin="lower"
    )


def _setup_position_axis(
    ax: plt.Axes,
    position_labels: list[str],
    max_labels: int,
) -> None:
    """Setup x-axis with position labels, subsampling if needed."""
    n_positions = len(position_labels)

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


def _set_title(ax: plt.Axes, title: str, subtitle: str | None) -> None:
    """Set plot title with optional subtitle."""
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    else:
        ax.set_title(title, fontsize=12)


def _draw_section_markers(
    ax: plt.Axes,
    section_markers: dict[str, int],
    n_positions: int,
    y_offset: float = -1.5,
) -> None:
    """Draw vertical section markers with labels."""
    for name, pos in section_markers.items():
        if 0 <= pos < n_positions:
            color = SECTION_COLORS.get(name, "gray")
            x_pos = pos + 0.5
            ax.axvline(x=x_pos, color=color, linestyle="--", linewidth=1.5, alpha=0.8)

            label = name.replace("before_", "").replace("_", " ")
            ax.annotate(
                label, xy=(x_pos, -0.5), xytext=(x_pos, y_offset),
                fontsize=7, color="white", ha="center", va="top",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor=color,
                    edgecolor="none", alpha=0.95
                ),
                annotation_clip=False,
            )


def _annotate_cells(
    ax: plt.Axes,
    matrix: np.ndarray,
    vmin: float,
    vmax: float,
) -> None:
    """Add value annotations to heatmap cells."""
    n_layers, n_positions = matrix.shape
    midpoint = (vmin + vmax) / 2

    for i in range(n_layers):
        for j in range(n_positions):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < midpoint else "black"
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center",
                    color=text_color, fontsize=7, fontweight="bold"
                )


def _finalize_plot(save_path: Path | None) -> None:
    """Finalize plot: save or show."""
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()
