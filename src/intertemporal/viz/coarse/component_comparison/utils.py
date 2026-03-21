"""Shared utilities for component comparison visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None  # type: ignore


def setup_grid(ax: Axes) -> None:
    """Configure grid styling for an axes."""
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)


def save_plot(fig: plt.Figure, path: Path, name: str) -> None:
    """Save figure and print confirmation."""
    output_path = path / name
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def adjust_labels(texts: list, ax: Axes) -> None:
    """Adjust text labels to avoid overlap if adjustText is available."""
    if adjust_text is not None and texts:
        try:
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.4, lw=0.3)
            )
        except Exception:
            pass


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray | Axes]:
    """Create figure with white background."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="white")
    if isinstance(axes, np.ndarray):
        for ax in axes.flatten():
            ax.set_facecolor("white")
    else:
        axes.set_facecolor("white")
    return fig, axes


def get_sqrt_colors(indices: list[int], cmap: str = "viridis") -> np.ndarray:
    """Get colors with sqrt mapping for more resolution at higher values."""
    if not indices:
        return np.array([])
    min_idx, max_idx = min(indices), max(indices)
    if max_idx > min_idx:
        normalized = np.array([(idx - min_idx) / (max_idx - min_idx) for idx in indices])
        color_values = np.sqrt(normalized)
    else:
        color_values = np.zeros(len(indices))
    return color_values
