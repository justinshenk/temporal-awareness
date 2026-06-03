"""Shared utilities for component comparison visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from ......viz.plot_helpers import save_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None  # type: ignore


def setup_grid(ax: Axes, integer_x_minor: bool = True) -> None:
    """Configure grid styling for an axes with minor ticks for full resolution.

    Args:
        ax: Matplotlib axes
        integer_x_minor: If True, minor x-axis ticks are placed at integer positions only
    """
    from matplotlib.ticker import MultipleLocator

    ax.minorticks_on()
    # Force x-axis minor ticks to integer positions only (e.g., tick at 23 between 22 and 24)
    if integer_x_minor:
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, which="major", alpha=0.5, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.25, linewidth=0.3)
    ax.set_axisbelow(True)


def save_plot(fig: plt.Figure, path: Path, name: str) -> None:
    """Save figure and print confirmation."""
    output_path = path / name
    save_figure(fig, output_path, dpi=150)
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
