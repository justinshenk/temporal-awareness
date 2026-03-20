"""Base utilities for column plotting.

Provides common functions and styles used across all column types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt

from ..coarse_colors import LINE_STYLES, LINE_WIDTHS, MARKERS, MARKER_SIZES, METRIC_COLORS
from ..coarse_helpers import setup_grid


@dataclass
class LineConfig:
    """Configuration for a single line in a plot."""

    values: Sequence[float]
    label: str
    color_key: str
    markevery_offset: int = 0
    markevery_step: int = 2
    hollow_marker: bool = False

    @property
    def color(self) -> str:
        return METRIC_COLORS[self.color_key]

    @property
    def linestyle(self) -> str:
        return LINE_STYLES.get(self.color_key, "-")

    @property
    def linewidth(self) -> float:
        return LINE_WIDTHS.get(self.color_key, 2.0)

    @property
    def marker(self) -> str:
        return MARKERS.get(self.color_key, "o")

    @property
    def markersize(self) -> float:
        return MARKER_SIZES.get(self.color_key, 8)


def plot_line(
    ax: plt.Axes,
    x_values: Sequence[int | float],
    config: LineConfig,
) -> None:
    """Plot a single line with consistent styling.

    Args:
        ax: Matplotlib axes to plot on
        x_values: X-axis values (layers or positions)
        config: Line configuration including values, label, and style keys
    """
    kwargs = {
        "linestyle": config.linestyle,
        "color": config.color,
        "linewidth": config.linewidth,
        "marker": config.marker,
        "markersize": config.markersize,
        "markevery": (config.markevery_offset, config.markevery_step),
        "label": config.label,
    }

    if config.hollow_marker:
        kwargs["markerfacecolor"] = "white"
        kwargs["markeredgecolor"] = config.color
        kwargs["markeredgewidth"] = 2
    else:
        kwargs["markerfacecolor"] = config.color

    ax.plot(x_values, config.values, **kwargs)


def add_dual_axis_legend(
    ax_primary: plt.Axes,
    ax_secondary: plt.Axes,
    fontsize: int = 9,
) -> None:
    """Add combined legend for dual-axis plot below the axes.

    Uses 2-3 columns for a balanced layout (not too wide, not too tall).

    Args:
        ax_primary: Primary (left) y-axis
        ax_secondary: Secondary (right) y-axis
        fontsize: Legend font size
    """
    lines1, labels1 = ax_primary.get_legend_handles_labels()
    lines2, labels2 = ax_secondary.get_legend_handles_labels()
    n_items = len(lines1) + len(lines2)
    # Use 2-3 columns for balanced layout
    ncol = min(3, max(2, (n_items + 1) // 2))
    ax_primary.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=fontsize,
        ncol=ncol,
        frameon=True,
        fancybox=True,
        columnspacing=0.4,
        handletextpad=0.2,
        handlelength=1.5,
    )


def setup_column(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    tick_positions: Sequence[int],
    ylim: tuple[float, float] | None = None,
) -> None:
    """Apply standard column setup.

    Args:
        ax: Matplotlib axes
        title: Column title
        xlabel: X-axis label
        ylabel: Y-axis label (primary)
        tick_positions: X-axis tick positions
        ylim: Optional y-axis limits
    """
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=16)
    if ylabel:  # Only set if not empty (allows pre-set colored labels)
        ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13)

    # tick_positions already subsampled by get_tick_spacing() in sweep_plots.py
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(t) for t in tick_positions])

    if ylim:
        ax.set_ylim(ylim)
    setup_grid(ax)
