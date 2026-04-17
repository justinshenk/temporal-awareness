"""Trajectory column: inverse perplexity.

Shows trajectory-level metrics for model confidence.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt

from ......activation_patching import IntervenedChoiceMetrics
from .column_base import LineConfig, add_dual_axis_legend, plot_line, setup_column


def plot(
    ax: plt.Axes,
    x_values: Sequence[int],
    metrics: list[IntervenedChoiceMetrics],
    tick_positions: Sequence[int],
    xlabel: str = "Layer",
    show_legend: bool = True,
    legend_fontsize: int = 9,
) -> plt.Axes:
    """Plot trajectory metrics column.

    Primary axis (left): inv_perplexity(short), inv_perplexity(long)

    Args:
        ax: Matplotlib axes
        x_values: X-axis values (layers or positions)
        metrics: List of IntervenedChoiceMetrics for each x value
        tick_positions: X-axis tick positions
        xlabel: X-axis label
        show_legend: Whether to show the legend
        legend_fontsize: Font size for legend

    Returns:
        Secondary (right) y-axis for synchronization
    """
    # Extract metric arrays
    inv_perps_short = [m.traj_inv_perplexity_short for m in metrics]
    inv_perps_long = [m.traj_inv_perplexity_long for m in metrics]

    # Primary axis: inverse perplexity
    plot_line(
        ax,
        x_values,
        LineConfig(
            values=inv_perps_short,
            label="inv_ppl(short)",
            color_key="inv_perplexity_short",
            markevery_offset=0,
        ),
    )
    plot_line(
        ax,
        x_values,
        LineConfig(
            values=inv_perps_long,
            label="inv_ppl(long)",
            color_key="inv_perplexity_long",
            markevery_offset=1,
        ),
    )

    setup_column(ax, "Trajectory", xlabel, "Inv Perplexity", tick_positions)

    # No secondary axis for this column
    ax_right = ax.twinx()
    ax_right.set_yticks([])
    ax_right.set_ylabel("")

    add_dual_axis_legend(ax, ax_right, fontsize=legend_fontsize, show=show_legend)

    return ax_right
