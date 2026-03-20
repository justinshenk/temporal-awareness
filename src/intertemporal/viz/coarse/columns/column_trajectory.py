"""Trajectory column: inverse perplexity and TCB.

Shows trajectory-level metrics for model confidence and behavior.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt

from .....activation_patching import IntervenedChoiceMetrics
from ..coarse_colors import METRIC_COLORS
from .column_base import LineConfig, add_dual_axis_legend, plot_line, setup_column


def plot(
    ax: plt.Axes,
    x_values: Sequence[int],
    metrics: list[IntervenedChoiceMetrics],
    tick_positions: Sequence[int],
    xlabel: str = "Layer",
) -> plt.Axes:
    """Plot trajectory metrics column.

    Primary axis (left): inv_perplexity(short), inv_perplexity(long)
    Secondary axis (right): TCB

    Args:
        ax: Matplotlib axes
        x_values: X-axis values (layers or positions)
        metrics: List of IntervenedChoiceMetrics for each x value
        tick_positions: X-axis tick positions
        xlabel: X-axis label

    Returns:
        Secondary (right) y-axis for synchronization
    """
    # Extract metric arrays
    inv_perps_short = [m.traj_inv_perplexity_short for m in metrics]
    inv_perps_long = [m.traj_inv_perplexity_long for m in metrics]
    vocab_tcbs = [m.vocab_tcb for m in metrics]

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
    setup_column(ax, "Trajectory + TCB", xlabel, "Inv Perplexity", tick_positions)

    # Secondary axis: TCB
    ax_right = ax.twinx()
    plot_line(
        ax_right,
        x_values,
        LineConfig(
            values=vocab_tcbs,
            label="TCB",
            color_key="vocab_tcb",
        ),
    )
    ax_right.set_ylabel(
        "TCB",
        fontsize=16,
        color=METRIC_COLORS["vocab_tcb"],
        fontweight="bold",
    )
    ax_right.tick_params(
        axis="y", labelcolor=METRIC_COLORS["vocab_tcb"], labelsize=13
    )

    add_dual_axis_legend(ax, ax_right)

    return ax_right
