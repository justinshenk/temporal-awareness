"""Probability/logprob column: prob(short), prob(long), logprob(short), logprob(long).

Shows the raw probability and log-probability values for each choice option.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt

from .....activation_patching import IntervenedChoiceMetrics
from .column_base import LineConfig, add_dual_axis_legend, plot_line, setup_column


def plot(
    ax: plt.Axes,
    x_values: Sequence[int],
    metrics: list[IntervenedChoiceMetrics],
    tick_positions: Sequence[int],
    xlabel: str = "Layer",
) -> plt.Axes:
    """Plot probability/logprob column.

    Primary axis (left): prob(short), prob(long)
    Secondary axis (right): logprob(short), logprob(long)

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
    prob_shorts = [m.prob_short for m in metrics]
    prob_longs = [m.prob_long for m in metrics]
    logprob_shorts = [m.logprob_short for m in metrics]
    logprob_longs = [m.logprob_long for m in metrics]

    # Primary axis: probabilities
    plot_line(
        ax,
        x_values,
        LineConfig(
            values=prob_shorts,
            label="prob(short)",
            color_key="prob_short",
            markevery_offset=0,
            hollow_marker=True,
        ),
    )
    plot_line(
        ax,
        x_values,
        LineConfig(
            values=prob_longs,
            label="prob(long)",
            color_key="prob_long",
            markevery_offset=1,
            hollow_marker=True,
        ),
    )
    setup_column(
        ax, "Probs/Logprobs", xlabel, "Probability", tick_positions, ylim=(-0.05, 1.05)
    )

    # Secondary axis: logprobs
    ax_right = ax.twinx()
    plot_line(
        ax_right,
        x_values,
        LineConfig(
            values=logprob_shorts,
            label="logprob(short)",
            color_key="logprob_short",
            markevery_offset=0,
        ),
    )
    plot_line(
        ax_right,
        x_values,
        LineConfig(
            values=logprob_longs,
            label="logprob(long)",
            color_key="logprob_long",
            markevery_offset=1,
        ),
    )
    ax_right.set_ylabel("Logprob", fontsize=16)
    ax_right.tick_params(axis="y", labelsize=13)

    add_dual_axis_legend(ax, ax_right)

    return ax_right
