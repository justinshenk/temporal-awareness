"""Vocab metrics column: entropy, diversity, simpson.

Shows information-theoretic metrics about the full vocabulary distribution.
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
    """Plot vocab metrics column.

    Primary axis (left): entropy
    Secondary axis (right): diversity, simpson

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
    vocab_entropies = [m.vocab_entropy for m in metrics]
    vocab_diversities = [m.vocab_diversity for m in metrics]
    vocab_simpsons = [m.vocab_simpson for m in metrics]

    # Primary axis: entropy
    plot_line(
        ax,
        x_values,
        LineConfig(
            values=vocab_entropies,
            label="entropy",
            color_key="vocab_entropy",
        ),
    )
    ax.set_ylabel(
        "Entropy (nats)",
        fontsize=16,
        color=METRIC_COLORS["vocab_entropy"],
        fontweight="bold",
    )
    ax.tick_params(
        axis="y", labelcolor=METRIC_COLORS["vocab_entropy"], labelsize=13
    )
    setup_column(ax, "Vocab", xlabel, "", tick_positions)

    # Secondary axis: diversity and simpson
    ax_right = ax.twinx()
    plot_line(
        ax_right,
        x_values,
        LineConfig(
            values=vocab_diversities,
            label="diversity",
            color_key="vocab_diversity",
            markevery_offset=0,
        ),
    )
    plot_line(
        ax_right,
        x_values,
        LineConfig(
            values=vocab_simpsons,
            label="simpson",
            color_key="vocab_simpson",
            markevery_offset=1,
        ),
    )
    ax_right.set_ylabel(
        "Effective #",
        fontsize=16,
        color=METRIC_COLORS["vocab_diversity"],
        fontweight="bold",
    )
    ax_right.tick_params(
        axis="y", labelcolor=METRIC_COLORS["vocab_diversity"], labelsize=13
    )

    add_dual_axis_legend(ax, ax_right)

    return ax_right
