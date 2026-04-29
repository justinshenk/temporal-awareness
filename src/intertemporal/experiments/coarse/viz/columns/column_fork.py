"""Fork metrics column: entropy, diversity, simpson, TCB.

Shows information-theoretic metrics about the fork (divergence point)
with TCB on the secondary axis.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt

from ......activation_patching import IntervenedChoiceMetrics
from ..coarse_colors import METRIC_COLORS
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
    """Plot fork metrics column.

    Primary axis (left): entropy, diversity, simpson
    Secondary axis (right): TCB

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
    fork_entropies = [m.fork_entropy for m in metrics]
    fork_diversities = [m.fork_diversity for m in metrics]
    fork_simpsons = [m.fork_simpson for m in metrics]
    vocab_tcbs = [m.vocab_tcb for m in metrics]

    # Primary axis: entropy, diversity, simpson
    plot_line(
        ax,
        x_values,
        LineConfig(
            values=fork_entropies,
            label="entropy",
            color_key="fork_entropy",
            markevery_offset=0,
        ),
    )
    plot_line(
        ax,
        x_values,
        LineConfig(
            values=fork_diversities,
            label="diversity",
            color_key="fork_diversity",
            markevery_offset=1,
        ),
    )
    plot_line(
        ax,
        x_values,
        LineConfig(
            values=fork_simpsons,
            label="simpson",
            color_key="fork_simpson",
            markevery_offset=2,
        ),
    )
    ax.set_ylabel(
        "Entropy / Effective #",
        fontsize=16,
        color=METRIC_COLORS["fork_entropy"],
        fontweight="bold",
    )
    ax.tick_params(
        axis="y", labelcolor=METRIC_COLORS["fork_entropy"], labelsize=13
    )
    setup_column(ax, "Fork", xlabel, "", tick_positions, ylim=(-0.05, 1.1))

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

    add_dual_axis_legend(ax, ax_right, fontsize=legend_fontsize, show=show_legend)

    return ax_right
