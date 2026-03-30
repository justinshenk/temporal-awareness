"""Core metrics column: recovery/disruption, logit_diff, reciprocal_rank.

This is the primary column showing the main intervention effect metrics.
"""

from __future__ import annotations

from typing import Literal, Sequence

import matplotlib.pyplot as plt

from .....activation_patching import IntervenedChoiceMetrics
from ..coarse_colors import METRIC_COLORS
from .column_base import LineConfig, add_dual_axis_legend, plot_line, setup_column


def plot(
    ax: plt.Axes,
    x_values: Sequence[int],
    metrics: list[IntervenedChoiceMetrics],
    mode: Literal["denoising", "noising"],
    tick_positions: Sequence[int],
    xlabel: str = "Layer",
) -> plt.Axes:
    """Plot core metrics column.

    Primary axis (left): recovery/disruption, reciprocal_rank
    Secondary axis (right): logit_diff, norm_logit_diff

    Args:
        ax: Matplotlib axes
        x_values: X-axis values (layers or positions)
        metrics: List of IntervenedChoiceMetrics for each x value
        mode: "denoising" or "noising" determines metric labels
        tick_positions: X-axis tick positions
        xlabel: X-axis label

    Returns:
        Secondary (right) y-axis for synchronization
    """
    # Extract metric arrays - use mode-aware effect metrics (target - source semantics)
    effect_values = [m.effect for m in metrics]
    effect_label = "recovery" if mode == "denoising" else "disruption"
    rr_values = [m.effect_reciprocal_rank for m in metrics]
    rr_label = "recip_rank(target)"
    logit_diffs = [m.effect_logit_diff for m in metrics]
    norm_logit_diffs = [m.effect_norm_logit_diff for m in metrics]

    # Primary axis: recovery/disruption and reciprocal_rank
    plot_line(
        ax,
        x_values,
        LineConfig(
            values=effect_values,
            label=effect_label,
            color_key="recovery",
            markevery_offset=0,
        ),
    )
    plot_line(
        ax,
        x_values,
        LineConfig(
            values=rr_values,
            label=rr_label,
            color_key="rr_short",
            markevery_offset=1,
        ),
    )

    ylabel_left = "Recovery / RR" if mode == "denoising" else "Disruption / RR"
    setup_column(ax, "Core", xlabel, ylabel_left, tick_positions, ylim=(-0.1, 1.5))

    # Secondary axis: logit differences
    ax_right = ax.twinx()
    plot_line(
        ax_right,
        x_values,
        LineConfig(
            values=logit_diffs,
            label="logit_diff",
            color_key="logit_diff",
            markevery_offset=0,
        ),
    )
    plot_line(
        ax_right,
        x_values,
        LineConfig(
            values=norm_logit_diffs,
            label="norm_logit_diff",
            color_key="norm_logit_diff",
            markevery_offset=1,
        ),
    )
    ax_right.set_ylabel("Logit Diff", fontsize=16, color=METRIC_COLORS["logit_diff"])
    ax_right.tick_params(axis="y", labelcolor=METRIC_COLORS["logit_diff"], labelsize=13)
    ax_right.axhline(y=0, color="gray", linestyle="-", alpha=0.5, linewidth=1)

    add_dual_axis_legend(ax, ax_right)

    return ax_right
