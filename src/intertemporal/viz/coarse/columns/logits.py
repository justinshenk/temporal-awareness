"""Logits column: raw logits, normalized logits, relative delta.

Shows logit values for each choice option with normalized variants.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt

from .....activation_patching import IntervenedChoiceMetrics
from .base import LineConfig, add_dual_axis_legend, plot_line, setup_column


def plot(
    ax: plt.Axes,
    x_values: Sequence[int],
    metrics: list[IntervenedChoiceMetrics],
    tick_positions: Sequence[int],
    xlabel: str = "Layer",
) -> plt.Axes | None:
    """Plot logits column.

    Primary axis (left): logit(short), logit(long)
    Secondary axis (right): rel_logit_delta, norm_logit(short), norm_logit(long)

    If no valid logit data, shows "No logit data" message.

    Args:
        ax: Matplotlib axes
        x_values: X-axis values (layers or positions)
        metrics: List of IntervenedChoiceMetrics for each x value
        tick_positions: X-axis tick positions
        xlabel: X-axis label

    Returns:
        Secondary (right) y-axis for synchronization, or None if no valid data
    """
    # Extract metric arrays
    logit_shorts = [m.logit_short for m in metrics]
    logit_longs = [m.logit_long for m in metrics]
    norm_logit_shorts = [m.norm_logit_short for m in metrics]
    norm_logit_longs = [m.norm_logit_long for m in metrics]
    rel_logit_deltas = [m.rel_logit_delta for m in metrics]

    # Check if logits have valid data
    has_valid_logits = any(v != 0.0 for v in logit_shorts + logit_longs)

    if not has_valid_logits:
        ax.text(
            0.5,
            0.5,
            "No logit data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=18,
        )
        setup_column(ax, "Logits", xlabel, "", tick_positions)
        return None

    # Primary axis: raw logits
    plot_line(
        ax,
        x_values,
        LineConfig(
            values=logit_shorts,
            label="logit(short)",
            color_key="logit_short",
            markevery_offset=0,
        ),
    )
    plot_line(
        ax,
        x_values,
        LineConfig(
            values=logit_longs,
            label="logit(long)",
            color_key="logit_long",
            markevery_offset=1,
        ),
    )
    setup_column(ax, "Logits", xlabel, "Raw Logit", tick_positions)

    # Secondary axis: normalized and relative metrics (3-line stagger)
    ax_right = ax.twinx()
    plot_line(
        ax_right,
        x_values,
        LineConfig(
            values=rel_logit_deltas,
            label="rel_logit_delta",
            color_key="rel_logit_delta",
            markevery_offset=0,
            markevery_step=3,
        ),
    )
    plot_line(
        ax_right,
        x_values,
        LineConfig(
            values=norm_logit_shorts,
            label="norm_logit(short)",
            color_key="norm_logit_short",
            markevery_offset=1,
            markevery_step=3,
            hollow_marker=True,
        ),
    )
    plot_line(
        ax_right,
        x_values,
        LineConfig(
            values=norm_logit_longs,
            label="norm_logit(long)",
            color_key="norm_logit_long",
            markevery_offset=2,
            markevery_step=3,
            hollow_marker=True,
        ),
    )
    ax_right.set_ylabel("Normalized", fontsize=16)
    ax_right.tick_params(axis="y", labelsize=13)
    ax_right.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Auto-scale y-axis to show rel_logit_delta variation
    if rel_logit_deltas:
        rld_min, rld_max = min(rel_logit_deltas), max(rel_logit_deltas)
        rld_range = rld_max - rld_min
        if rld_range > 0:
            padding = rld_range * 0.2
            ax_right.set_ylim(rld_min - padding, rld_max + padding)

    add_dual_axis_legend(ax, ax_right)

    return ax_right
