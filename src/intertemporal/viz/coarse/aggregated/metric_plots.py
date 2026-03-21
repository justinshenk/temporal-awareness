"""Metric plotting for aggregated visualization.

Generates per-column plots with mean + spread visualization.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data_extraction import ColumnData
from .style import (
    AXIS_LABEL_FONTSIZE,
    DPI,
    GRID_ALPHA,
    GRID_LINE_WIDTH,
    MEAN_LINE_ALPHA,
    MEAN_LINE_WIDTH,
    MEAN_MARKER,
    MEAN_MARKER_SIZE,
    METRIC_DISPLAY_NAMES,
    PAIR_LINE_ALPHA,
    PAIR_LINE_WIDTH,
    SPREAD_ALPHA,
    SUBPLOT_HEIGHT,
    SUBPLOT_WIDTH,
    TICK_LABEL_FONTSIZE,
    TITLE_FONTSIZE,
    TITLE_FONTWEIGHT,
    get_metric_color,
)


def plot_column(
    column_data: ColumnData,
    output_path: Path,
    title_prefix: str,
) -> None:
    """Plot all metrics in a column as a horizontal row of subplots.

    Shows:
    - Shaded region for min/max spread across pairs
    - Individual pair lines (faint)
    - Bold mean line with markers

    Args:
        column_data: Extracted column data with metrics
        output_path: Path to save the PNG file
        title_prefix: Prefix for the figure title
    """
    metrics = column_data.metrics
    if not metrics:
        return

    n_metrics = len(metrics)
    fig_width = SUBPLOT_WIDTH * n_metrics
    fig_height = SUBPLOT_HEIGHT

    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=(fig_width, fig_height),
        facecolor="white",
        squeeze=False,
    )
    axes = axes[0]

    for idx, metric_series in enumerate(metrics):
        ax = axes[idx]
        ax.set_facecolor("white")

        x_values = metric_series.x_values
        if not x_values:
            continue

        # Get color for this metric
        color = get_metric_color(metric_series.metric_name)

        # Compute min/max and valid spread in single pass
        valid_spread_x = []
        valid_min = []
        valid_max = []
        for x_idx, x in enumerate(x_values):
            values = [
                s[x_idx] for s in metric_series.per_pair_series
                if s[x_idx] is not None
            ]
            if values:
                valid_spread_x.append(x)
                valid_min.append(min(values))
                valid_max.append(max(values))

        if valid_spread_x:
            ax.fill_between(
                valid_spread_x,
                valid_min,
                valid_max,
                color=color,
                alpha=SPREAD_ALPHA,
                linewidth=0,
                zorder=1,
            )

        # Plot individual pair lines
        for series in metric_series.per_pair_series:
            valid_pairs = [(x, y) for x, y in zip(x_values, series) if y is not None]
            if valid_pairs:
                valid_x, valid_y = zip(*valid_pairs)
                ax.plot(
                    valid_x,
                    valid_y,
                    color=color,
                    alpha=PAIR_LINE_ALPHA,
                    linewidth=PAIR_LINE_WIDTH,
                    zorder=2,
                )

        # Plot mean line
        if metric_series.mean_series:
            ax.plot(
                x_values,
                metric_series.mean_series,
                color=color,
                alpha=MEAN_LINE_ALPHA,
                linewidth=MEAN_LINE_WIDTH,
                zorder=3,
            )
            ax.scatter(
                x_values,
                metric_series.mean_series,
                marker=MEAN_MARKER,
                s=MEAN_MARKER_SIZE**2,
                facecolors="white",
                edgecolors=color,
                linewidths=1.5,
                zorder=4,
            )

        # Title
        metric_display = METRIC_DISPLAY_NAMES.get(
            metric_series.metric_name, metric_series.metric_name
        )
        ax.set_title(metric_display, fontsize=TITLE_FONTSIZE, fontweight=TITLE_FONTWEIGHT)

        # Grid
        ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH)
        ax.set_axisbelow(True)

        # Axis labels
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        x_label = "Layer" if "layer" in str(output_path).lower() else "Position"
        ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)

    # Figure title
    fig.suptitle(
        f"{title_prefix} - {column_data.column_name.title()}",
        fontsize=TITLE_FONTSIZE + 2,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
