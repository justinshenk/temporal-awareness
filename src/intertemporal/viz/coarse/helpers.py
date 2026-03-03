"""Helper functions for coarse patching visualization.

Tick coloring, spacing, save utilities.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ....viz.palettes import TOKEN_COLORS
from ....viz.token_coloring import PairTokenColoring


def get_tick_spacing(n_points: int) -> int:
    """Return tick spacing to keep x-axis legible."""
    if n_points <= 15:
        return 1
    elif n_points <= 30:
        return 2
    elif n_points <= 60:
        return 5
    return 10


def get_tick_color(pos: int, coloring: PairTokenColoring | None) -> str:
    """Get the color for a tick label at given position."""
    if coloring is None or not coloring.short_colors:
        return TOKEN_COLORS["response_edge"]

    color_info = coloring.short_colors.get(pos)
    if color_info is None:
        for offset in range(20):
            if pos + offset in coloring.short_colors:
                color_info = coloring.short_colors[pos + offset]
                break
            if pos - offset in coloring.short_colors:
                color_info = coloring.short_colors[pos - offset]
                break

    return color_info.edgecolor if color_info else TOKEN_COLORS["response_edge"]


def color_xaxis_ticks(
    ax: plt.Axes,
    positions: list[int],
    coloring: PairTokenColoring | None,
) -> None:
    """Color x-axis tick labels by token type."""
    colors = [get_tick_color(pos, coloring) for pos in positions]
    ax.set_xticks(positions)
    ax.set_xticklabels([str(p) for p in positions], fontsize=10, fontweight="bold")
    for label, color in zip(ax.get_xticklabels(), colors):
        label.set_color(color)


def save_with_colored_ticks(
    fig: plt.Figure,
    ax: plt.Axes,
    positions: list[int],
    coloring: PairTokenColoring | None,
    save_path: Path,
) -> None:
    """Save figure with colored x-axis tick labels.

    Forces a canvas draw before setting colors to ensure tick labels exist.
    """
    ax.set_xticks(positions)
    ax.set_xticklabels([str(p) for p in positions], fontsize=11, fontweight="bold")

    fig.canvas.draw()

    colors = [get_tick_color(pos, coloring) for pos in positions]
    for label, color in zip(ax.get_xticklabels(), colors):
        label.set_color(color)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def save_with_colored_ticks_multi(
    fig: plt.Figure,
    axes: list[plt.Axes],
    positions: list[int],
    coloring: PairTokenColoring | None,
    save_path: Path,
) -> None:
    """Save figure with colored x-axis tick labels on multiple axes."""
    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels([str(p) for p in positions], fontsize=10, fontweight="bold")

    fig.canvas.draw()

    colors = [get_tick_color(pos, coloring) for pos in positions]

    for ax in axes:
        for label, color in zip(ax.get_xticklabels(), colors):
            label.set_color(color)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def add_token_type_legend(fig: plt.Figure) -> None:
    """Add a small legend for token type colors at the bottom."""
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0], [0],
            marker="s", color="w",
            markerfacecolor=TOKEN_COLORS["prompt"],
            markeredgecolor=TOKEN_COLORS["prompt_edge"],
            markersize=10, markeredgewidth=2,
            label="Prompt",
        ),
        Line2D(
            [0], [0],
            marker="s", color="w",
            markerfacecolor=TOKEN_COLORS["response"],
            markeredgecolor=TOKEN_COLORS["response_edge"],
            markersize=10, markeredgewidth=2,
            label="Response",
        ),
        Line2D(
            [0], [0],
            marker="s", color="w",
            markerfacecolor=TOKEN_COLORS["choice_div"],
            markeredgecolor=TOKEN_COLORS["choice_div_edge"],
            markersize=10, markeredgewidth=2,
            label="Choice Div",
        ),
        Line2D(
            [0], [0],
            marker="s", color="w",
            markerfacecolor=TOKEN_COLORS["contrast_div"],
            markeredgecolor=TOKEN_COLORS["contrast_div_edge"],
            markersize=10, markeredgewidth=2,
            label="Contrast Div",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        fontsize=9,
        title="Tick colors by token type",
        title_fontsize=9,
        ncol=4,
        bbox_to_anchor=(0.42, -0.01),
        frameon=True,
        fancybox=True,
        shadow=False,
    )


def finalize_plot(fig: plt.Figure, output_path: Path) -> None:
    """Save figure with standard formatting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
