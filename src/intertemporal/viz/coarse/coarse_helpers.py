"""Helper functions for coarse patching visualization.

Tick coloring, spacing, save utilities.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from ....viz.viz_palettes import TOKEN_COLORS
from ....viz.token_coloring import PairTokenColoring
from .coarse_colors import VLINE_COLORS


def setup_grid(ax: plt.Axes) -> None:
    """Set up granular grid with both major and minor gridlines."""
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.5, linewidth=0.6, color="#AAAAAA")
    ax.grid(True, which="minor", alpha=0.3, linewidth=0.3, color="#CCCCCC")
    ax.set_axisbelow(True)


def get_tick_spacing(n_points: int, max_ticks: int = 15) -> int:
    """Return tick spacing to keep x-axis legible.

    Adaptively computes spacing to show at most max_ticks labels.
    """
    if n_points <= max_ticks:
        return 1
    # Compute spacing to get ~max_ticks labels
    spacing = (n_points + max_ticks - 1) // max_ticks
    # Round up to nice numbers for readability
    if spacing <= 2:
        return 2
    elif spacing <= 5:
        return 5
    elif spacing <= 10:
        return 10
    elif spacing <= 20:
        return 20
    elif spacing <= 25:
        return 25
    elif spacing <= 50:
        return 50
    return 100


def get_tick_color(pos: int, coloring: PairTokenColoring | None) -> str:
    """Get the color for a tick label at given position."""
    if coloring is None or not coloring.clean_colors:
        return TOKEN_COLORS["response_edge"]

    color_info = coloring.clean_colors.get(pos)
    if color_info is None:
        for offset in range(20):
            if pos + offset in coloring.clean_colors:
                color_info = coloring.clean_colors[pos + offset]
                break
            if pos - offset in coloring.clean_colors:
                color_info = coloring.clean_colors[pos - offset]
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
    ax.set_xticklabels(
        [str(p) for p in positions],
        fontsize=9,
        fontweight="bold",
        rotation=45,
        ha="right",
    )
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
    """Add a legend for token type colors at the bottom left."""
    # Use edge colors for squares to match tick label colors
    legend_elements = [
        Line2D(
            [0], [0],
            marker="s", color="w",
            markerfacecolor=TOKEN_COLORS["prompt_edge"],
            markeredgecolor=TOKEN_COLORS["prompt_edge"],
            markersize=10, markeredgewidth=2,
            label="Prompt",
        ),
        Line2D(
            [0], [0],
            marker="s", color="w",
            markerfacecolor=TOKEN_COLORS["response_edge"],
            markeredgecolor=TOKEN_COLORS["response_edge"],
            markersize=10, markeredgewidth=2,
            label="Response",
        ),
        Line2D(
            [0], [0],
            marker="s", color="w",
            markerfacecolor=TOKEN_COLORS["choice_div_edge"],
            markeredgecolor=TOKEN_COLORS["choice_div_edge"],
            markersize=10, markeredgewidth=2,
            label="Choice Div",
        ),
        Line2D(
            [0], [0],
            marker="s", color="w",
            markerfacecolor=TOKEN_COLORS["contrast_div_edge"],
            markeredgecolor=TOKEN_COLORS["contrast_div_edge"],
            markersize=10, markeredgewidth=2,
            label="Contrast Div",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=9,
        title="Token Position Types",
        title_fontsize=9,
        ncol=4,
        bbox_to_anchor=(0.10, 0.02),
        frameon=True,
        fancybox=True,
        shadow=False,
    )


def add_boundary_legend(fig: plt.Figure, y_offset: float = -0.06) -> None:
    """Add a separate legend for boundary markers (vertical lines) at bottom right."""
    legend_elements = [
        Line2D(
            [0], [0],
            color=VLINE_COLORS["prompt_boundary"],
            linewidth=2.5,
            linestyle="--",
            label="Prompt End",
        ),
        Line2D(
            [0], [0],
            color=VLINE_COLORS["choice_div_pos"],
            linewidth=2.5,
            linestyle=":",
            label="Choice Div Pos",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9,
        title="Context Boundaries",
        title_fontsize=9,
        ncol=2,
        bbox_to_anchor=(0.90, 0.02),
        frameon=True,
        fancybox=True,
        shadow=False,
    )


def add_xaxis_boundary_markers(
    ax: plt.Axes,
    prompt_boundary: int | None,
    choice_div_pos: int | None,
) -> None:
    """Add vertical line markers with triangle knicks at top/bottom edges for boundaries."""
    marker_size = 10
    # Transform: x in data coords, y in axes coords (0=bottom, 1=top)
    trans = blended_transform_factory(ax.transData, ax.transAxes)

    if prompt_boundary is not None:
        color = VLINE_COLORS["prompt_boundary"]
        # Transparent vertical line
        ax.axvline(
            prompt_boundary,
            color=color,
            linewidth=1.5, linestyle="--", alpha=0.2, zorder=5,
        )
        # Triangle knicks at top and bottom edges of plot box
        ax.plot(
            prompt_boundary, 0, marker="^", color=color, transform=trans,
            markersize=marker_size, clip_on=False, zorder=10,
            markeredgecolor="#222222", markeredgewidth=0.8,
        )
        ax.plot(
            prompt_boundary, 1, marker="v", color=color, transform=trans,
            markersize=marker_size, clip_on=False, zorder=10,
            markeredgecolor="#222222", markeredgewidth=0.8,
        )

    if choice_div_pos is not None:
        color = VLINE_COLORS["choice_div_pos"]
        # Transparent vertical line
        ax.axvline(
            choice_div_pos,
            color=color,
            linewidth=1.5, linestyle=":", alpha=0.2, zorder=5,
        )
        # Triangle knicks at top and bottom edges of plot box
        ax.plot(
            choice_div_pos, 0, marker="^", color=color, transform=trans,
            markersize=marker_size, clip_on=False, zorder=10,
            markeredgecolor="#222222", markeredgewidth=0.8,
        )
        ax.plot(
            choice_div_pos, 1, marker="v", color=color, transform=trans,
            markersize=marker_size, clip_on=False, zorder=10,
            markeredgecolor="#222222", markeredgewidth=0.8,
        )


def finalize_plot(fig: plt.Figure, output_path: Path) -> None:
    """Save figure with standard formatting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
