"""Visualization for tokenization alignment in contrastive pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from ...common.contrastive_pair import ContrastivePair
from ...viz.plot_helpers import finalize_plot as _finalize_plot
from ...viz.palettes import TOKEN_COLORS
from ...viz.token_coloring import (
    TokenColorInfo,
    PairTokenColoring,
    get_token_coloring_for_pair,
)


def visualize_tokenization(
    pairs: list[ContrastivePair],
    runner: Any,
    output_dir: Path,
    max_pairs: int = 3,
) -> None:
    """Visualize tokenization alignment for contrastive pairs.

    Creates visualizations showing:
    - Token ID to text mapping for each position
    - Prompt token count boundary
    - Divergent regions between short/long trajectories

    Args:
        pairs: List of ContrastivePair objects
        runner: Model runner with tokenizer
        output_dir: Directory to save plots
        max_pairs: Maximum number of pairs to visualize
    """
    if not pairs:
        print("[viz] No pairs to visualize tokenization")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, pair in enumerate(pairs[:max_pairs]):
        # Decode tokens
        short_tokens = [runner.decode_ids([tid]) for tid in pair.short_traj.token_ids]
        long_tokens = [runner.decode_ids([tid]) for tid in pair.long_traj.token_ids]

        # Get coloring info
        coloring = get_token_coloring_for_pair(pair)

        # Use simple index if multiple pairs, otherwise no suffix
        suffix = f"_{i}" if max_pairs > 1 else ""
        _plot_tokenization_detail(
            pair, coloring, short_tokens, long_tokens, output_dir / f"tokenization{suffix}.png"
        )

    print(f"[viz] Tokenization plots saved to {output_dir}")


def _plot_tokenization_detail(
    pair: ContrastivePair,
    coloring: PairTokenColoring,
    short_tokens: list[str],
    long_tokens: list[str],
    save_path: Path,
) -> None:
    """Plot detailed tokenization for a contrastive pair.

    Args:
        pair: ContrastivePair with token IDs and labels
        coloring: PairTokenColoring with color info
        short_tokens: Decoded token strings for short trajectory
        long_tokens: Decoded token strings for long trajectory
        save_path: Path to save the plot
    """
    short_ids = pair.short_traj.token_ids
    long_ids = pair.long_traj.token_ids

    # Create figure with detailed layout - size based on sequence length
    max_len = max(len(short_ids), len(long_ids))
    fig_height = max(14, min(32, 3 + (max_len // 15) * 0.8))
    fig = plt.figure(figsize=(20, fig_height))

    # Info panel at top
    ax_info = fig.add_axes([0.05, 0.92, 0.9, 0.06])
    ax_info.axis("off")

    # Get labels
    short_term_label = pair.short_label or "?"
    long_term_label = pair.long_label or "?"

    info_text = (
        f"Short-term label: {short_term_label}    |    Long-term label: {long_term_label}    |    "
        f"Prompt tokens: {coloring.short_prompt_len}/{coloring.long_prompt_len}    |    "
        f"Lengths: {len(short_ids)}/{len(long_ids)}"
    )
    ax_info.text(
        0.5,
        0.5,
        info_text,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    # Short trajectory - leave space on right for legend
    ax_short = fig.add_axes([0.02, 0.48, 0.88, 0.42])
    _plot_token_grid(
        ax_short,
        short_ids,
        short_tokens,
        coloring.short_colors,
        f"Short-term chooser (chose {short_term_label}, rejected {long_term_label})",
    )

    # Long trajectory
    ax_long = fig.add_axes([0.02, 0.02, 0.88, 0.42])
    _plot_token_grid(
        ax_long,
        long_ids,
        long_tokens,
        coloring.long_colors,
        f"Long-term chooser (chose {long_term_label}, rejected {short_term_label})",
    )

    _finalize_plot(save_path)


def _plot_token_grid(
    ax: plt.Axes,
    token_ids: list[int],
    tokens: list[str],
    colors: dict[int, TokenColorInfo],
    title: str,
    max_response_tokens: int = 100,
) -> None:
    """Plot token grid with IDs, text, and boundaries.

    Args:
        ax: Matplotlib axes to plot on
        token_ids: List of token IDs
        tokens: List of decoded token strings
        colors: Dict mapping position to TokenColorInfo
        title: Title for the plot
        max_response_tokens: Max response tokens to show
    """
    # Find prompt length from colors
    prompt_token_count = sum(1 for c in colors.values() if c.is_prompt)

    # Show ALL prompt tokens + max_response_tokens
    response_len = len(tokens) - prompt_token_count
    response_to_show = min(response_len, max_response_tokens)
    n_tokens = prompt_token_count + response_to_show

    # Layout
    tokens_per_row = 15
    n_rows = (n_tokens + tokens_per_row - 1) // tokens_per_row

    ax.set_xlim(-0.5, tokens_per_row - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    for i in range(n_tokens):
        row = i // tokens_per_row
        col = i % tokens_per_row

        # Get color info from dict
        color_info = colors.get(i)
        if color_info is None:
            # Fallback
            facecolor = "#E8F5E9" if i < prompt_token_count else "#E3F2FD"
            edgecolor = "#388E3C" if i < prompt_token_count else "#1976D2"
            linewidth = 1.5
            is_choice_div = False
            is_contrastive_div = False
        else:
            facecolor = color_info.facecolor
            edgecolor = color_info.edgecolor
            linewidth = color_info.linewidth
            is_choice_div = color_info.is_choice_divergent
            is_contrastive_div = color_info.is_contrastive_divergent

        # Draw token box
        rect = mpatches.FancyBboxPatch(
            (col - 0.45, row - 0.4),
            0.9,
            0.8,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        ax.add_patch(rect)

        # Token text (escape special chars)
        token_text = tokens[i].replace("\n", "\\n").replace("\t", "\\t")

        # Adaptive font size based on text length
        if len(token_text) > 12:
            token_text = token_text[:10] + ".."
            fontsize = 5
        elif len(token_text) > 8:
            fontsize = 5.5
        elif len(token_text) > 5:
            fontsize = 6
        else:
            fontsize = 7

        ax.text(
            col,
            row - 0.08,
            token_text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontfamily="monospace",
            fontweight="bold",
        )

        # Token ID
        ax.text(
            col,
            row + 0.22,
            f"id:{token_ids[i]}",
            ha="center",
            va="center",
            fontsize=5,
            color="gray",
        )

        # Position number - color based on type
        if is_choice_div and is_contrastive_div:
            pos_color = "#D32F2F"  # Red (both)
            pos_weight = "bold"
        elif is_choice_div:
            pos_color = "#7B1FA2"  # Purple
            pos_weight = "bold"
        elif is_contrastive_div:
            pos_color = "#D32F2F"  # Red
            pos_weight = "bold"
        else:
            pos_color = "darkgray"
            pos_weight = "normal"

        ax.text(
            col - 0.35,
            row - 0.32,
            str(i),
            ha="left",
            va="center",
            fontsize=5,
            color=pos_color,
            fontweight=pos_weight,
        )

    # Legend - place outside plot area
    legend_elements = [
        mpatches.Patch(facecolor=TOKEN_COLORS["prompt_light"], edgecolor=TOKEN_COLORS["prompt_edge"], label="Prompt"),
        mpatches.Patch(facecolor=TOKEN_COLORS["response_light"], edgecolor=TOKEN_COLORS["response_edge"], label="Response"),
        mpatches.Patch(facecolor=TOKEN_COLORS["choice_div_light"], edgecolor=TOKEN_COLORS["choice_div_edge"], label="Choice Div"),
        mpatches.Patch(facecolor=TOKEN_COLORS["contrast_div_light"], edgecolor=TOKEN_COLORS["contrast_div_edge"], label="Contrastive Div"),
        mpatches.Patch(facecolor=TOKEN_COLORS["choice_div_light"], edgecolor=TOKEN_COLORS["contrast_div_edge"], linewidth=2, label="Both"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=7,
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
    )
