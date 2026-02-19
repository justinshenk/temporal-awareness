"""Visualization for tokenization alignment in contrastive pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from ...common.contrastive_pair import ContrastivePair
from ...viz.layer_position_heatmaps import _finalize_plot


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
        _plot_tokenization_detail(
            pair, runner, output_dir / f"tokenization_pair_{i}.png"
        )

    print(f"[viz] Tokenization plots saved to {output_dir}")


def _plot_tokenization_detail(
    pair: ContrastivePair,
    runner: Any,
    save_path: Path,
) -> None:
    """Plot detailed tokenization for a contrastive pair.

    Shows token IDs, decoded text, and marks prompt boundary.
    Colors:
    - Purple: choice divergent position (where A vs B diverge in original binary choice)
    - Red: contrastive divergent positions (where short_traj vs long_traj differ)
    - Purple+Red border: position is both choice and contrastive divergent
    """
    short_ids = pair.short_traj.token_ids
    long_ids = pair.long_traj.token_ids

    short_prompt_len = pair.short_prompt_length
    long_prompt_len = pair.long_prompt_length

    # Decode tokens (include special tokens for visualization)
    short_tokens = [runner.decode_ids([tid]) for tid in short_ids]
    long_tokens = [runner.decode_ids([tid]) for tid in long_ids]

    # Find FIRST contrastive divergent position in prompt and response
    # Prompt: compare at same absolute positions (both start at 0)
    # Response: compare at same RELATIVE positions (offset from each trajectory's prompt end)

    min_prompt_len = min(short_prompt_len, long_prompt_len)

    # First divergent in prompt region (same absolute position)
    first_prompt_div = None
    for j in range(min_prompt_len):
        if short_ids[j] != long_ids[j]:
            first_prompt_div = j
            break

    # First divergent in response region (same RELATIVE position within response)
    short_response_len = len(short_ids) - short_prompt_len
    long_response_len = len(long_ids) - long_prompt_len
    min_response_len = min(short_response_len, long_response_len)

    first_response_div_offset = None  # Relative offset from response start
    for k in range(min_response_len):
        short_resp_idx = short_prompt_len + k
        long_resp_idx = long_prompt_len + k
        if short_ids[short_resp_idx] != long_ids[long_resp_idx]:
            first_response_div_offset = k
            break

    # Convert to absolute positions for each trajectory
    short_first_prompt_div = first_prompt_div
    long_first_prompt_div = first_prompt_div
    short_first_response_div = short_prompt_len + first_response_div_offset if first_response_div_offset is not None else None
    long_first_response_div = long_prompt_len + first_response_div_offset if first_response_div_offset is not None else None

    # Get choice divergent positions (where A vs B diverge)
    choice_div_short = None
    choice_div_long = None
    if pair.choice_divergent_positions:
        choice_div_short, choice_div_long = pair.choice_divergent_positions

    # Create figure with detailed layout - size based on sequence length
    max_len = max(len(short_ids), len(long_ids))
    fig_height = max(14, min(32, 3 + (max_len // 15) * 0.8))
    fig = plt.figure(figsize=(20, fig_height))

    # Info panel at top
    ax_info = fig.add_axes([0.05, 0.92, 0.9, 0.06])
    ax_info.axis("off")

    # Get labels - pair.labels is (short_term_label, long_term_label)
    short_term_label = pair.short_label or "?"
    long_term_label = pair.long_label or "?"

    info_text = (
        f"Short-term label: {short_term_label}    |    Long-term label: {long_term_label}    |    "
        f"Prompt tokens: {short_prompt_len}/{long_prompt_len}    |    "
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
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    # Short trajectory - leave space on right for legend
    ax_short = fig.add_axes([0.02, 0.48, 0.88, 0.42])
    _plot_token_grid(
        ax_short,
        short_ids,
        short_tokens,
        short_prompt_len,
        f"Short-term chooser (chose {short_term_label}, rejected {long_term_label})",
        choice_divergent_pos=choice_div_short,
        first_prompt_divergent=short_first_prompt_div,
        first_response_divergent=short_first_response_div,
    )

    # Long trajectory
    ax_long = fig.add_axes([0.02, 0.02, 0.88, 0.42])
    _plot_token_grid(
        ax_long,
        long_ids,
        long_tokens,
        long_prompt_len,
        f"Long-term chooser (chose {long_term_label}, rejected {short_term_label})",
        choice_divergent_pos=choice_div_long,
        first_prompt_divergent=long_first_prompt_div,
        first_response_divergent=long_first_response_div,
    )

    _finalize_plot(save_path)


def _plot_token_grid(
    ax: plt.Axes,
    token_ids: list[int],
    tokens: list[str],
    prompt_token_count: int,
    title: str,
    choice_divergent_pos: int | None = None,
    first_prompt_divergent: int | None = None,
    first_response_divergent: int | None = None,
    max_response_tokens: int = 100,
) -> None:
    """Plot token grid with IDs, text, and boundaries.

    Colors:
    - Green: prompt tokens
    - Blue: response tokens
    - Purple: choice divergent position (where A vs B diverge)
    - Red: first contrastive divergent in prompt and response (2 positions max)
    """
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

    # Contrastive divergent positions (at most 2: first in prompt, first in response)
    contrastive_div_positions = set()
    if first_prompt_divergent is not None:
        contrastive_div_positions.add(first_prompt_divergent)
    if first_response_divergent is not None:
        contrastive_div_positions.add(first_response_divergent)

    for i in range(n_tokens):
        row = i // tokens_per_row
        col = i % tokens_per_row

        is_choice_div = choice_divergent_pos is not None and i == choice_divergent_pos
        is_contrastive_div = i in contrastive_div_positions

        # Determine color based on position type
        if is_choice_div and is_contrastive_div:
            # Both: purple fill with red border
            facecolor = "#E1BEE7"  # Light purple
            edgecolor = "#D32F2F"  # Red border
            linewidth = 3.0
        elif is_choice_div:
            # Purple: choice divergent position (A vs B)
            facecolor = "#E1BEE7"  # Light purple
            edgecolor = "#7B1FA2"  # Purple
            linewidth = 1.5
        elif is_contrastive_div:
            # Red: first contrastive divergent (short vs long trajectory)
            facecolor = "#FFCDD2"  # Light red
            edgecolor = "#D32F2F"  # Red
            linewidth = 1.5
        elif i < prompt_token_count:
            facecolor = "#E8F5E9"  # Light green for prompt
            edgecolor = "#388E3C"
            linewidth = 1.5
        else:
            facecolor = "#E3F2FD"  # Light blue for response
            edgecolor = "#1976D2"
            linewidth = 1.5

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
        mpatches.Patch(facecolor="#E8F5E9", edgecolor="#388E3C", label="Prompt"),
        mpatches.Patch(facecolor="#E3F2FD", edgecolor="#1976D2", label="Response"),
        mpatches.Patch(facecolor="#E1BEE7", edgecolor="#7B1FA2", label="Choice Div"),
        mpatches.Patch(facecolor="#FFCDD2", edgecolor="#D32F2F", label="Contrastive Div"),
        mpatches.Patch(facecolor="#E1BEE7", edgecolor="#D32F2F", linewidth=2, label="Both"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=7,
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
    )
