"""Visualization for tokenization alignment in contrastive pairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from ...common import profile
from ...common.base_schema import BaseSchema
from ...common.contrastive_pair import ContrastivePair
from ...common.file_io import load_json, save_json
from ...viz.plot_helpers import finalize_plot as _finalize_plot
from ...viz.viz_palettes import TOKEN_COLORS
from ...viz.token_coloring import (
    TokenColorInfo,
    PairTokenColoring,
)


TOKENIZATION_CACHE_FILENAME = "tokenization_viz_cache.json"


@dataclass
class TokenizationVizData(BaseSchema):
    """Cached data for tokenization visualization.

    Stores all data needed to render the tokenization visualization
    without requiring a model/tokenizer.
    """

    # Token IDs for each trajectory
    clean_token_ids: list[int]
    corrupted_token_ids: list[int]

    # Decoded token strings (the key data that requires the tokenizer)
    clean_tokens: list[str]
    corrupted_tokens: list[str]

    # Labels
    clean_label: str
    corrupted_label: str

    # Prompt lengths
    clean_prompt_len: int
    corrupted_prompt_len: int

    # Divergent positions (optional)
    choice_divergent_positions: tuple[int, int] | None = None

    @classmethod
    def from_pair_and_runner(
        cls, pair: ContrastivePair, runner: Any
    ) -> "TokenizationVizData":
        """Create visualization data from a ContrastivePair and runner.

        This is the only place that requires the model/tokenizer.
        """
        clean_tokens = [
            runner.decode_ids([tid]) for tid in pair.clean_traj.token_ids
        ]
        corrupted_tokens = [
            runner.decode_ids([tid]) for tid in pair.corrupted_traj.token_ids
        ]

        return cls(
            clean_token_ids=list(pair.clean_traj.token_ids),
            corrupted_token_ids=list(pair.corrupted_traj.token_ids),
            clean_tokens=clean_tokens,
            corrupted_tokens=corrupted_tokens,
            clean_label=pair.clean_labels[0] if pair.clean_labels else "?",
            corrupted_label=pair.clean_labels[1] if pair.clean_labels else "?",
            clean_prompt_len=pair.clean_prompt_length,
            corrupted_prompt_len=pair.corrupted_prompt_length,
            choice_divergent_positions=pair.choice_divergent_positions,
        )

    def get_coloring(self) -> PairTokenColoring:
        """Compute token coloring from cached data."""
        return _compute_coloring_from_viz_data(self)

    def save(self, output_dir: Path) -> None:
        """Save visualization cache to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.to_dict(), output_dir / TOKENIZATION_CACHE_FILENAME)

    @classmethod
    def load(cls, output_dir: Path) -> "TokenizationVizData | None":
        """Load visualization cache from directory, or None if not found."""
        cache_path = Path(output_dir) / TOKENIZATION_CACHE_FILENAME
        if not cache_path.exists():
            return None
        return cls.from_dict(load_json(cache_path))

    @classmethod
    def from_token_tree(cls, output_dir: Path) -> "TokenizationVizData | None":
        """Create minimal TokenizationVizData from token_tree.json as fallback.

        This is used when tokenization_viz_cache.json is missing but token_tree.json
        exists. The resulting data will have token IDs and prompt lengths but
        placeholder strings for decoded tokens.

        Args:
            output_dir: Directory containing token_tree.json

        Returns:
            TokenizationVizData with minimal data, or None if token_tree.json not found
        """
        token_tree_path = Path(output_dir) / "token_tree.json"
        if not token_tree_path.exists():
            return None

        try:
            tree_data = load_json(token_tree_path)
            trajs = tree_data.get("trajs", [])
            if len(trajs) < 2:
                return None

            # In token_tree.json, trajectories are stored with analysis data
            # that includes trunk_last_idx (which is the last prompt token index)
            clean_traj = trajs[0]
            corrupted_traj = trajs[1]

            clean_token_ids = clean_traj.get("token_ids", [])
            corrupted_token_ids = corrupted_traj.get("token_ids", [])

            if not clean_token_ids or not corrupted_token_ids:
                return None

            # Extract prompt lengths from analysis.trunk_last_idx + 1
            # trunk_last_idx is the index of the last prompt token
            clean_analysis = clean_traj.get("analysis", {})
            corrupted_analysis = corrupted_traj.get("analysis", {})

            clean_prompt_len = clean_analysis.get("trunk_last_idx", 0) + 1
            corrupted_prompt_len = corrupted_analysis.get("trunk_last_idx", 0) + 1

            # Use placeholder strings for tokens (we don't have decoded text)
            clean_tokens = [""] * len(clean_token_ids)
            corrupted_tokens = [""] * len(corrupted_token_ids)

            # Use generic labels since we don't have them
            clean_label = "short"
            corrupted_label = "long"

            return cls(
                clean_token_ids=clean_token_ids,
                corrupted_token_ids=corrupted_token_ids,
                clean_tokens=clean_tokens,
                corrupted_tokens=corrupted_tokens,
                clean_label=clean_label,
                corrupted_label=corrupted_label,
                clean_prompt_len=clean_prompt_len,
                corrupted_prompt_len=corrupted_prompt_len,
                choice_divergent_positions=None,  # Not available in token_tree.json
            )
        except Exception:
            # If anything fails, return None to allow other fallbacks
            return None


def _compute_coloring_from_viz_data(data: TokenizationVizData) -> PairTokenColoring:
    """Compute PairTokenColoring from TokenizationVizData.

    This replicates the logic from get_token_coloring_for_pair but works
    with the cached data structure.
    """
    from ...viz.token_coloring import _build_position_colors

    clean_ids = data.clean_token_ids
    corrupted_ids = data.corrupted_token_ids
    clean_prompt_len = data.clean_prompt_len
    corrupted_prompt_len = data.corrupted_prompt_len

    # Find first contrastive divergent position in prompt and response
    min_prompt_len = min(clean_prompt_len, corrupted_prompt_len)

    # First divergent in prompt region
    first_prompt_div = None
    for j in range(min_prompt_len):
        if clean_ids[j] != corrupted_ids[j]:
            first_prompt_div = j
            break

    # First divergent in response region (same RELATIVE position)
    clean_response_len = len(clean_ids) - clean_prompt_len
    corrupted_response_len = len(corrupted_ids) - corrupted_prompt_len
    min_response_len = min(clean_response_len, corrupted_response_len)

    first_response_div_offset = None
    for k in range(min_response_len):
        clean_resp_idx = clean_prompt_len + k
        corrupted_resp_idx = corrupted_prompt_len + k
        if clean_ids[clean_resp_idx] != corrupted_ids[corrupted_resp_idx]:
            first_response_div_offset = k
            break

    # Convert to absolute positions
    clean_first_response_div = (
        clean_prompt_len + first_response_div_offset
        if first_response_div_offset is not None
        else None
    )
    corrupted_first_response_div = (
        corrupted_prompt_len + first_response_div_offset
        if first_response_div_offset is not None
        else None
    )

    # Choice divergent positions
    choice_div_clean = None
    choice_div_corrupted = None
    if data.choice_divergent_positions:
        choice_div_clean, choice_div_corrupted = data.choice_divergent_positions

    # Build color dictionaries
    clean_colors = _build_position_colors(
        n_tokens=len(clean_ids),
        prompt_len=clean_prompt_len,
        choice_div_pos=choice_div_clean,
        first_prompt_div=first_prompt_div,
        first_response_div=clean_first_response_div,
    )

    corrupted_colors = _build_position_colors(
        n_tokens=len(corrupted_ids),
        prompt_len=corrupted_prompt_len,
        choice_div_pos=choice_div_corrupted,
        first_prompt_div=first_prompt_div,
        first_response_div=corrupted_first_response_div,
    )

    return PairTokenColoring(
        clean_colors=clean_colors,
        corrupted_colors=corrupted_colors,
        clean_prompt_len=clean_prompt_len,
        corrupted_prompt_len=corrupted_prompt_len,
    )


@profile
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
        # Create viz data (decodes tokens) and save cache
        viz_data = TokenizationVizData.from_pair_and_runner(pair, runner)
        viz_data.save(output_dir)

        # Get coloring info
        coloring = viz_data.get_coloring()

        # Use simple index if multiple pairs, otherwise no suffix
        suffix = f"_{i}" if max_pairs > 1 else ""
        _plot_tokenization_from_data(
            viz_data, coloring, output_dir / f"tokenization{suffix}.png"
        )

    print(f"[viz] Tokenization plots saved to {output_dir}")


@profile
def visualize_tokenization_from_cache(output_dir: Path) -> bool:
    """Visualize tokenization from cached data.

    Args:
        output_dir: Directory containing the cache file

    Returns:
        True if visualization was successful, False if cache not found
    """
    viz_data = TokenizationVizData.load(output_dir)
    if viz_data is None:
        return False

    coloring = viz_data.get_coloring()
    _plot_tokenization_from_data(
        viz_data, coloring, Path(output_dir) / "tokenization.png"
    )
    return True


def _plot_tokenization_from_data(
    viz_data: TokenizationVizData,
    coloring: PairTokenColoring,
    save_path: Path,
) -> None:
    """Plot detailed tokenization from cached visualization data.

    Args:
        viz_data: TokenizationVizData with all needed info
        coloring: PairTokenColoring with color info
        save_path: Path to save the plot
    """
    clean_ids = viz_data.clean_token_ids
    corrupted_ids = viz_data.corrupted_token_ids

    # Create figure with detailed layout - size based on sequence length
    max_len = max(len(clean_ids), len(corrupted_ids))
    fig_height = max(14, min(32, 3 + (max_len // 15) * 0.8))
    fig = plt.figure(figsize=(20, fig_height))

    # Info panel at top
    ax_info = fig.add_axes([0.05, 0.92, 0.9, 0.06])
    ax_info.axis("off")

    clean_label = viz_data.clean_label
    corrupted_label = viz_data.corrupted_label

    info_text = (
        f"Clean label: {clean_label}    |    Corrupted label: {corrupted_label}    |    "
        f"Prompt tokens: {coloring.clean_prompt_len}/{coloring.corrupted_prompt_len}    |    "
        f"Lengths: {len(clean_ids)}/{len(corrupted_ids)}"
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

    # Clean trajectory
    ax_clean = fig.add_axes([0.02, 0.48, 0.88, 0.42])
    _plot_token_grid(
        ax_clean,
        clean_ids,
        viz_data.clean_tokens,
        coloring.clean_colors,
        f"Clean trajectory (chose {clean_label}, rejected {corrupted_label})",
    )

    # Corrupted trajectory
    ax_corrupted = fig.add_axes([0.02, 0.02, 0.88, 0.42])
    _plot_token_grid(
        ax_corrupted,
        corrupted_ids,
        viz_data.corrupted_tokens,
        coloring.corrupted_colors,
        f"Corrupted trajectory (chose {corrupted_label}, rejected {clean_label})",
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
