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

# Color palette for format positions - distinct, visually appealing colors
# Pattern: marker=saturated, content=light, tail=medium (lighter than marker)
FORMAT_POS_COLORS = {
    # === SITUATION (orange spectrum) ===
    "situation_marker": "#E65100",   # Deep orange
    "situation_content": "#FFE0B2",  # Light orange
    "situation_tail": "#FF9800",     # Medium orange (lighter than marker)
    # === TASK (blue spectrum) ===
    "task_marker": "#1565C0",        # Strong blue
    "task_content": "#90CAF9",       # Light blue
    "task_tail": "#42A5F5",          # Medium blue (lighter than marker)
    # === CONSIDER (purple spectrum) ===
    "consider_marker": "#6A1B9A",    # Deep purple
    "consider_content": "#CE93D8",   # Light purple
    "consider_tail": "#AB47BC",      # Medium purple (lighter than marker)
    # Key variable positions in consider section
    "time_horizon": "#D32F2F",       # Red - IMPORTANT
    "post_time_horizon": "#EF9A9A",  # Light red
    # === ACTION (green spectrum) ===
    "action_marker": "#2E7D32",      # Forest green
    "action_content": "#A5D6A7",     # Light green
    "action_tail": "#66BB6A",        # Medium green (lighter than marker)
    # === FORMAT (brown spectrum) ===
    "format_marker": "#4E342E",      # Dark brown
    "format_content": "#BCAAA4",     # Light brown
    "format_tail": "#8D6E63",        # Medium brown (lighter than marker)
    "format_choice_prefix": "#AD1457",     # Deep pink
    "format_reasoning_prefix": "#00838F",  # Dark cyan
    # === OPTIONS (in task section) ===
    # Left option (olive-yellow spectrum)
    "left_label": "#558B2F",         # Olive green
    "left_reward": "#9E9D24",        # Yellow-green
    "left_time": "#F9A825",          # Amber
    # Right option (blue-teal spectrum)
    "right_label": "#0277BD",        # Light blue
    "right_reward": "#00897B",       # Teal
    "right_time": "#00695C",         # Dark teal
    # Options region
    "option_content": "#B2DFDB",     # Light teal (between option values)
    "options_tail": "#4DD0E1",        # Light cyan (lighter shade for options end)
    # === RESPONSE (purple-indigo spectrum) ===
    "response_choice_prefix": "#7B1FA2",   # Purple
    "response_choice": "#512DA8",          # Deep purple
    "response_reasoning_prefix": "#303F9F",  # Indigo
    "response_reasoning": "#7986CB",       # Light indigo
    # === CHAT (gray spectrum) ===
    "chat_prefix": "#546E7A",        # Blue-gray
    "chat_prefix_tail": "#90A4AE",   # Light blue-gray (lighter than prefix)
    "chat_suffix": "#37474F",        # Dark blue-gray
    "chat_suffix_tail": "#607D8B",   # Medium blue-gray (lighter than suffix)
    # === OTHER (neutral) ===
    "prompt_other": "#CFD8DC",       # Light gray
    "response_other": "#ECEFF1",     # Very light gray
}

# Define legend sections for organized display
LEGEND_SECTIONS = {
    "Situation": ["situation_marker", "situation_content", "situation_tail"],
    "Task": ["task_marker", "task_content", "task_tail"],
    "Consider": ["consider_marker", "consider_content", "consider_tail", "time_horizon", "post_time_horizon"],
    "Action": ["action_marker", "action_content", "action_tail"],
    "Format": ["format_marker", "format_content", "format_tail", "format_choice_prefix", "format_reasoning_prefix"],
    "Options": ["left_label", "left_reward", "left_time", "right_label", "right_reward", "right_time", "option_content", "options_tail"],
    "Response": ["response_choice_prefix", "response_choice", "response_reasoning_prefix", "response_reasoning"],
    "Chat": ["chat_prefix", "chat_prefix_tail", "chat_suffix", "chat_suffix_tail"],
    "Other": ["prompt_other", "response_other"],
}


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


def visualize_tokenization_from_position_mapping(
    mapping_path: Path,
    output_path: Path | None = None,
) -> bool:
    """Visualize tokenization from a SamplePositionMapping JSON file.

    Creates a token grid visualization colored by format_pos assignments.

    Args:
        mapping_path: Path to sample_position_mapping.json
        output_path: Path to save the PNG (default: same dir as mapping, tokenization.png)

    Returns:
        True if successful, False if mapping not found
    """
    from ..common.sample_position_mapping import SamplePositionMapping

    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        return False

    # Load mapping
    mapping = SamplePositionMapping.from_json(mapping_path)

    # Default output path
    if output_path is None:
        output_path = mapping_path.parent / "position_mapping.png"

    # Plot
    _plot_position_mapping_grid(mapping, output_path)
    return True


def _plot_position_mapping_grid(
    mapping,  # SamplePositionMapping
    save_path: Path,
    max_response_tokens: int = 100,
) -> None:
    """Plot token grid from SamplePositionMapping with format_pos coloring.

    Args:
        mapping: SamplePositionMapping with token info
        save_path: Path to save the plot
        max_response_tokens: Max response tokens to show
    """
    positions = mapping.positions
    prompt_len = mapping.prompt_len
    full_len = mapping.full_len

    # Limit response tokens
    response_len = full_len - prompt_len
    response_to_show = min(response_len, max_response_tokens)
    n_tokens = prompt_len + response_to_show

    # Create figure
    tokens_per_row = 15
    n_rows = (n_tokens + tokens_per_row - 1) // tokens_per_row
    fig_height = max(14, min(32, 3 + (n_rows * 0.8)))
    fig = plt.figure(figsize=(20, fig_height))

    # Info panel at top
    ax_info = fig.add_axes([0.05, 0.94, 0.9, 0.04])
    ax_info.axis("off")
    info_text = (
        f"Sample {mapping.sample_idx}    |    "
        f"Prompt: {prompt_len} tokens    |    "
        f"Total: {full_len} tokens    |    "
        f"Format positions: {len(mapping.named_positions)}"
    )
    ax_info.text(0.5, 0.5, info_text, ha="center", va="center", fontsize=11, fontweight="bold")

    # Main token grid
    ax = fig.add_axes([0.02, 0.02, 0.75, 0.90])
    ax.set_xlim(-0.5, tokens_per_row - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw each token
    for i in range(n_tokens):
        if i >= len(positions):
            break

        pos_info = positions[i]
        row = i // tokens_per_row
        col = i % tokens_per_row

        # Get color based on format_pos
        format_pos = pos_info.format_pos
        if format_pos and format_pos in FORMAT_POS_COLORS:
            facecolor = FORMAT_POS_COLORS[format_pos]
        elif pos_info.traj_section == "prompt":
            facecolor = "#E8F5E9"  # Light green for prompt
        else:
            facecolor = "#E3F2FD"  # Light blue for response

        # Edge color based on section
        edgecolor = "#388E3C" if pos_info.traj_section == "prompt" else "#1976D2"

        # Draw token box
        rect = mpatches.FancyBboxPatch(
            (col - 0.45, row - 0.4),
            0.9,
            0.8,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.5,
        )
        ax.add_patch(rect)

        # Token text (escape special chars)
        token_text = pos_info.decoded_token.replace("\n", "\\n").replace("\t", "\\t")

        # Adaptive font size
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
            col, row - 0.12, token_text,
            ha="center", va="center",
            fontsize=fontsize, fontfamily="monospace", fontweight="bold",
        )

        # Format position label (abbreviated)
        if format_pos:
            label = format_pos
            if pos_info.rel_pos >= 0:
                label = f"{label}:{pos_info.rel_pos}"
            # Truncate long labels
            if len(label) > 12:
                label = label[:10] + ".."
            ax.text(
                col, row + 0.22, label,
                ha="center", va="center",
                fontsize=4, color="#444444",
            )
        else:
            ax.text(
                col, row + 0.22, f"P{i}",
                ha="center", va="center",
                fontsize=4, color="gray",
            )

        # Position number in corner
        pos_color = "#D32F2F" if format_pos == "time_horizon" else "darkgray"
        pos_weight = "bold" if format_pos == "time_horizon" else "normal"
        ax.text(
            col - 0.35, row - 0.32, str(i),
            ha="left", va="center",
            fontsize=5, color=pos_color, fontweight=pos_weight,
        )

    # Legend - show unique format_pos values present in this sample
    ax_legend = fig.add_axes([0.78, 0.02, 0.20, 0.90])
    ax_legend.axis("off")

    # Get unique format_pos values
    unique_formats = set()
    for pos_info in positions[:n_tokens]:
        if pos_info.format_pos:
            unique_formats.add(pos_info.format_pos)

    # Sort by category
    category_order = [
        # Markers first
        "situation_marker", "task_marker", "consider_marker", "action_marker", "format_marker",
        "format_choice_prefix", "format_reasoning_prefix",
        # Content regions
        "situation_content", "task_content", "consider_content", "action_content", "format_content",
        # Variable positions
        "time_horizon", "post_time_horizon",
        "left_label", "left_reward", "left_time",
        "right_label", "right_reward", "right_time",
        # Response
        "response_choice_prefix", "response_choice", "response_reasoning_prefix", "response_reasoning",
        # Other
        "chat_prefix", "chat_suffix", "prompt_other", "response_other",
    ]

    sorted_formats = [f for f in category_order if f in unique_formats]
    # Add any not in category_order
    sorted_formats.extend([f for f in sorted(unique_formats) if f not in sorted_formats])

    # Draw legend entries
    y_pos = 0.98
    ax_legend.text(0.05, y_pos, "Format Positions:", fontsize=9, fontweight="bold", va="top")
    y_pos -= 0.04

    for fmt in sorted_formats:
        if y_pos < 0.02:
            break
        color = FORMAT_POS_COLORS.get(fmt, "#CCCCCC")
        rect = mpatches.FancyBboxPatch(
            (0.05, y_pos - 0.015), 0.08, 0.025,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            facecolor=color, edgecolor="#666666", linewidth=0.5,
            transform=ax_legend.transAxes, clip_on=False,
        )
        ax_legend.add_patch(rect)
        ax_legend.text(0.15, y_pos, fmt, fontsize=7, va="center", transform=ax_legend.transAxes)
        y_pos -= 0.028

    _finalize_plot(save_path)


def visualize_position_mapping_pair(
    mapping_short: "SamplePositionMapping",
    mapping_long: "SamplePositionMapping",
    save_path: Path,
    max_response_tokens: int = 100,
) -> None:
    """Visualize position mappings for both samples in a contrastive pair.

    Creates a two-panel visualization showing the short-term (clean) and
    long-term (corrupted) samples side-by-side with format_pos coloring.

    Args:
        mapping_short: Position mapping for short-term (clean) sample
        mapping_long: Position mapping for long-term (corrupted) sample
        save_path: Path to save the PNG
        max_response_tokens: Max response tokens to show per sample
    """
    # Calculate dimensions
    tokens_per_row = 15

    def calc_rows(mapping):
        prompt_len = mapping.prompt_len
        full_len = mapping.full_len
        response_len = full_len - prompt_len
        response_to_show = min(response_len, max_response_tokens)
        n_tokens = prompt_len + response_to_show
        return (n_tokens + tokens_per_row - 1) // tokens_per_row, n_tokens

    n_rows_short, n_tokens_short = calc_rows(mapping_short)
    n_rows_long, n_tokens_long = calc_rows(mapping_long)
    n_rows = max(n_rows_short, n_rows_long)

    # Create figure - two panels stacked
    fig_height = max(20, min(45, 4 + (n_rows * 1.6)))
    fig = plt.figure(figsize=(22, fig_height))

    # Helper to draw one sample
    def draw_sample(mapping, ax, n_tokens, title_text):
        positions = mapping.positions
        prompt_len = mapping.prompt_len

        ax.set_xlim(-0.5, tokens_per_row - 0.5)
        n_rows_sample = (n_tokens + tokens_per_row - 1) // tokens_per_row
        ax.set_ylim(-0.5, n_rows_sample - 0.5)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")

        # Title
        ax.set_title(title_text, fontsize=10, fontweight="bold", pad=5)

        for i in range(n_tokens):
            if i >= len(positions):
                break

            pos_info = positions[i]
            row = i // tokens_per_row
            col = i % tokens_per_row

            # Get color based on format_pos
            format_pos = pos_info.format_pos
            if format_pos and format_pos in FORMAT_POS_COLORS:
                facecolor = FORMAT_POS_COLORS[format_pos]
            elif pos_info.traj_section == "prompt":
                facecolor = "#E8F5E9"
            else:
                facecolor = "#E3F2FD"

            edgecolor = "#388E3C" if pos_info.traj_section == "prompt" else "#1976D2"

            # Draw token box
            rect = mpatches.FancyBboxPatch(
                (col - 0.45, row - 0.4), 0.9, 0.8,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5,
            )
            ax.add_patch(rect)

            # Token text
            token_text = pos_info.decoded_token.replace("\n", "\\n").replace("\t", "\\t")
            if len(token_text) > 12:
                token_text = token_text[:10] + ".."
                fontsize = 5
            elif len(token_text) > 8:
                fontsize = 5.5
            elif len(token_text) > 5:
                fontsize = 6
            else:
                fontsize = 7

            ax.text(col, row - 0.12, token_text, ha="center", va="center",
                    fontsize=fontsize, fontfamily="monospace", fontweight="bold")

            # Format position label
            if format_pos:
                label = format_pos
                if pos_info.rel_pos >= 0:
                    label = f"{label}:{pos_info.rel_pos}"
                if len(label) > 12:
                    label = label[:10] + ".."
                ax.text(col, row + 0.22, label, ha="center", va="center",
                        fontsize=4, color="#444444")
            else:
                ax.text(col, row + 0.22, f"P{i}", ha="center", va="center",
                        fontsize=4, color="gray")

            # Position number in corner
            pos_color = "#D32F2F" if format_pos == "time_horizon" else "darkgray"
            pos_weight = "bold" if format_pos == "time_horizon" else "normal"
            ax.text(col - 0.35, row - 0.32, str(i), ha="left", va="center",
                    fontsize=5, color=pos_color, fontweight=pos_weight)

    # Info panel at top
    ax_info = fig.add_axes([0.05, 0.96, 0.9, 0.03])
    ax_info.axis("off")
    info_text = (
        f"Contrastive Pair Position Mapping    |    "
        f"Short-term: {mapping_short.full_len} tokens    |    "
        f"Long-term: {mapping_long.full_len} tokens"
    )
    ax_info.text(0.5, 0.5, info_text, ha="center", va="center", fontsize=11, fontweight="bold")

    # Draw short-term sample (top)
    ax_short = fig.add_axes([0.02, 0.52, 0.75, 0.42])
    draw_sample(
        mapping_short, ax_short, n_tokens_short,
        f"Sample 0 (Short-term/Clean)    |    Prompt: {mapping_short.prompt_len} tokens"
    )

    # Draw long-term sample (bottom)
    ax_long = fig.add_axes([0.02, 0.05, 0.75, 0.42])
    draw_sample(
        mapping_long, ax_long, n_tokens_long,
        f"Sample 1 (Long-term/Corrupted)    |    Prompt: {mapping_long.prompt_len} tokens"
    )

    # Legend (shared) - single column, compact layout
    ax_legend = fig.add_axes([0.78, 0.02, 0.21, 0.94])
    ax_legend.axis("off")

    # Get unique format_pos values from both samples
    unique_formats = set()
    for pos_info in mapping_short.positions[:n_tokens_short]:
        if pos_info.format_pos:
            unique_formats.add(pos_info.format_pos)
    for pos_info in mapping_long.positions[:n_tokens_long]:
        if pos_info.format_pos:
            unique_formats.add(pos_info.format_pos)

    # Section order for legend
    section_order = ["Chat", "Situation", "Task", "Options", "Consider", "Action", "Format", "Response"]

    # Count total items to calculate spacing
    total_items = 0
    sections_with_items = []
    for section_name in section_order:
        if section_name not in LEGEND_SECTIONS:
            continue
        present_items = [item for item in LEGEND_SECTIONS[section_name] if item in unique_formats]
        if present_items:
            sections_with_items.append((section_name, present_items))
            total_items += len(present_items) + 1  # +1 for header

    # Calculate spacing
    available_height = 0.96
    item_height = available_height / max(total_items, 1)
    item_height = min(item_height, 0.022)  # Cap max height

    y = 0.98
    for section_name, present_items in sections_with_items:
        # Section header - bold, larger
        ax_legend.text(0.0, y, section_name, fontsize=9, fontweight="bold",
                       va="top", transform=ax_legend.transAxes, color="#000000")
        y -= item_height * 1.1

        # Section items
        for fmt in present_items:
            color = FORMAT_POS_COLORS.get(fmt, "#CCCCCC")

            # Draw color swatch (small square)
            rect = mpatches.Rectangle(
                (0.0, y - item_height * 0.35), 0.10, item_height * 0.7,
                facecolor=color, edgecolor="#666666", linewidth=0.5,
                transform=ax_legend.transAxes, clip_on=False,
            )
            ax_legend.add_patch(rect)

            # Format display name - remove underscores, keep full name
            display_name = fmt.replace("_", " ")

            # Tail items in italics
            fontstyle = "italic" if "_tail" in fmt else "normal"
            ax_legend.text(0.12, y - item_height * 0.1, display_name, fontsize=8, va="center",
                           fontstyle=fontstyle, transform=ax_legend.transAxes)
            y -= item_height

    _finalize_plot(save_path)
