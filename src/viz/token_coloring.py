"""Token coloring types for visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..common.contrastive_pair import ContrastivePair
from .palettes import TOKEN_COLORS


@dataclass
class TokenColorInfo:
    """Color info for a single token position."""

    facecolor: str
    edgecolor: str
    linewidth: float = 1.5
    is_choice_divergent: bool = False
    is_contrastive_divergent: bool = False
    is_prompt: bool = True


@dataclass
class PairTokenColoring:
    """Token coloring for both trajectories in a contrastive pair.

    Attributes:
        short_colors: position -> TokenColorInfo for short trajectory
        long_colors: position -> TokenColorInfo for long trajectory
        short_prompt_len: prompt length for short trajectory
        long_prompt_len: prompt length for long trajectory
    """

    short_colors: dict[int, TokenColorInfo] = field(default_factory=dict)
    long_colors: dict[int, TokenColorInfo] = field(default_factory=dict)
    short_prompt_len: int = 0
    long_prompt_len: int = 0

    def get_position_labels(self, trajectory: str = "short") -> list[str]:
        """Get position labels for use in heatmap visualizations.

        Returns labels like 'p0', 'p1', etc. with special markers for
        divergent positions.
        """
        colors = self.short_colors if trajectory == "short" else self.long_colors
        labels = []
        for pos in sorted(colors.keys()):
            info = colors[pos]
            prefix = ""
            if info.is_choice_divergent and info.is_contrastive_divergent:
                prefix = "*"  # Both
            elif info.is_choice_divergent:
                prefix = "^"  # Choice div
            elif info.is_contrastive_divergent:
                prefix = "~"  # Contrastive div
            labels.append(f"{prefix}p{pos}")
        return labels

    def get_section_markers(self, trajectory: str = "short") -> dict[str, int]:
        """Get section markers for prompt/response boundary."""
        prompt_len = self.short_prompt_len if trajectory == "short" else self.long_prompt_len
        return {"response": prompt_len}


def get_token_coloring_for_pair(
    pair: ContrastivePair,
    runner: Any | None = None,
) -> PairTokenColoring:
    """Build token coloring info for a contrastive pair.

    Extracts coloring logic from tokenization visualization for reuse
    in other visualizations (heatmaps, etc.).

    Colors:
    - Green: prompt tokens
    - Blue: response tokens
    - Purple: choice divergent position (where A vs B diverge)
    - Red: first contrastive divergent in prompt and response

    Args:
        pair: ContrastivePair with token information
        runner: Optional model runner (unused but kept for API consistency)

    Returns:
        PairTokenColoring with color info for both trajectories
    """
    short_ids = pair.short_traj.token_ids
    long_ids = pair.long_traj.token_ids

    short_prompt_len = pair.short_prompt_length
    long_prompt_len = pair.long_prompt_length

    # Find first contrastive divergent position in prompt and response
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

    first_response_div_offset = None
    for k in range(min_response_len):
        short_resp_idx = short_prompt_len + k
        long_resp_idx = long_prompt_len + k
        if short_ids[short_resp_idx] != long_ids[long_resp_idx]:
            first_response_div_offset = k
            break

    # Convert to absolute positions for each trajectory
    short_first_prompt_div = first_prompt_div
    long_first_prompt_div = first_prompt_div
    short_first_response_div = (
        short_prompt_len + first_response_div_offset
        if first_response_div_offset is not None
        else None
    )
    long_first_response_div = (
        long_prompt_len + first_response_div_offset
        if first_response_div_offset is not None
        else None
    )

    # Get choice divergent positions (where A vs B diverge)
    choice_div_short = None
    choice_div_long = None
    if pair.choice_divergent_positions:
        choice_div_short, choice_div_long = pair.choice_divergent_positions

    # Build color dictionaries
    short_colors = _build_position_colors(
        n_tokens=len(short_ids),
        prompt_len=short_prompt_len,
        choice_div_pos=choice_div_short,
        first_prompt_div=short_first_prompt_div,
        first_response_div=short_first_response_div,
    )

    long_colors = _build_position_colors(
        n_tokens=len(long_ids),
        prompt_len=long_prompt_len,
        choice_div_pos=choice_div_long,
        first_prompt_div=long_first_prompt_div,
        first_response_div=long_first_response_div,
    )

    return PairTokenColoring(
        short_colors=short_colors,
        long_colors=long_colors,
        short_prompt_len=short_prompt_len,
        long_prompt_len=long_prompt_len,
    )


def _build_position_colors(
    n_tokens: int,
    prompt_len: int,
    choice_div_pos: int | None,
    first_prompt_div: int | None,
    first_response_div: int | None,
) -> dict[int, TokenColorInfo]:
    """Build color info for each position in a trajectory."""
    colors = {}

    # Contrastive divergent positions (at most 2: first in prompt, first in response)
    contrastive_div_positions = set()
    if first_prompt_div is not None:
        contrastive_div_positions.add(first_prompt_div)
    if first_response_div is not None:
        contrastive_div_positions.add(first_response_div)

    for i in range(n_tokens):
        is_choice_div = choice_div_pos is not None and i == choice_div_pos
        is_contrastive_div = i in contrastive_div_positions
        is_prompt = i < prompt_len

        # Determine color based on position type
        if is_choice_div and is_contrastive_div:
            # Both: purple fill with red border
            facecolor = TOKEN_COLORS["choice_div_light"]
            edgecolor = TOKEN_COLORS["contrast_div_edge"]
            linewidth = 3.0
        elif is_choice_div:
            # Purple: choice divergent position (A vs B)
            facecolor = TOKEN_COLORS["choice_div_light"]
            edgecolor = TOKEN_COLORS["choice_div_edge"]
            linewidth = 1.5
        elif is_contrastive_div:
            # Red: first contrastive divergent (short vs long trajectory)
            facecolor = TOKEN_COLORS["contrast_div_light"]
            edgecolor = TOKEN_COLORS["contrast_div_edge"]
            linewidth = 1.5
        elif is_prompt:
            facecolor = TOKEN_COLORS["prompt_light"]
            edgecolor = TOKEN_COLORS["prompt_edge"]
            linewidth = 1.5
        else:
            facecolor = TOKEN_COLORS["response_light"]
            edgecolor = TOKEN_COLORS["response_edge"]
            linewidth = 1.5

        colors[i] = TokenColorInfo(
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            is_choice_divergent=is_choice_div,
            is_contrastive_divergent=is_contrastive_div,
            is_prompt=is_prompt,
        )

    return colors
