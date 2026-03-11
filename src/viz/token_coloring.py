"""Token coloring types for visualization."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..common.contrastive_pair import ContrastivePair
from ..common.patching_types import TrajectoryType
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
        clean_colors: position -> TokenColorInfo for clean trajectory
        corrupted_colors: position -> TokenColorInfo for corrupted trajectory
        clean_prompt_len: prompt length for clean trajectory
        corrupted_prompt_len: prompt length for corrupted trajectory
    """

    clean_colors: dict[int, TokenColorInfo] = field(default_factory=dict)
    corrupted_colors: dict[int, TokenColorInfo] = field(default_factory=dict)
    clean_prompt_len: int = 0
    corrupted_prompt_len: int = 0

    def get_position_labels(self, trajectory: TrajectoryType = "clean") -> list[str]:
        """Get position labels for use in heatmap visualizations.

        Returns labels like 'p0', 'p1', etc. with special markers for
        divergent positions.
        """
        colors = self.clean_colors if trajectory == "clean" else self.corrupted_colors
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

    def get_section_markers(self, trajectory: TrajectoryType = "clean") -> dict[str, int | None]:
        """Get section markers for prompt/response boundary and choice divergence.

        Returns:
            Dict with:
            - prompt_boundary: position where prompt ends (response starts)
            - choice_div_pos: position where A/B choices diverge (if any)
        """
        colors = self.clean_colors if trajectory == "clean" else self.corrupted_colors
        prompt_len = self.clean_prompt_len if trajectory == "clean" else self.corrupted_prompt_len

        # Find choice divergent position from colors
        choice_div_pos = None
        for pos, info in colors.items():
            if info.is_choice_divergent:
                choice_div_pos = pos
                break

        return {
            "prompt_boundary": prompt_len,
            "choice_div_pos": choice_div_pos,
        }


def get_token_coloring_for_pair(
    pair: ContrastivePair,
) -> PairTokenColoring:
    """Build token coloring info for a contrastive pair.

    Extracts coloring logic from tokenization visualization for reuse
    in other visualizations (heatmaps, etc.).

    Token categories (colors defined in palettes.TOKEN_COLORS):
    - Prompt tokens: tokens before the response
    - Response tokens: tokens in the model's response
    - Choice divergent: position where A vs B choices diverge
    - Contrastive divergent: first divergent position in prompt and response

    Args:
        pair: ContrastivePair with token information

    Returns:
        PairTokenColoring with color info for both trajectories
    """
    clean_ids = pair.clean_traj.token_ids
    corrupted_ids = pair.corrupted_traj.token_ids

    clean_prompt_len = pair.clean_prompt_length
    corrupted_prompt_len = pair.corrupted_prompt_length

    # Find first contrastive divergent position in prompt and response
    min_prompt_len = min(clean_prompt_len, corrupted_prompt_len)

    # First divergent in prompt region (same absolute position)
    first_prompt_div = None
    for j in range(min_prompt_len):
        if clean_ids[j] != corrupted_ids[j]:
            first_prompt_div = j
            break

    # First divergent in response region (same RELATIVE position within response)
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

    # Convert to absolute positions for each trajectory
    clean_first_prompt_div = first_prompt_div
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

    # Get choice divergent positions (where A vs B diverge)
    choice_div_clean = None
    choice_div_corrupted = None
    if pair.choice_divergent_positions:
        choice_div_clean, choice_div_corrupted = pair.choice_divergent_positions

    # Build color dictionaries
    clean_colors = _build_position_colors(
        n_tokens=len(clean_ids),
        prompt_len=clean_prompt_len,
        choice_div_pos=choice_div_clean,
        first_prompt_div=clean_first_prompt_div,
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


def _build_position_colors(
    n_tokens: int,
    prompt_len: int,
    choice_div_pos: int | None,
    first_prompt_div: int | None,
    first_response_div: int | None,
) -> dict[int, TokenColorInfo]:
    """Build color info for each position in a trajectory.

    Args:
        n_tokens: Total number of tokens in the trajectory.
        prompt_len: Number of tokens in the prompt portion (tokens before this
            index are prompt, at or after are response).
        choice_div_pos: Position where choice A vs B diverge, or None if not
            applicable.
        first_prompt_div: Position of first contrastive divergence in the prompt
            region (where clean and corrupted trajectories first differ), or None.
        first_response_div: Position of first contrastive divergence in the
            response region, or None.

    Returns:
        Dictionary mapping token position to TokenColorInfo with appropriate
        colors based on position type (prompt/response, choice divergent,
        contrastive divergent).
    """
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
            # Red: first contrastive divergent (clean vs corrupted trajectory)
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
