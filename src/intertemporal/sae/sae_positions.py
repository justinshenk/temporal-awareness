"""Position extraction utilities for SAE analysis.

Extracts specific token positions of interest from prompts using structural markers.
These positions correspond to:
- P86-type: Source position where time horizon information appears
- P145-type: Destination position where the model makes its choice
- P87-type: Secondary source position (adjacent to primary source)
"""

from dataclasses import dataclass, field

from ..formatting.configs.default_prompt_format import DefaultPromptFormat


# =============================================================================
# Position Names
# =============================================================================

# Named position constants
SOURCE_POS = "source"  # Primary source (time horizon number position)
DEST_POS = "dest"  # Destination (choice position)
SECONDARY_SOURCE_POS = "secondary_source"  # Secondary source (adjacent to primary)

POSITION_NAMES = [SOURCE_POS, DEST_POS, SECONDARY_SOURCE_POS]


# =============================================================================
# Component Types
# =============================================================================

COMPONENTS = ["resid_pre", "resid_post", "mlp_out", "attn_out"]

# Hook name patterns for each component type
# Uses ModelRunner convention: blocks.{layer}.hook_{component}
HOOK_PATTERNS = {
    "resid_pre": "blocks.{layer}.hook_resid_pre",
    "resid_post": "blocks.{layer}.hook_resid_post",
    "mlp_out": "blocks.{layer}.hook_mlp_out",
    "attn_out": "blocks.{layer}.hook_attn_out",
}


def get_hook_name(component: str, layer: int) -> str:
    """Get the hook name for a component at a specific layer."""
    if component not in HOOK_PATTERNS:
        raise ValueError(f"Unknown component: {component}. Must be one of {COMPONENTS}")
    return HOOK_PATTERNS[component].format(layer=layer)


def get_names_filter(components: list[str], layers: list[int]):
    """Build a names filter function for run_with_cache."""
    hook_names = set()
    for component in components:
        for layer in layers:
            hook_names.add(get_hook_name(component, layer))

    def names_filter(name: str) -> bool:
        return name in hook_names

    return names_filter


# =============================================================================
# Position Data Classes
# =============================================================================


@dataclass
class ResolvedPositions:
    """Resolved token positions for a specific sample.

    Attributes:
        source: Token index for primary source (time horizon info)
        dest: Token index for destination (choice position)
        secondary_source: Token index for secondary source (adjacent to primary)
        prompt_len: Total length of the prompt tokens
        full_len: Total length of prompt + response tokens
    """

    source: int
    dest: int
    secondary_source: int
    prompt_len: int
    full_len: int

    def get(self, pos_name: str) -> int:
        """Get position index by name."""
        if pos_name == SOURCE_POS:
            return self.source
        elif pos_name == DEST_POS:
            return self.dest
        elif pos_name == SECONDARY_SOURCE_POS:
            return self.secondary_source
        else:
            raise ValueError(f"Unknown position name: {pos_name}")

    def to_dict(self) -> dict[str, int]:
        """Convert to dict for serialization."""
        return {
            SOURCE_POS: self.source,
            DEST_POS: self.dest,
            SECONDARY_SOURCE_POS: self.secondary_source,
            "prompt_len": self.prompt_len,
            "full_len": self.full_len,
        }


# =============================================================================
# Position Resolution
# =============================================================================


def _find_marker_token_position(
    tokens: list[str], marker: str, from_end: bool = False
) -> int:
    """Find the token position of a text marker.

    Args:
        tokens: List of decoded token strings
        marker: Text to search for
        from_end: If True, search from end of sequence

    Returns:
        Token index where marker is found, or -1 if not found
    """
    marker_lower = marker.lower().strip()

    # For searching, we look for the token that contains the marker
    indices = range(len(tokens) - 1, -1, -1) if from_end else range(len(tokens))

    for i in indices:
        tok = tokens[i].lower().strip()
        if marker_lower in tok or tok in marker_lower:
            return i

    # Second pass: check for partial matches at word boundaries
    for i in indices:
        tok = tokens[i].lower().strip()
        # Handle punctuation-attached tokens
        tok_clean = tok.rstrip(":,.")
        marker_clean = marker_lower.rstrip(":,.")
        if tok_clean == marker_clean:
            return i

    return -1


def resolve_positions(
    tokens: list[str],
    prompt_len: int,
    prompt_format: DefaultPromptFormat | None = None,
) -> ResolvedPositions:
    """Resolve token positions of interest from a tokenized sequence.

    Uses structural markers from the prompt format to find:
    - Source position: Located after "CONSIDER:" marker (time horizon info)
    - Destination position: Located after "I choose:" in response
    - Secondary source: Position adjacent to primary source

    Args:
        tokens: List of decoded token strings for the full sequence
        prompt_len: Number of tokens in the prompt (before response)
        prompt_format: Prompt format config (uses default if None)

    Returns:
        ResolvedPositions with token indices for each position of interest
    """
    if prompt_format is None:
        prompt_format = DefaultPromptFormat()

    full_len = len(tokens)

    # Find source position: after CONSIDER: marker (where time horizon appears)
    consider_marker = prompt_format.get_prompt_marker_before_time_horizon()
    consider_pos = _find_marker_token_position(tokens[:prompt_len], consider_marker)

    if consider_pos >= 0:
        # Source is a few tokens after CONSIDER: (where the time horizon number would be)
        # The time horizon spec template is: "You are primarily concerned about outcome in [time_horizon]."
        # We want the position of the time horizon value
        source_pos = min(consider_pos + 10, prompt_len - 1)  # Approximate offset
    else:
        # Fallback: use position around 60% into prompt
        source_pos = int(prompt_len * 0.6)

    # Secondary source is adjacent to primary source
    secondary_source_pos = min(source_pos + 1, prompt_len - 1)

    # Find destination position: after "I choose:" in response
    choice_marker = prompt_format.get_response_prefix_before_choice()
    # Search in the response portion (after prompt_len)
    choice_pos = _find_marker_token_position(
        tokens[prompt_len:], choice_marker, from_end=False
    )

    if choice_pos >= 0:
        # Destination is right after "I choose:" where the choice token appears
        dest_pos = prompt_len + choice_pos + 1  # +1 to get past the marker
    else:
        # Fallback: use first position in response
        dest_pos = prompt_len

    # Clamp all positions to valid range
    source_pos = max(0, min(source_pos, full_len - 1))
    secondary_source_pos = max(0, min(secondary_source_pos, full_len - 1))
    dest_pos = max(0, min(dest_pos, full_len - 1))

    return ResolvedPositions(
        source=source_pos,
        dest=dest_pos,
        secondary_source=secondary_source_pos,
        prompt_len=prompt_len,
        full_len=full_len,
    )


def decode_tokens(tokenizer, token_ids: list[int]) -> list[str]:
    """Decode token IDs to individual token strings."""
    return [tokenizer.decode([t]) for t in token_ids]


# =============================================================================
# SAE Configuration
# =============================================================================


@dataclass
class SAETarget:
    """Specification for what an SAE should analyze.

    Defines the (layer, component, position) tuple for SAE training.
    """

    layer: int
    component: str
    position_name: str

    def get_hook_name(self) -> str:
        """Get the TransformerLens hook name for this target."""
        return get_hook_name(self.component, self.layer)

    def get_name(self) -> str:
        """Get a short name for this target."""
        return f"L{self.layer}_{self.component}_P{self.position_name}"


@dataclass
class SAEConfig:
    """Configuration for SAE analysis targets.

    Defines priority levels for layers, components, and positions.
    """

    # Priority layers (from user's analysis)
    high_priority_layers: list[int] = field(
        default_factory=lambda: [21, 31, 24]
    )  # L21 resid_post, L31 mlp_out, L24 resid_post
    medium_priority_layers: list[int] = field(
        default_factory=lambda: [19, 34]
    )  # L19 resid_pre, L34 resid_post
    lower_priority_layers: list[int] = field(
        default_factory=lambda: [25]
    )  # L25 attn_out (counterproductive)

    # Components of interest
    components: list[str] = field(
        default_factory=lambda: ["resid_post", "mlp_out", "attn_out", "resid_pre"]
    )

    # Position names
    position_names: list[str] = field(
        default_factory=lambda: [DEST_POS, SOURCE_POS, SECONDARY_SOURCE_POS]
    )

    @property
    def all_layers(self) -> list[int]:
        """Get all layers in priority order."""
        return (
            self.high_priority_layers
            + self.medium_priority_layers
            + self.lower_priority_layers
        )

    def get_targets(self, priority: str = "high") -> list[SAETarget]:
        """Get SAE targets for a given priority level.

        Args:
            priority: "high", "medium", "lower", or "all"

        Returns:
            List of SAETarget objects
        """
        if priority == "high":
            layers = self.high_priority_layers
        elif priority == "medium":
            layers = self.medium_priority_layers
        elif priority == "lower":
            layers = self.lower_priority_layers
        elif priority == "all":
            layers = self.all_layers
        else:
            raise ValueError(f"Unknown priority: {priority}")

        targets = []
        for layer in layers:
            for component in self.components:
                for pos_name in self.position_names:
                    targets.append(
                        SAETarget(
                            layer=layer,
                            component=component,
                            position_name=pos_name,
                        )
                    )
        return targets

    def get_recommended_targets(self) -> list[SAETarget]:
        """Get the recommended targets based on user's analysis.

        Returns targets in the order specified in the user's requirements:
        1. L21 resid_post at P145 (dest)
        2. L31 mlp_out at P145 (dest)
        3. L19 resid_pre at P86 (source)
        4. L24 resid_post at P145 (dest)
        5. L24 attn_out at P145 (dest)
        6. L34 resid_post at P145 (dest)
        """
        recommended = [
            # Highest priority - dest position
            SAETarget(21, "resid_post", DEST_POS),
            SAETarget(31, "mlp_out", DEST_POS),
            SAETarget(24, "resid_post", DEST_POS),
            # Source positions for comparison
            SAETarget(19, "resid_pre", SOURCE_POS),
            SAETarget(21, "resid_post", SOURCE_POS),
            # Additional high priority
            SAETarget(24, "attn_out", DEST_POS),
            SAETarget(34, "resid_post", DEST_POS),
            # Lower priority / counterproductive
            SAETarget(19, "mlp_out", DEST_POS),
            SAETarget(25, "attn_out", DEST_POS),
            # Secondary source comparisons
            SAETarget(19, "resid_pre", SECONDARY_SOURCE_POS),
        ]
        return recommended
