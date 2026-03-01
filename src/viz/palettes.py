"""Color palettes for visualizations."""


def _lighten(hex_color: str, factor: float = 0.4) -> str:
    """Create a lighter version of a hex color by blending with white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


# Main palettes - use these for all plots
PALETTES = [
    [
        "#D95252",
        "#1F7A7A",
        "#D4A030",
        "#6B4580",
        "#5E8C46",
        "#C4436B",
        "#3B7A8C",
        "#B87530",
        "#4A5F7A",
        "#8B6E3A",
    ],
    [
        "#B5465A",
        "#D4943A",
        "#6A8C69",
        "#3A6078",
        "#7B6B8A",
        "#C75C3A",
        "#2E7D7B",
        "#A67C3D",
        "#5B4A6E",
        "#8C6A52",
    ],
    [
        "#8C3B52",
        "#4A7B78",
        "#C49038",
        "#5E6F8E",
        "#A06040",
        "#6E8B5A",
        "#7B5A7B",
        "#3D6858",
        "#B8785E",
        "#556B8A",
    ],
    [
        "#A34878",
        "#C0683B",
        "#7A9B6E",
        "#3D4F7C",
        "#C9A035",
        "#6B8585",
        "#8B4A50",
        "#5C7A3E",
        "#9A7A5C",
        "#4C6B6E",
    ],
    [
        "#B85637",
        "#2D6A72",
        "#B8902E",
        "#7B5A7B",
        "#5B8B5B",
        "#D06B5E",
        "#3A5F80",
        "#8C7A3A",
        "#6E4B6E",
        "#4A8C6A",
    ],
]

# Default palette (first one)
DEFAULT_PALETTE = PALETTES[0]

# Named colors for semantic use
COLORS = {
    # Primary colors (from palette 0)
    "primary_0": PALETTES[1][0],
    "primary_1": PALETTES[1][1],
    "primary_2": PALETTES[1][2],
    "primary_3": PALETTES[1][3],
    "primary_4": PALETTES[1][4],
    # Secondary colors (from palette 6)
    "secondary_0": PALETTES[2][0],
    "secondary_1": PALETTES[2][1],
    "secondary_2": PALETTES[2][2],
    "secondary_3": PALETTES[2][3],
    "secondary_4": PALETTES[2][4],
    # Tertiary colors (from palette 8)
    "tertiary_0": PALETTES[3][0],
    "tertiary_1": PALETTES[3][1],
    "tertiary_2": PALETTES[3][2],
    "tertiary_3": PALETTES[3][3],
    "tertiary_4": PALETTES[3][4],
}

# Semantic mappings for common use cases (palette 0)
DENOISING_COLOR = PALETTES[4][0]
NOISING_COLOR = PALETTES[4][1]

# For short/long choice visualization (palette 0)
SHORT_COLOR = PALETTES[3][0]
LONG_COLOR = PALETTES[3][1]

# For recovery/metrics (palette 6)
RECOVERY_COLOR = PALETTES[2][4]
LOGIT_DIFF_COLOR = PALETTES[2][5]

# Token type colors (for position sweeps) - palette 0
TOKEN_COLORS = {
    # Prompt tokens
    "prompt": PALETTES[0][2],
    "prompt_light": _lighten(PALETTES[0][2]),
    "prompt_edge": PALETTES[0][4],
    # Response tokens
    "response": PALETTES[0][4],
    "response_light": _lighten(PALETTES[0][4]),
    "response_edge": PALETTES[0][4],
    # Choice divergent tokens
    "choice_div": PALETTES[0][3],
    "choice_div_light": _lighten(PALETTES[0][3]),
    "choice_div_edge": PALETTES[0][3],
    # Contrastive divergent tokens
    "contrast_div": PALETTES[0][0],
    "contrast_div_light": _lighten(PALETTES[0][0]),
    "contrast_div_edge": PALETTES[0][0],
}

# Line plot colors (for sweep plots with multiple lines)
# Consistent colors for short (greenish) and long (reddish) across all metrics
# Both use solid lines - color is the distinguisher
_SHORT_COLOR = "#2E8B57"  # Sea green
_LONG_COLOR = "#CD5C5C"   # Indian red

LINE_COLORS = {
    # Short trajectory metrics (all greenish)
    "logprob_short": _SHORT_COLOR,
    "prob_short": _SHORT_COLOR,
    "rr_short": _SHORT_COLOR,
    # Long trajectory metrics (all reddish)
    "logprob_long": _LONG_COLOR,
    "prob_long": _LONG_COLOR,
    "rr_long": _LONG_COLOR,
    # Other metrics (distinct colors - blue/orange/purple family)
    "logit_diff": "#4169E1",     # Royal blue
    "recovery": "#FF8C00",       # Dark orange
    "fork_div": "#8B008B",       # Dark magenta
    "vocab_entropy": "#20B2AA",  # Light sea green (clearly different from red)
}

# Bar chart colors (palette 0)
BAR_COLORS = {
    "denoising": PALETTES[-3][0],
    "noising": PALETTES[-3][3],
}
