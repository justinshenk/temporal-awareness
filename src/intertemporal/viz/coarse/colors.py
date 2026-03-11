"""Color palette and styling constants for coarse patching visualization.

Uses Okabe-Ito colorblind-safe palette as base.
"""

from __future__ import annotations


# Okabe-Ito palette (colorblind-safe)
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Semantic colors for metrics
METRIC_COLORS = {
    # Recovery/disruption (prominent - vermillion)
    "recovery": OKABE_ITO["vermillion"],
    "disruption": OKABE_ITO["vermillion"],
    # Logit diff (blue family)
    "logit_diff": OKABE_ITO["blue"],
    "norm_logit_diff": OKABE_ITO["sky_blue"],
    # Column 2: Short trajectory - teal/cyan family (dark to light)
    "short": "#008080",
    "logprob_short": "#006666",
    "prob_short": "#20B2AA",
    # Column 2: Long trajectory - orange family (dark to light)
    "long": "#FF8C00",
    "logprob_long": "#FF6600",
    "prob_long": "#FFB347",
    # Column 3: Short logits - purple family (dark to light)
    "logit_short": "#6A0DAD",
    "norm_logit_short": "#9370DB",
    # Column 3: Long logits - magenta/pink family (dark to light)
    "logit_long": "#C71585",
    "norm_logit_long": "#FF69B4",
    # Column 3: Relative logit delta - bright green for visibility
    "rel_logit_delta": "#00FF00",
    # Fork and Vocab metrics (same colors for same quantities)
    "fork_entropy": OKABE_ITO["blue"],
    "fork_diversity": OKABE_ITO["reddish_purple"],
    "fork_simpson": OKABE_ITO["vermillion"],
    "vocab_entropy": OKABE_ITO["blue"],
    "vocab_diversity": OKABE_ITO["reddish_purple"],
    "vocab_simpson": OKABE_ITO["vermillion"],
    # Column 1: Reciprocal rank
    "rr_short": "#2E8B57",
    "rr_long": "#DAA520",
    # Column 6: Trajectory metrics
    "inv_perplexity": OKABE_ITO["vermillion"],
    "inv_perplexity_short": "#228B22",
    "inv_perplexity_long": "#B8860B",
    # TCB
    "vocab_tcb": "#8B4513",
}

# Line styles - solid for primary, dashed/dotted for secondary
LINE_STYLES = {
    # Primary metrics (solid)
    "recovery": "-",
    "logit_diff": "-",
    "logprob_short": "-",
    "logprob_long": "-",
    "logit_short": "-",
    "logit_long": "-",
    "fork_entropy": "-",
    "vocab_entropy": "-",
    "vocab_tcb": "-",
    "rr_short": "-",
    "rr_long": "-",
    "inv_perplexity_short": "-",
    "inv_perplexity_long": "-",
    # Secondary metrics (dashed/dotted)
    "rel_logit_delta": "-",
    "norm_logit_diff": "--",
    "prob_short": "--",
    "prob_long": "--",
    "norm_logit_short": "--",
    "norm_logit_long": "--",
    "fork_diversity": "--",
    "fork_simpson": "-.",
    "vocab_diversity": "--",
    "vocab_simpson": "-.",
    "inv_perplexity": "-",
}

# Line widths - thinner for better visibility with granular sweeps
LINE_WIDTHS = {
    "recovery": 2.0,
    "logit_diff": 1.8,
    "rel_logit_delta": 2.0,
    "norm_logit_diff": 1.5,
    "logprob_short": 1.8,
    "logprob_long": 1.8,
    "prob_short": 1.5,
    "prob_long": 1.5,
    "logit_short": 1.8,
    "logit_long": 1.8,
    "norm_logit_short": 1.5,
    "norm_logit_long": 1.5,
    "fork_entropy": 1.8,
    "fork_diversity": 1.8,
    "fork_simpson": 1.5,
    "vocab_entropy": 1.8,
    "vocab_diversity": 1.5,
    "vocab_simpson": 1.5,
    "vocab_tcb": 1.5,
    "rr_short": 1.8,
    "rr_long": 1.8,
    "inv_perplexity": 1.8,
    "inv_perplexity_short": 1.8,
    "inv_perplexity_long": 1.8,
}

# Markers - distinct shapes for easy identification
MARKERS = {
    "recovery": "D",
    "logit_diff": "o",
    "rel_logit_delta": "H",  # Hexagon for rel_logit_delta
    "norm_logit_diff": "s",
    "logprob_short": "o",
    "logprob_long": "^",
    "prob_short": "o",
    "prob_long": "^",
    "logit_short": "P",
    "logit_long": "X",
    "norm_logit_short": "s",
    "norm_logit_long": "v",
    "fork_entropy": "o",
    "fork_diversity": "s",
    "fork_simpson": "D",
    "vocab_entropy": "h",
    "vocab_diversity": "p",
    "vocab_simpson": "D",
    "vocab_tcb": "*",
    "rr_short": "o",
    "rr_long": "^",
    "inv_perplexity": "*",
    "inv_perplexity_short": "o",
    "inv_perplexity_long": "^",
}

# Marker sizes - smaller for granular sweeps
MARKER_SIZES = {
    "recovery": 6,
    "logit_diff": 5,
    "rel_logit_delta": 6,
    "norm_logit_diff": 4,
    "logprob_short": 5,
    "logprob_long": 5,
    "prob_short": 4,
    "prob_long": 4,
    "logit_short": 5,
    "logit_long": 5,
    "norm_logit_short": 4,
    "norm_logit_long": 4,
    "fork_entropy": 5,
    "fork_diversity": 4,
    "fork_simpson": 4,
    "vocab_entropy": 5,
    "vocab_diversity": 4,
    "vocab_simpson": 4,
    "vocab_tcb": 5,
    "rr_short": 5,
    "rr_long": 5,
    "inv_perplexity": 6,
    "inv_perplexity_short": 5,
    "inv_perplexity_long": 5,
}

# Vertical line markers for position sweeps
VLINE_COLORS = {
    "prompt_boundary": "#1E90FF",  # Dodger blue - prompt/response boundary
    "choice_div_pos": "#FF1493",   # Deep pink - where A/B tokens diverge
}

# Subplot title style
SUBPLOT_TITLE_STYLE = {
    "fontsize": 14,
    "fontweight": "bold",
}
