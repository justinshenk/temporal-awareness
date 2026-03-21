"""Plot styling constants for aggregated visualization."""

from __future__ import annotations

from ..colors import METRIC_COLORS

# Individual pair line style - use original colors
PAIR_LINE_ALPHA = 0.15
PAIR_LINE_WIDTH = 0.8

# Mean line style
MEAN_LINE_WIDTH = 2.0
MEAN_LINE_ALPHA = 1.0

# Spread fill (showing distribution)
SPREAD_ALPHA = 0.25

# Data point markers on mean line
MEAN_MARKER = "o"
MEAN_MARKER_SIZE = 5

# Grid style
GRID_ALPHA = 0.5
GRID_LINE_WIDTH = 0.5

# Figure settings
SUBPLOT_WIDTH = 4
SUBPLOT_HEIGHT = 3
DPI = 150

# Title styles
TITLE_FONTSIZE = 11
TITLE_FONTWEIGHT = "bold"
AXIS_LABEL_FONTSIZE = 9
TICK_LABEL_FONTSIZE = 8

# Column definitions: column_name -> list of metric field names
# Uses computed properties (effect, effect_reciprocal_rank) that are mode-aware
COLUMN_METRICS = {
    "core": [
        "effect",  # recovery for denoising, disruption for noising
        "effect_logit_diff",  # target - source semantics
        "effect_norm_logit_diff",  # normalized version
        "effect_reciprocal_rank",  # rank of target option
    ],
    "probs": [
        "prob_short",
        "prob_long",
        "logprob_short",
        "logprob_long",
    ],
    "logits": [
        "logit_short",
        "logit_long",
        "norm_logit_short",
        "norm_logit_long",
        "effect_rel_logit_delta",
    ],
    "fork": [
        "fork_entropy",
        "fork_diversity",
        "fork_simpson",
    ],
    "vocab": [
        "vocab_entropy",
        "vocab_diversity",
        "vocab_simpson",
    ],
    "trajectory": [
        "traj_inv_perplexity_short",
        "traj_inv_perplexity_long",
        "vocab_tcb",
    ],
    # Combined multilabel metrics (only populated for "combined" perspective)
    "combined": [
        "combined_logit_diff",
        "combined_prob_short",
        "combined_prob_long",
    ],
}

# Display names for metrics
METRIC_DISPLAY_NAMES = {
    "effect": "Effect",  # Mode-aware: recovery or disruption
    "effect_logit_diff": "Logit Diff",  # Mode-aware
    "effect_norm_logit_diff": "Norm Logit Diff",  # Mode-aware
    "effect_reciprocal_rank": "RR (Target)",  # Mode-aware: rank of target option
    "recovery": "Recovery",
    "disruption": "Disruption",
    "logit_diff": "Logit Diff",
    "norm_logit_diff": "Norm Logit Diff",
    "reciprocal_rank_short": "RR (Short)",
    "reciprocal_rank_long": "RR (Long)",
    "prob_short": "P(Short)",
    "prob_long": "P(Long)",
    "logprob_short": "LogP(Short)",
    "logprob_long": "LogP(Long)",
    "logit_short": "Logit Short",
    "logit_long": "Logit Long",
    "norm_logit_short": "Norm Logit Short",
    "norm_logit_long": "Norm Logit Long",
    "rel_logit_delta": "Rel Logit Delta",
    "effect_rel_logit_delta": "Rel Logit Delta",  # Mode-aware
    "fork_entropy": "Fork Entropy",
    "fork_diversity": "Fork Diversity",
    "fork_simpson": "Fork Simpson",
    "vocab_entropy": "Vocab Entropy",
    "vocab_diversity": "Vocab Diversity",
    "vocab_simpson": "Vocab Simpson",
    "traj_inv_perplexity_short": "Inv Perp (Short)",
    "traj_inv_perplexity_long": "Inv Perp (Long)",
    "vocab_tcb": "Vocab TCB",
    # Combined multilabel metrics
    "combined_logit_diff": "Combined Logit Diff",
    "combined_prob_short": "Combined P(Short)",
    "combined_prob_long": "Combined P(Long)",
    "combined_logit_short": "Combined Logit Short",
    "combined_logit_long": "Combined Logit Long",
}

# Map metric field names to color keys
METRIC_TO_COLOR_KEY = {
    "effect": "recovery",  # Use recovery color for effect
    "effect_logit_diff": "logit_diff",  # Use logit_diff color
    "effect_norm_logit_diff": "norm_logit_diff",  # Use norm_logit_diff color
    "effect_rel_logit_delta": "rel_logit_delta",  # Use rel_logit_delta color
    "effect_reciprocal_rank": "rr_short",  # Use rr_short color
    "reciprocal_rank_short": "rr_short",
    "reciprocal_rank_long": "rr_long",
    "traj_inv_perplexity_short": "inv_perplexity_short",
    "traj_inv_perplexity_long": "inv_perplexity_long",
    # Combined multilabel metrics
    "combined_logit_diff": "logit_diff",
    "combined_prob_short": "prob_short",
    "combined_prob_long": "prob_long",
    "combined_logit_short": "logit_short",
    "combined_logit_long": "logit_long",
}


def get_metric_color(metric_name: str) -> str:
    """Get color for metric from original color palette."""
    # Map field name to color key if needed
    color_key = METRIC_TO_COLOR_KEY.get(metric_name, metric_name)
    return METRIC_COLORS.get(color_key, "#1E90FF")
