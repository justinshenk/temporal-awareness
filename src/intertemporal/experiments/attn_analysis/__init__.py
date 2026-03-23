"""Attention pattern analysis for intertemporal experiments.

Provides per-head analysis of attention layers to understand:
1. Which heads attend to source positions (time horizon tokens)
2. Per-head output projections onto logit direction
3. Per-head contribution to the 2D PCA geometry
4. Cross-prompt comparison (dynamic vs static attention patterns)
"""

from .attn_analysis_results import (
    HeadAttnInfo,
    AttnLayerResult,
    AttnPairResult,
    AttnAggregatedResults,
)
from .attn_analysis_run import run_attn_analysis

__all__ = [
    "HeadAttnInfo",
    "AttnLayerResult",
    "AttnPairResult",
    "AttnAggregatedResults",
    "run_attn_analysis",
]
