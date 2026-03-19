"""Difference-in-means analysis for activation patching.

Analyzes causal directions by computing clean - corrupted activation differences.
"""

from .analysis import run_diffmeans_analysis
from .results import (
    DiffMeansLayerResult,
    DiffMeansPairResult,
    DiffMeansAggregatedResults,
)
from .rotation import compute_rotation_decomposition
from .svd import compute_svd_analysis

__all__ = [
    "run_diffmeans_analysis",
    "DiffMeansLayerResult",
    "DiffMeansPairResult",
    "DiffMeansAggregatedResults",
    "compute_rotation_decomposition",
    "compute_svd_analysis",
]
