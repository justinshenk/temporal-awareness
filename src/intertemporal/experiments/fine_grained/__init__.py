"""Fine-grained activation patching analysis.

Provides comprehensive head-level, position-level, path patching,
and neuron-level analysis with corresponding visualizations.
"""

from .fine_grained_config import FineGrainedConfig, DEFAULT_FINE_GRAINED_CONFIG
from .fine_grained_results import (
    HeadPatchingResult,
    HeadSweepResults,
    PositionPatchingResult,
    PathPatchingResult,
    MultiSiteResult,
    NeuronPatchingResult,
    LayerPositionResult,
    FineGrainedResults,
)
from .fine_grained_analysis import run_fine_grained_analysis
from .fine_grained_viz import visualize_fine_grained

__all__ = [
    "FineGrainedConfig",
    "DEFAULT_FINE_GRAINED_CONFIG",
    "HeadPatchingResult",
    "HeadSweepResults",
    "PositionPatchingResult",
    "PathPatchingResult",
    "MultiSiteResult",
    "NeuronPatchingResult",
    "LayerPositionResult",
    "FineGrainedResults",
    "run_fine_grained_analysis",
    "visualize_fine_grained",
]
