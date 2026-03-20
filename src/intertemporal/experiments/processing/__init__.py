"""Processing module: algorithmic analysis extracted from visualization code.

This module contains all the computational logic for processing activation
patching results into structured analysis results. The visualization code
consumes these pre-computed results.
"""

from .circuit_analysis import extract_circuit_hypothesis
from .component_analysis import (
    compute_cumulative_recovery,
    detect_hub_regions,
    rank_component_importance,
)
from .method_agreement import (
    analyze_attribution_agreement,
    compute_method_agreement,
    MethodAgreementResults,
    MethodPairAgreement,
)
from .redundancy_analysis import compute_redundancy_gaps
from .results import (
    CircuitHypothesis,
    ComponentComparisonResults,
    ComponentImportance,
    CumulativeRecovery,
    HubRegion,
    LayerInfo,
    LayerPositionBinding,
    PositionAnalysis,
    PositionInfo,
    ProcessedResults,
    RankedComponent,
    RedundancyAnalysis,
    RedundancyGap,
)

__all__ = [
    # Analysis functions
    "extract_circuit_hypothesis",
    "compute_cumulative_recovery",
    "detect_hub_regions",
    "rank_component_importance",
    "compute_redundancy_gaps",
    "analyze_attribution_agreement",
    "compute_method_agreement",
    # Result dataclasses
    "CircuitHypothesis",
    "ComponentComparisonResults",
    "ComponentImportance",
    "CumulativeRecovery",
    "HubRegion",
    "LayerInfo",
    "LayerPositionBinding",
    "MethodAgreementResults",
    "MethodPairAgreement",
    "PositionAnalysis",
    "PositionInfo",
    "ProcessedResults",
    "RankedComponent",
    "RedundancyAnalysis",
    "RedundancyGap",
]
