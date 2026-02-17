"""Analysis utilities for interpretability experiments."""

from .patching import (
    PatchingMetric,
    build_position_mapping,
    create_metric,
)
from .markers import (
    SECTION_COLORS,
    find_section_markers,
    get_token_labels,
)
from .attribution import (
    compute_attribution,
    compute_eap,
    compute_eap_ig,
    run_all_attribution_methods,
    aggregate_attribution_results,
    find_top_attributions,
    AttributionResult,
)

# Backwards compatibility
get_section_markers = find_section_markers

__all__ = [
    "PatchingMetric",
    "build_position_mapping",
    "create_metric",
    "SECTION_COLORS",
    "find_section_markers",
    "get_section_markers",
    "get_token_labels",
    "compute_attribution",
    "compute_eap",
    "compute_eap_ig",
    "run_all_attribution_methods",
    "aggregate_attribution_results",
    "find_top_attributions",
    "AttributionResult",
]
