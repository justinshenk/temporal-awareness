"""Result dataclasses for processed analysis results.

These dataclasses hold pre-computed analysis results that are used by
visualization code. All calculations are done in step_process_results
before visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ....common.base_schema import BaseSchema
from ....common.logging import log
from ..attrib import MethodAgreementResults
from ..attrib.method_agreement import analyze_attribution_agreement
from ..coarse.coarse_results import ComponentComparisonResults

if TYPE_CHECKING:
    from ..experiment_context import ExperimentContext


@dataclass
class ProcessedResults(BaseSchema):
    """Top-level container for all processed results.

    This is the main result object stored in ExperimentContext and cached
    to disk. It contains processed results for all analysis types.

    Attributes:
        component_comparison: Results keyed by component name
        attribution_agreement: Method agreement for attribution patching
    """

    component_comparison: dict[str, ComponentComparisonResults] = field(
        default_factory=dict
    )
    attribution_agreement: dict[str, MethodAgreementResults] = field(
        default_factory=dict
    )


def process_coarse_results(ctx: ExperimentContext) -> None:
    """Process coarse patching results into circuit hypothesis."""
    from ..coarse.circuit_analysis import extract_circuit_hypothesis
    from ..coarse.component_analysis import (
        compute_cumulative_recovery,
        detect_hub_regions,
        rank_component_importance,
    )
    from ..coarse.redundancy_analysis import compute_redundancy_gaps

    layer_data: dict = {}
    pos_data: dict = {}

    for component, agg in ctx.coarse_agg_by_component.items():
        if not agg or agg.n_samples == 0:
            continue
        if agg.by_sample:
            first_sample = next(iter(agg.by_sample.values()))
            layer_data[component] = first_sample.get_layer_results_for_step(1)
            pos_data[component] = first_sample.get_position_results_for_step(1)

    if not layer_data:
        log("[process] No layer data available from coarse patching")
        return

    log(f"[process] Processing {len(layer_data)} components: {list(layer_data.keys())}")

    circuit = None
    redundancy = None
    cumulative = None
    importance = None
    position_analysis = None

    if layer_data and pos_data:
        circuit = extract_circuit_hypothesis(layer_data, pos_data)

    if layer_data:
        redundancy = compute_redundancy_gaps(layer_data)
        cumulative = compute_cumulative_recovery(layer_data)
        importance = rank_component_importance(layer_data)

    if pos_data:
        position_analysis = detect_hub_regions(pos_data)

    ctx.processed_results.component_comparison["all"] = ComponentComparisonResults(
        circuit=circuit,
        redundancy=redundancy,
        cumulative=cumulative,
        component_importance=importance,
        position_analysis=position_analysis,
    )


def compute_component_comparison_from_pair(
    results_by_component: dict[str, "CoarseActPatchResults"],
) -> "ComponentComparisonResults | None":
    """Compute ComponentComparisonResults from per-pair coarse results.

    This is used for per-pair visualization when we don't have aggregated data.

    Args:
        results_by_component: Dict mapping component name to per-pair results

    Returns:
        ComponentComparisonResults if computation succeeds, None otherwise
    """
    from ..coarse.circuit_analysis import extract_circuit_hypothesis
    from ..coarse.component_analysis import (
        compute_cumulative_recovery,
        detect_hub_regions,
        rank_component_importance,
    )
    from ..coarse.redundancy_analysis import compute_redundancy_gaps

    layer_data: dict = {}
    pos_data: dict = {}

    for component, result in results_by_component.items():
        if not result:
            continue
        layer_data[component] = result.get_layer_results_for_step(1)
        pos_data[component] = result.get_position_results_for_step(1)

    if not layer_data:
        return None

    circuit = None
    redundancy = None
    cumulative = None
    importance = None
    position_analysis = None

    if layer_data and pos_data:
        circuit = extract_circuit_hypothesis(layer_data, pos_data)

    if layer_data:
        redundancy = compute_redundancy_gaps(layer_data)
        cumulative = compute_cumulative_recovery(layer_data)
        importance = rank_component_importance(layer_data)

    if pos_data:
        position_analysis = detect_hub_regions(pos_data)

    return ComponentComparisonResults(
        circuit=circuit,
        redundancy=redundancy,
        cumulative=cumulative,
        component_importance=importance,
        position_analysis=position_analysis,
    )


def process_attribution_agreement(ctx: ExperimentContext) -> None:
    """Process attribution patching results into method agreement analysis."""
    log("[process] Computing attribution method agreement...")
    agreement_results = analyze_attribution_agreement(ctx.attrib_agg, top_k=20)

    for mode, result in agreement_results.items():
        ctx.processed_results.attribution_agreement[mode] = MethodAgreementResults(
            pair_agreements=[pa.to_dict() for pa in result.pair_agreements],
            mean_jaccard=result.mean_jaccard,
            methods_analyzed=result.methods_analyzed,
            top_k=result.top_k,
            mode=result.mode,
        )
        log(
            f"  {mode}: {result.overall_agreement} agreement (Jaccard={result.mean_jaccard:.3f})"
        )
