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
from .method_agreement import MethodAgreementResults

if TYPE_CHECKING:
    from ..experiment_context import ExperimentContext


@dataclass
class PositionInfo(BaseSchema):
    """Information about a single position in the circuit.

    Attributes:
        position: Token position index
        recovery: Denoising recovery score
        disruption: Noising disruption score
        is_redundant: True if position has backup pathways (source positions)
        is_bottleneck: True if position is critical (destination positions)
    """

    position: int
    recovery: float
    disruption: float
    is_redundant: bool = False
    is_bottleneck: bool = False


@dataclass
class LayerInfo(BaseSchema):
    """Information about a single layer in the circuit.

    Attributes:
        layer: Layer index
        recovery: Denoising recovery score
        disruption: Noising disruption score
        component: Component type ("attn_out" or "mlp_out")
    """

    layer: int
    recovery: float
    disruption: float
    component: str = ""


@dataclass
class LayerPositionBinding(BaseSchema):
    """Inferred binding between layers and positions.

    Attributes:
        layers: Layer range string (e.g., "L21-L30")
        positions: Position range string (e.g., "P85-P101")
        binding_type: Type of binding ("attn_reads_from" or "mlp_writes_to")
    """

    layers: str
    positions: str
    binding_type: str


@dataclass
class CircuitHypothesis(BaseSchema):
    """Complete circuit hypothesis extracted from activation patching.

    This contains the key components identified by the circuit analysis:
    - Source positions: Where information is read from
    - Destination positions: Where information is written to (bottlenecks)
    - Attention layers: Layers that route information
    - MLP layers: Layers that transform information
    - Multi-function layers: Layers appearing in both attention and MLP top lists
    - Bindings: Inferred layer-position relationships

    Attributes:
        source_positions: Top source positions ranked by importance
        destination_positions: Top destination positions (bottlenecks)
        attn_layers: Top attention layers ranked by noising score
        mlp_layers: Top MLP layers ranked by noising score
        multi_function_layers: Layers appearing in both attn and mlp top lists
        layer_position_bindings: Inferred layer-position relationships
        top3_attn_noising: Sum of top-3 attention noising scores
        top3_mlp_noising: Sum of top-3 MLP noising scores
    """

    source_positions: list[PositionInfo] = field(default_factory=list)
    destination_positions: list[PositionInfo] = field(default_factory=list)
    attn_layers: list[LayerInfo] = field(default_factory=list)
    mlp_layers: list[LayerInfo] = field(default_factory=list)
    multi_function_layers: list[int] = field(default_factory=list)
    layer_position_bindings: list[LayerPositionBinding] = field(default_factory=list)
    top3_attn_noising: float = 0.0
    top3_mlp_noising: float = 0.0


@dataclass
class RedundancyGap(BaseSchema):
    """Redundancy gap for a single layer-component pair.

    The gap is computed as disruption - recovery:
    - Positive: Component is necessary (high noising, low denoising)
    - Negative: Component is sufficient (low noising, high denoising)

    Attributes:
        layer: Layer index
        component: Component type
        gap: Disruption minus recovery
    """

    layer: int
    component: str
    gap: float


@dataclass
class RedundancyAnalysis(BaseSchema):
    """Complete redundancy analysis results.

    Attributes:
        gaps: All redundancy gaps by layer and component
        layers_sorted_by_magnitude: Layer indices sorted by total |gap|
    """

    gaps: list[RedundancyGap] = field(default_factory=list)
    layers_sorted_by_magnitude: list[int] = field(default_factory=list)


@dataclass
class CumulativeRecovery(BaseSchema):
    """Cumulative recovery analysis across layers.

    Attributes:
        layers: Layer indices
        attn_cumsum: Cumulative sum of attention recovery
        mlp_cumsum: Cumulative sum of MLP recovery
        total_cumsum: Cumulative sum of total recovery
        dip_layers: Layers with significant dips (counterproductive or relative)
        key_layers: Top contributor layers
        attn_recovery: Per-layer attention recovery values
        mlp_recovery: Per-layer MLP recovery values
    """

    layers: list[int] = field(default_factory=list)
    attn_cumsum: list[float] = field(default_factory=list)
    mlp_cumsum: list[float] = field(default_factory=list)
    total_cumsum: list[float] = field(default_factory=list)
    dip_layers: list[int] = field(default_factory=list)
    key_layers: list[int] = field(default_factory=list)
    attn_recovery: list[float] = field(default_factory=list)
    mlp_recovery: list[float] = field(default_factory=list)


@dataclass
class RankedComponent(BaseSchema):
    """A single component in the importance ranking.

    Attributes:
        label: Display label (e.g., "L24_attn")
        layer: Layer index
        recovery: Denoising recovery score
        disruption: Noising disruption score
        component: Component type ("attn_out" or "mlp_out")
    """

    label: str
    layer: int
    recovery: float
    disruption: float
    component: str


@dataclass
class ComponentImportance(BaseSchema):
    """Component importance ranking results.

    Attributes:
        ranked_components: Components ranked by recovery
        multi_layer_groups: Layers that appear multiple times (both attn and mlp)
        total_recovery: Sum of all component recoveries
    """

    ranked_components: list[RankedComponent] = field(default_factory=list)
    multi_layer_groups: dict[int, list[str]] = field(default_factory=dict)
    total_recovery: float = 0.0


@dataclass
class HubRegion(BaseSchema):
    """A contiguous region of high-activity positions.

    Attributes:
        start: Start position
        end: End position
    """

    start: int
    end: int


@dataclass
class PositionAnalysis(BaseSchema):
    """Position-level analysis results.

    Attributes:
        hub_regions: Detected high-activity regions
        hub_threshold: Threshold used for hub detection
    """

    hub_regions: list[HubRegion] = field(default_factory=list)
    hub_threshold: float = 0.0


@dataclass
class ComponentComparisonResults(BaseSchema):
    """All processed results for component comparison visualization.

    This aggregates all analysis results needed for the component comparison
    plots. Each field is optional since not all analyses may be run.

    Attributes:
        circuit: Circuit hypothesis from synthesis analysis
        redundancy: Redundancy gap analysis
        cumulative: Cumulative recovery analysis
        component_importance: Component importance ranking
        position_analysis: Position-level analysis with hub regions
    """

    circuit: CircuitHypothesis | None = None
    redundancy: RedundancyAnalysis | None = None
    cumulative: CumulativeRecovery | None = None
    component_importance: ComponentImportance | None = None
    position_analysis: PositionAnalysis | None = None


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
    from .circuit_analysis import extract_circuit_hypothesis
    from .component_analysis import (
        compute_cumulative_recovery,
        detect_hub_regions,
        rank_component_importance,
    )
    from .redundancy_analysis import compute_redundancy_gaps

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


def process_attribution_agreement(ctx: ExperimentContext) -> None:
    """Process attribution patching results into method agreement analysis."""
    from .method_agreement import analyze_attribution_agreement

    log("[process] Computing attribution method agreement...")
    agreement_results = analyze_attribution_agreement(ctx.att_agg, top_k=20)

    for mode, result in agreement_results.items():
        ctx.processed_results.attribution_agreement[mode] = MethodAgreementResults(
            pair_agreements=[pa.to_dict() for pa in result.pair_agreements],
            mean_jaccard=result.mean_jaccard,
            methods_analyzed=result.methods_analyzed,
            top_k=result.top_k,
            mode=result.mode,
        )
        log(f"  {mode}: {result.overall_agreement} agreement (Jaccard={result.mean_jaccard:.3f})")
