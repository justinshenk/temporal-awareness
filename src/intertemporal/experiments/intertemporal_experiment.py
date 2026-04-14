"""Intertemporal preference experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...common import profile
from ...common.device_utils import clear_gpu_memory
from ...common.logging import log
from ...viz.plot_helpers import set_svg_mode
from ...activation_patching.coarse import (
    run_coarse_act_patching,
    CoarseActPatchAggregatedResults,
)
from ...attribution_patching import (
    attribute_pair,
    AttrPatchAggregatedResults,
    AttributionSettings,
)
from .coarse import CoarsePatchingConfig
from .diffmeans import (
    run_diffmeans_analysis,
    DiffMeansAggregatedResults,
    DiffMeansConfig,
)
from .mlp import run_mlp_analysis, MLPAggregatedResults, MLPAnalysisConfig
from .attn import (
    run_attn_analysis,
    AttnAggregatedResults,
    AttnAnalysisConfig,
    run_head_attribution,
    run_head_position_patching,
    run_head_patching_sweep,
)
from .fine import (
    FineGrainedConfig,
    FineAggregatedResults,
    run_fine_analysis,
    run_layer_position_patching_single,
)
from .analysis import (
    ProcessedResults,
    process_attribution_agreement,
    process_coarse_results,
    build_horizon_analysis,
    save_horizon_analysis,
    build_pair_analysis,
    save_pair_analysis,
)
from ..common import get_pref_dataset_dir
from ..preference import (
    generate_preference_data,
    load_preference_data,
    analyze_preferences,
    print_analysis,
)
from .experiment_context import ExperimentConfig, ExperimentContext

if TYPE_CHECKING:
    from ...common.contrastive_pair import ContrastivePair


# =============================================================================
# Step State Management
# =============================================================================


@dataclass
class StepState:
    """Configuration and state for a single analysis step."""

    name: str
    config: Any = None
    enabled: bool = False
    viz_per_pair: bool = False
    viz_agg: bool = False
    use_cache: bool = True

    @classmethod
    def from_cfg(
        cls,
        name: str,
        raw_cfg: dict,
        config_class: type,
        try_loading_data: bool = True,
        viz_enabled: bool = True,
    ) -> "StepState":
        """Create StepState from raw config dict."""
        enabled = raw_cfg.get("enabled", False)
        if not enabled:
            return cls(name=name, enabled=False)

        config = config_class.from_dict(raw_cfg)
        no_viz = raw_cfg.get("no_viz", False)
        only_viz_agg = raw_cfg.get("only_viz_agg", False)

        viz_per_pair = viz_enabled and not no_viz and not only_viz_agg
        viz_agg = viz_enabled and not no_viz
        use_cache = try_loading_data and not raw_cfg.get("no_cache", False)

        return cls(
            name=name,
            config=config,
            enabled=True,
            viz_per_pair=viz_per_pair,
            viz_agg=viz_agg,
            use_cache=use_cache,
        )


@dataclass
class ExperimentSteps:
    """All step states for an experiment."""

    attrib: StepState = field(default_factory=lambda: StepState("attrib"))
    coarse: StepState = field(default_factory=lambda: StepState("coarse"))
    diffmeans: StepState = field(default_factory=lambda: StepState("diffmeans"))
    mlp: StepState = field(default_factory=lambda: StepState("mlp"))
    attn: StepState = field(default_factory=lambda: StepState("attn"))
    fine: StepState = field(default_factory=lambda: StepState("fine"))

    @classmethod
    def from_cfg(
        cls,
        cfg: ExperimentConfig,
        try_loading_data: bool = True,
        viz_enabled: bool = True,
    ) -> "ExperimentSteps":
        """Create all step states from experiment config."""
        return cls(
            attrib=StepState.from_cfg(
                "attrib",
                cfg.attrib_cfg,
                AttributionSettings,
                try_loading_data,
                viz_enabled,
            ),
            coarse=StepState.from_cfg(
                "coarse",
                cfg.coarse_cfg,
                CoarsePatchingConfig,
                try_loading_data,
                viz_enabled,
            ),
            diffmeans=StepState.from_cfg(
                "diffmeans",
                cfg.diffmeans_cfg,
                DiffMeansConfig,
                try_loading_data,
                viz_enabled,
            ),
            mlp=StepState.from_cfg(
                "mlp", cfg.mlp_cfg, MLPAnalysisConfig, try_loading_data, viz_enabled
            ),
            attn=StepState.from_cfg(
                "attn", cfg.attn_cfg, AttnAnalysisConfig, try_loading_data, viz_enabled
            ),
            fine=StepState.from_cfg(
                "fine", cfg.fine_cfg, FineGrainedConfig, try_loading_data, viz_enabled
            ),
        )

    def init_aggregators(self, ctx: ExperimentContext) -> None:
        """Initialize all enabled aggregators."""
        if self.attrib.enabled:
            ctx.attrib_agg = AttrPatchAggregatedResults()
        if self.coarse.enabled:
            for component in self.coarse.config.components:
                ctx.coarse_agg_by_component[component] = (
                    CoarseActPatchAggregatedResults()
                )
        if self.diffmeans.enabled:
            ctx.diffmeans_agg = DiffMeansAggregatedResults()
        if self.mlp.enabled:
            ctx.mlp_agg = MLPAggregatedResults(layers_analyzed=self.mlp.config.layers)
        if self.attn.enabled:
            ctx.attn_agg = AttnAggregatedResults(
                layers_analyzed=self.attn.config.layers
            )
        if self.fine.enabled:
            ctx.fine_agg = FineAggregatedResults()


# =============================================================================
# Process Functions (single pair, single step)
# =============================================================================


def process_attrib(
    ctx: ExperimentContext, pair_idx: int, pair: "ContrastivePair", step: StepState
) -> None:
    """Process attribution patching for a single pair."""
    if not step.enabled:
        return

    # Load aggregator from disk if not initialized
    if ctx.attrib_agg is None:
        ctx.attrib_agg = AttrPatchAggregatedResults()
        ctx.load_attrib_agg()

    # Check cache
    if step.use_cache and ctx.is_pair_cached("attrib", pair_idx):
        if ctx.load_attrib_pair(pair_idx):
            ctx.attrib_agg.add(ctx.attrib_patching[pair_idx])
            if step.viz_per_pair:
                ctx.viz_attrib_pair(pair_idx)
            log(f"[attrib] Loaded cached pair {pair_idx + 1}")
            del ctx.attrib_patching[pair_idx]
            return
        # Cache file exists but load failed - fall through to recompute
        log(f"[attrib] Cache load failed for pair {pair_idx + 1}, recomputing")

    # Compute
    result = attribute_pair(ctx.runner, pair, settings=step.config)
    ctx.attrib_patching[pair_idx] = result
    ctx.attrib_agg.add(result)
    log(f"[attrib] Pair {pair_idx + 1}: computed attribution")

    # Save and viz
    ctx.save_attrib_pair(pair_idx)
    if step.viz_per_pair:
        ctx.viz_attrib_pair(pair_idx)

    # Cleanup
    del ctx.attrib_patching[pair_idx]
    pair.pop_heavy()
    clear_gpu_memory(aggressive=True)


def process_coarse(
    ctx: ExperimentContext, pair_idx: int, pair: "ContrastivePair", step: StepState
) -> None:
    """Process coarse patching for a single pair (all components)."""
    if not step.enabled:
        return

    config: CoarsePatchingConfig = step.config

    # Load aggregators from disk if not initialized
    if not ctx.coarse_agg_by_component:
        ctx.load_coarse_agg(config)
    for component in config.components:
        if component not in ctx.coarse_agg_by_component:
            ctx.coarse_agg_by_component[component] = CoarseActPatchAggregatedResults()

    for component in config.components:
        # Check cache
        if step.use_cache and ctx.is_pair_cached("coarse", pair_idx, component):
            if ctx.load_coarse_pair(pair_idx, component):
                ctx.coarse_agg_by_component[component].add(
                    ctx.coarse_patching[(pair_idx, component)]
                )
                if step.viz_per_pair:
                    ctx.viz_coarse_pair(pair_idx, component)
                log(
                    f"[coarse] Loaded cached pair {pair_idx + 1}, component={component}"
                )
                del ctx.coarse_patching[(pair_idx, component)]
                continue
            # Cache file exists but load failed - fall through to recompute
            log(
                f"[coarse] Cache load failed for pair {pair_idx + 1} {component}, recomputing"
            )

        # Compute
        result = run_coarse_act_patching(
            ctx.runner,
            pair,
            component=component,
            layer_step_sizes=config.layer_steps,
            pos_step_sizes=config.pos_steps,
        )
        result.sample_id = pair_idx
        ctx.coarse_patching[(pair_idx, component)] = result
        ctx.coarse_agg_by_component[component].add(result)
        log(f"[coarse] Pair {pair_idx + 1}: computed {component}")

        # Save and viz
        ctx.save_coarse_pair(pair_idx, component)
        if step.viz_per_pair:
            ctx.viz_coarse_pair(pair_idx, component)

        # Cleanup
        del ctx.coarse_patching[(pair_idx, component)]

    # Component comparison viz (requires loading all components)
    if step.viz_per_pair and len(config.components) > 1:
        for component in config.components:
            ctx.load_coarse_pair(pair_idx, component)
        ctx.viz_coarse_pair_component_comparison(pair_idx, config.components)
        for component in config.components:
            ctx.coarse_patching.pop((pair_idx, component), None)

    pair.pop_heavy()
    clear_gpu_memory(aggressive=True)


def process_diffmeans(
    ctx: ExperimentContext, pair_idx: int, pair: "ContrastivePair", step: StepState
) -> None:
    """Process diffmeans analysis for a single pair."""
    if not step.enabled:
        return

    config: DiffMeansConfig = step.config

    # Load aggregator from disk if not initialized
    if ctx.diffmeans_agg is None:
        ctx.diffmeans_agg = DiffMeansAggregatedResults()
        ctx.load_diffmeans_agg()

    # Check cache
    if step.use_cache and ctx.is_pair_cached("diffmeans", pair_idx):
        if ctx.load_diffmeans_pair(pair_idx):
            ctx.diffmeans_agg.add(ctx.diffmeans_patching[pair_idx])
            if step.viz_per_pair:
                ctx.viz_diffmeans_pair(pair_idx)
            log(f"[diffmeans] Loaded cached pair {pair_idx + 1}")
            del ctx.diffmeans_patching[pair_idx]
            return
        # Cache file exists but load failed - fall through to recompute
        log(f"[diffmeans] Cache load failed for pair {pair_idx + 1}, recomputing")

    # Compute
    clean_mapping, corrupted_mapping = ctx.position_mappings[pair_idx]
    result = run_diffmeans_analysis(
        ctx.runner,
        pair,
        clean_mapping=clean_mapping,
        corrupted_mapping=corrupted_mapping,
        pair_idx=pair_idx,
        positions=config.positions,
    )
    log(
        f"[diffmeans] Pair {pair_idx + 1}: {len(result.position_results)} positions analyzed"
    )
    ctx.diffmeans_patching[pair_idx] = result
    ctx.diffmeans_agg.add(result)

    # Save and viz
    ctx.save_diffmeans_pair(pair_idx)
    if step.viz_per_pair:
        ctx.viz_diffmeans_pair(pair_idx)

    # Cleanup
    del ctx.diffmeans_patching[pair_idx]
    pair.pop_heavy()
    clear_gpu_memory(aggressive=True)


def process_mlp(
    ctx: ExperimentContext, pair_idx: int, pair: "ContrastivePair", step: StepState
) -> None:
    """Process MLP analysis for a single pair."""
    if not step.enabled:
        return

    config: MLPAnalysisConfig = step.config

    # Load aggregator from disk if not initialized
    if ctx.mlp_agg is None:
        ctx.mlp_agg = MLPAggregatedResults(layers_analyzed=config.layers)
        ctx.load_mlp_agg()

    # Check cache
    if step.use_cache and ctx.is_pair_cached("mlp", pair_idx):
        if ctx.load_mlp_pair(pair_idx):
            ctx.mlp_agg.add(ctx.mlp[pair_idx])
            if step.viz_per_pair:
                # Use "short" mapping (longer sequence with all semantic positions)
                mapping = ctx.get_position_mapping(pair_idx, sample="short")
                ctx.viz_mlp_pair(pair_idx, mapping)
            log(f"[mlp] Loaded cached pair {pair_idx + 1}")
            del ctx.mlp[pair_idx]
            return
        # Cache file exists but load failed - fall through to recompute
        log(f"[mlp] Cache load failed for pair {pair_idx + 1}, recomputing")

    # Compute
    clean_mapping, corrupted_mapping = ctx.position_mappings[pair_idx]
    result = run_mlp_analysis(
        ctx.runner,
        pair,
        clean_mapping=clean_mapping,
        corrupted_mapping=corrupted_mapping,
        pair_idx=pair_idx,
        layers=config.layers,
        positions=config.positions,
        n_top_neurons=config.n_top_neurons,
    )

    # Layer x position patching for mlp_out
    if config.layer_position_enabled and corrupted_mapping:
        lp_positions = []
        for pos_name in config.layer_position_positions:
            lp_positions.extend(corrupted_mapping.named_positions.get(pos_name, []))
        lp_positions = sorted(set(lp_positions))

        if lp_positions:
            log(
                f"[mlp][layer_pos] Running for {len(config.layers)} layers x {len(lp_positions)} positions..."
            )
            lp_result = run_layer_position_patching_single(
                ctx.runner,
                pair,
                component="mlp_out",
                layers=config.layers,
                positions=lp_positions,
            )
            if lp_result is not None:
                result.layer_position = lp_result
                log("[mlp][layer_pos] Complete")

    ctx.mlp[pair_idx] = result
    ctx.mlp_agg.add(result)
    log(f"[mlp] Pair {pair_idx + 1}: analyzed {len(config.layers)} layers")

    # Save and viz
    ctx.save_mlp_pair(pair_idx)
    if step.viz_per_pair:
        # Use clean_mapping for viz (longer sequence with all semantic positions)
        ctx.viz_mlp_pair(pair_idx, clean_mapping)

    # Cleanup
    del ctx.mlp[pair_idx]
    pair.pop_heavy()
    clear_gpu_memory(aggressive=True)


def process_attn(
    ctx: ExperimentContext, pair_idx: int, pair: "ContrastivePair", step: StepState
) -> None:
    """Process attention analysis for a single pair."""
    if not step.enabled:
        return

    config: AttnAnalysisConfig = step.config

    # Load aggregator from disk if not initialized
    if ctx.attn_agg is None:
        ctx.attn_agg = AttnAggregatedResults(layers_analyzed=config.layers)
        ctx.load_attn_agg()

    # Check cache
    if step.use_cache and ctx.is_pair_cached("attn", pair_idx):
        if ctx.load_attn_pair(pair_idx):
            ctx.attn_agg.add(ctx.attn[pair_idx])
            if step.viz_per_pair:
                # Use "short" mapping (longer sequence with all semantic positions)
                mapping = ctx.get_position_mapping(pair_idx, sample="short")
                ctx.viz_attn_pair(pair_idx, mapping)
            log(f"[attn] Loaded cached pair {pair_idx + 1}")
            del ctx.attn[pair_idx]
            return
        # Cache file exists but load failed - fall through to recompute
        log(f"[attn] Cache load failed for pair {pair_idx + 1}, recomputing")

    # Get position mapping
    mapping = ctx.get_position_mapping(pair_idx)
    if mapping is None:
        log(f"[attn] Pair {pair_idx}: No position mapping found, skipping")
        return

    # 1. Attention pattern analysis
    result = run_attn_analysis(
        ctx.runner, pair, mapping, pair_idx=pair_idx, config=config
    )
    ctx.attn[pair_idx] = result
    ctx.attn_agg.add(result)
    log(f"[attn] Pair {pair_idx + 1}: analyzed {len(config.layers)} layers")

    # 2. Head attribution (if enabled)
    if config.head_attribution_enabled:
        log(f"[attn][head_attrib] Running for pair {pair_idx + 1}...")
        head_attrib = run_head_attribution(ctx.runner, pair, layers=config.layers)
        result.head_attribution = head_attrib
        top_heads = head_attrib.get_top_heads(config.n_top_heads_for_position)
        log(f"[attn][head_attrib] Top heads: {[h.label for h in top_heads[:5]]}")

        # 3. Position patching for top heads
        if config.position_patching_enabled and top_heads:
            positions = []
            for pos_name in config.position_patching_positions:
                positions.extend(mapping.named_positions.get(pos_name, []))
            positions = sorted(set(positions))

            if positions:
                log(
                    f"[attn][pos_patch] Running for {len(top_heads[: config.n_top_heads_for_position])} heads at {len(positions)} positions..."
                )
                pos_results = run_head_position_patching(
                    ctx.runner,
                    pair,
                    top_heads,
                    positions=positions,
                    position_names=config.position_patching_positions,
                    n_heads=config.n_top_heads_for_position,
                )
                result.head_position_patching = pos_results
                log("[attn][pos_patch] Complete")

        # 4. Head redundancy analysis
        if config.head_redundancy_enabled:
            log("[attn][redundancy] Running head patching sweep...")
            sweep = run_head_patching_sweep(
                ctx.runner,
                pair,
                layers=config.layers,
                top_n=config.head_redundancy_top_n,
            )
            result.head_redundancy = sweep
            log(f"[attn][redundancy] Complete: {len(sweep.results)} heads analyzed")

        # 5. Layer-position patching for attn_out
        if config.layer_position_enabled:
            lp_positions = []
            for pos_name in config.layer_position_positions:
                lp_positions.extend(mapping.named_positions.get(pos_name, []))
            lp_positions = sorted(set(lp_positions))

            if lp_positions:
                log(
                    f"[attn][layer_pos] Running for {len(config.layers)} layers x {len(lp_positions)} positions..."
                )
                lp_result = run_layer_position_patching_single(
                    ctx.runner,
                    pair,
                    component="attn_out",
                    layers=config.layers,
                    positions=lp_positions,
                )
                result.layer_position = lp_result
                log("[attn][layer_pos] Complete")

    # Save and viz
    ctx.save_attn_pair(pair_idx, store_patterns=config.store_patterns)
    if step.viz_per_pair:
        # Use "short" mapping for viz (longer sequence with all semantic positions)
        viz_mapping = ctx.get_position_mapping(pair_idx, sample="short")
        ctx.viz_attn_pair(pair_idx, viz_mapping)

    # Cleanup
    del ctx.attn[pair_idx]
    pair.pop_heavy()
    clear_gpu_memory(aggressive=True)


def process_fine(
    ctx: ExperimentContext, pair_idx: int, pair: "ContrastivePair", step: StepState
) -> None:
    """Process fine-grained analysis for a single pair."""
    if not step.enabled:
        return

    config: FineGrainedConfig = step.config

    # Load aggregator from disk if not initialized
    if ctx.fine_agg is None:
        ctx.fine_agg = FineAggregatedResults()
        ctx.load_fine_agg()

    # Check cache
    if step.use_cache and ctx.is_pair_cached("fine", pair_idx):
        if ctx.load_fine_pair(pair_idx):
            ctx.fine_agg.add(ctx.fine[pair_idx])
            if step.viz_per_pair:
                # Use "short" mapping (longer sequence with all semantic positions)
                mapping = ctx.get_position_mapping(pair_idx, sample="short")
                ctx.viz_fine_pair(pair_idx, mapping)
            log(f"[fine] Loaded cached pair {pair_idx + 1}")
            del ctx.fine[pair_idx]
            return
        # Cache file exists but load failed - fall through to recompute
        log(f"[fine] Cache load failed for pair {pair_idx + 1}, recomputing")

    # Get top heads from attn (if available)
    corrupted_mapping = ctx.get_position_mapping(pair_idx, sample="long")
    top_heads = None
    attn_result = ctx.attn.get(pair_idx)
    if attn_result is None:
        ctx.load_attn_pair(pair_idx)
        attn_result = ctx.attn.get(pair_idx)
    if attn_result and attn_result.head_attribution:
        top_heads = attn_result.head_attribution.get_top_heads(
            config.n_top_source_heads * 2
        )
        log(f"[fine] Using {len(top_heads)} top heads from attn")

    # Compute
    result = run_fine_analysis(ctx.runner, pair, config, top_heads=top_heads)
    result.sample_id = pair_idx
    ctx.fine[pair_idx] = result
    ctx.fine_agg.add(result)
    log(f"[fine] Pair {pair_idx + 1}: computed fine-grained analysis")

    # Save and viz
    ctx.save_fine_pair(pair_idx)
    if step.viz_per_pair:
        # Use "short" mapping for viz (longer sequence with all semantic positions)
        clean_mapping = ctx.get_position_mapping(pair_idx, sample="short")
        ctx.viz_fine_pair(pair_idx, clean_mapping)

    # Cleanup
    del ctx.fine[pair_idx]
    pair.pop_heavy()
    clear_gpu_memory(aggressive=True)


# =============================================================================
# Progressive Aggregation
# =============================================================================


def process_progressive_agg(
    ctx: ExperimentContext, pair_idx: int, steps: "ExperimentSteps"
) -> None:
    """Save aggregations, print summaries, and generate viz periodically during processing."""
    log(f"[agg] Running progressive aggregation at pair {pair_idx + 1}...", gap=1)

    # Attrib
    if ctx.attrib_agg and ctx.attrib_agg.denoising:
        ctx.attrib_agg.print_summary()
        ctx.save_attrib_agg()
        if ctx.attrib_agg.denoising_agg or ctx.attrib_agg.noising_agg:
            ctx.processed_results = ctx.processed_results or ProcessedResults()
            process_attribution_agreement(ctx)
            ctx.save_processed_results()
        if steps.attrib.viz_agg:
            ctx.make_attrib_viz(steps.attrib.config)()
            log("[attrib] Generated attribution visualizations")
        ctx.unload_attrib_agg()

    # Coarse
    if ctx.coarse_agg_by_component:
        for agg in ctx.coarse_agg_by_component.values():
            agg.print_summary()
        ctx.save_coarse_agg()
        ctx.processed_results = ctx.processed_results or ProcessedResults()
        process_coarse_results(ctx)
        ctx.save_processed_results()
        if steps.coarse.viz_agg:
            ctx.make_coarse_viz(steps.coarse.config)()
            log("[coarse] Generated coarse patching visualizations")
        ctx.unload_coarse_agg()

    # Diffmeans
    if ctx.diffmeans_agg and ctx.diffmeans_agg.pair_results:
        ctx.diffmeans_agg.print_summary()
        ctx.save_diffmeans_agg()
        if steps.diffmeans.viz_agg:
            ctx.make_diffmeans_viz(steps.diffmeans.config)()
            log("[diffmeans] Generated diffmeans visualizations")
        ctx.unload_diffmeans_agg()

    # MLP
    if ctx.mlp_agg and ctx.mlp_agg.pair_results:
        ctx.mlp_agg.print_summary()
        ctx.save_mlp_agg()
        if steps.mlp.viz_agg:
            ctx.make_mlp_viz(steps.mlp.config)()
            log("[mlp] Generated MLP analysis visualizations")
        ctx.unload_mlp_agg()

    # Attn
    if ctx.attn_agg and ctx.attn_agg.pair_results:
        ctx.attn_agg.print_summary()
        ctx.save_attn_agg()
        if steps.attn.viz_agg:
            ctx.make_attn_viz(steps.attn.config)()
            log("[attn] Generated attention analysis visualizations")
        ctx.unload_attn_agg()

    # Fine
    if ctx.fine_agg and ctx.fine_agg.pair_results:
        ctx.fine_agg.print_summary()
        ctx.save_fine_agg()
        if steps.fine.viz_agg:
            ctx.make_fine_viz(steps.fine.config)()
            log("[fine] Generated fine-grained visualizations")
        ctx.unload_fine_agg()

    log(f"[agg] Progressive aggregation complete at pair {pair_idx + 1}")


# =============================================================================
# Base Setup and Analysis
# =============================================================================


@profile("step_preference_data")
def step_preference_data(
    ctx: ExperimentContext, try_loading_data: bool = False
) -> None:
    """Load or generate preference data."""
    if try_loading_data:
        result = load_preference_data(
            ctx.cfg.get_preference_data_prefix(), get_pref_dataset_dir()
        )
        if result:
            ctx.pref_data, ctx.prompt_dataset = result
        ctx.enable_cached_pairs()

    if not ctx.pref_data:
        ctx.pref_data, ctx.prompt_dataset = generate_preference_data(
            model=ctx.cfg.model,
            dataset_config=ctx.cfg.dataset_config,
            max_samples=ctx.cfg.max_samples,
            save_data=True,
        )


def process_analysis_data(
    ctx: ExperimentContext,
    pair_idx: int,
    use_cache: bool = True,
) -> None:
    """Save analysis data files for a single pair (contrastive pref, position mapping JSON)."""
    if use_cache and ctx.is_analysis_cached(pair_idx):
        return

    ctx.save_contrastive_pref(pair_idx)
    ctx.save_position_mapping_data(pair_idx)


def process_analysis_viz(
    ctx: ExperimentContext,
    pair_idx: int,
) -> None:
    """Generate position mapping viz for a single pair (tokenization PNGs)."""
    pair_dir = ctx.get_pair_dir(pair_idx)
    viz_exists = (pair_dir / "tokenization_short.png").exists()
    if not viz_exists:
        ctx.save_position_mapping_viz(pair_idx)


def finalize_analysis(ctx: ExperimentContext) -> None:
    """Finalize analysis: save horizon and pair analysis."""
    analysis_dir = ctx.get_analysis_dir()
    save_horizon_analysis(build_horizon_analysis(ctx.pref_pairs), analysis_dir)
    save_pair_analysis(build_pair_analysis(ctx.pref_pairs), analysis_dir)


# =============================================================================
# Main Experiment Functions
# =============================================================================


PROGRESSIVE_AGG = 30  # Aggregate every N pairs


@profile("run_experiment_per_pair")
def run_experiment_per_pair(
    cfg: ExperimentConfig,
    try_loading_data: bool = False,
    output_dir: Path | None = None,
    backend: str | None = None,
) -> ExperimentContext:
    """Run experiment processing each pair through all analysis steps.

    Each pair goes through all analysis steps before moving to the next pair.
    Aggressive memory cleanup is performed between pairs to prevent memory leaks.

    Progressive aggregation runs periodically.
    """
    ctx = ExperimentContext(cfg, output_dir=output_dir, backend=backend)
    cfg.save(ctx.output_dir)

    # Set SVG mode for camera-ready figures
    set_svg_mode(ctx.save_svg)

    # Load preference data
    step_preference_data(ctx, try_loading_data=True)
    print_analysis(analyze_preferences(ctx.pref_data))

    # Initialize all step states and aggregators
    steps = ExperimentSteps.from_cfg(
        cfg, try_loading_data=try_loading_data, viz_enabled=ctx.viz_enabled
    )
    steps.init_aggregators(ctx)

    skip_viz = ctx.only_viz_agg
    n_pairs = len(ctx.pairs)

    # Process each pair through all analysis steps
    for pair_idx in range(n_pairs):
        pair = ctx.pairs[pair_idx]

        log(f"\n{'=' * 60}", gap=1)
        log(f"[per_pair] Processing pair {pair_idx + 1}/{n_pairs}", gap=1)
        log(f"{'=' * 60}")

        # Process all steps for this pair
        process_analysis_data(ctx, pair_idx, use_cache=try_loading_data)
        if not skip_viz:
            process_analysis_viz(ctx, pair_idx)
        process_attrib(ctx, pair_idx, pair, steps.attrib)
        process_coarse(ctx, pair_idx, pair, steps.coarse)
        process_diffmeans(ctx, pair_idx, pair, steps.diffmeans)
        process_mlp(ctx, pair_idx, pair, steps.mlp)
        process_attn(ctx, pair_idx, pair, steps.attn)
        process_fine(ctx, pair_idx, pair, steps.fine)

        # Aggressive memory cleanup between pairs
        clear_gpu_memory(aggressive=True)
        log(f"[per_pair] Pair {pair_idx + 1} complete, memory cleaned")

        # Progressive aggregation
        if pair_idx > 0 and (
            pair_idx % PROGRESSIVE_AGG == 0 or pair_idx == n_pairs - 1
        ):
            process_progressive_agg(ctx, pair_idx, steps)

    # Final analysis (horizon and pair analysis)
    finalize_analysis(ctx)

    return ctx
