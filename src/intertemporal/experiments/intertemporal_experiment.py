"""Intertemporal preference experiment orchestration."""

from __future__ import annotations

from pathlib import Path

from ...common import profile
from ...common.logging import log, log_progress
from ...common.patching_types import PATCHING_COMPONENTS
from ...activation_patching import patch_pair, ActPatchAggregatedResult
from ...activation_patching.coarse import (
    run_coarse_act_patching,
    CoarseActPatchAggregatedResults,
)
from ...activation_patching.fine import (
    run_fine_patching,
    FineConfig,
    FinePatchingResults,
)
from ...attribution_patching import (
    attribute_pair,
    AttrPatchAggregatedResults,
    AttributionSettings,
)
from .diffmeans import run_diffmeans_analysis, DiffMeansAggregatedResults
from .geo import run_geo_analysis, GeoAggregatedResults
from .processing import (
    ProcessedResults,
    process_attribution_agreement,
    process_coarse_results,
)

from ..common import get_pref_dataset_dir
from ..preference import (
    generate_preference_data,
    load_and_merge_preference_data,
    analyze_preferences,
    print_analysis,
)
from .experiment_context import ExperimentConfig, ExperimentContext
from .horizon_analysis import build_horizon_analysis, save_horizon_analysis
from .pair_analysis import build_pair_analysis, save_pair_analysis
from .intertemporal_viz import generate_viz


@profile("step_preference_data")
def step_preference_data(
    ctx: ExperimentContext, try_loading_data: bool = False
) -> None:
    """Load or generate preference data."""
    if try_loading_data:
        ctx.pref_data = load_and_merge_preference_data(
            ctx.cfg.get_prefix(), get_pref_dataset_dir()
        )
    if not ctx.pref_data:
        ctx.pref_data = generate_preference_data(
            model=ctx.cfg.model,
            dataset_config=ctx.cfg.dataset_config,
            max_samples=ctx.cfg.max_samples,
            save_data=True,
        )
    analysis = analyze_preferences(ctx.pref_data)
    print_analysis(analysis)

    # Save contrastive preferences for each pair
    ctx.save_all_contrastive_prefs()

    # Build and save horizon analysis
    horizon_analysis = build_horizon_analysis(ctx.pref_pairs)
    save_horizon_analysis(horizon_analysis, ctx.output_dir)

    # Build and save pair analysis (labels and order)
    pair_analysis = build_pair_analysis(ctx.pref_pairs)
    save_pair_analysis(pair_analysis, ctx.output_dir)


@profile("step_attribution_patching")
def step_attribution_patching(
    ctx: ExperimentContext, try_loading_data: bool = False
) -> None:
    """Run attribution patching on each contrastive pair."""
    att_cfg = ctx.cfg.att_patch
    if not att_cfg.get("enabled", True):
        log("[attr] Attribution patching disabled, skipping")
        return

    # Build settings from config (only override defaults for fields present in att_cfg)
    settings = AttributionSettings.from_dict(att_cfg)

    ctx.att_agg = AttrPatchAggregatedResults()

    # Detect cached pairs first when loading from cache (unless no_cache is set)
    cached_pair_indices = set()
    use_cache = try_loading_data and not att_cfg.get("no_cache", False)
    if use_cache:
        cached_pair_indices = set(ctx.detect_cached_att_pairs())

    # Load all cached pairs
    for pair_idx in sorted(cached_pair_indices):
        if ctx.load_att_pair(pair_idx):
            result = ctx.att_patching[pair_idx]
            ctx.att_agg.add(result)
            log(f"[attr] Loaded cached pair {pair_idx + 1}")

    # Process any new pairs that aren't cached
    for pair_idx, pair in enumerate(ctx.pairs):
        if pair_idx in cached_pair_indices:
            continue  # Already loaded from cache

        log_progress(pair_idx + 1, len(ctx.pairs), "[attr] Processing pair ")
        result = attribute_pair(ctx.runner, pair, settings=settings)
        ctx.att_patching[pair_idx] = result
        ctx.att_agg.add(result)
        ctx.save_att_pair(pair_idx)

    n_loaded = len(cached_pair_indices)
    n_total = len(ctx.att_agg.denoising) + len(ctx.att_agg.noising)
    log(f"[attr] Attribution: {n_loaded} loaded from cache, {n_total} total")
    ctx.att_agg.print_summary()
    ctx.save_att_agg()


@profile("step_coarse_activation_patching")
def step_coarse_activation_patching(
    ctx: ExperimentContext, try_loading_data: bool = False
) -> None:
    """Run layer and position sweeps on each contrastive pair for each component."""
    coarse_cfg = ctx.cfg.coarse_patch
    if not coarse_cfg.get("enabled", True):
        log("[coarse] Coarse patching disabled, skipping")
        return

    components = coarse_cfg.get("components", [])
    computed_any = False

    # Check if cache should be skipped for this step
    use_cache = try_loading_data and not coarse_cfg.get("no_cache", False)

    for component in components:
        ctx.coarse_agg_by_component[component] = CoarseActPatchAggregatedResults()

        # Detect all cached pairs first when loading from cache
        cached_pair_indices = set()
        if use_cache:
            cached_pair_indices = set(ctx.detect_cached_coarse_pairs(component))

        # Load all cached pairs (may be more than len(ctx.pairs))
        for pair_idx in sorted(cached_pair_indices):
            if ctx.load_coarse_pair(pair_idx, component):
                result = ctx.coarse_patching[(pair_idx, component)]
                ctx.coarse_agg_by_component[component].add(result)
                log(
                    f"[coarse] Loaded cached pair {pair_idx + 1}, component={component}"
                )

        # Process any new pairs that aren't cached
        for pair_idx, pair in enumerate(ctx.pairs):
            if pair_idx in cached_pair_indices:
                continue  # Already loaded from cache

            computed_any = True
            log(
                f"[coarse] Processing pair {pair_idx + 1}/{len(ctx.pairs)}, component={component}"
            )
            result = run_coarse_act_patching(
                ctx.runner,
                pair,
                component=component,
                layer_step_sizes=coarse_cfg.get("layer_steps"),
                pos_step_sizes=coarse_cfg.get("pos_steps"),
            )
            result.sample_id = pair_idx
            ctx.coarse_patching[(pair_idx, component)] = result
            ctx.coarse_agg_by_component[component].add(result)
            ctx.save_coarse_pair(pair_idx, component)

        n_loaded = len(cached_pair_indices)
        n_total = ctx.coarse_agg_by_component[component].n_samples
        log(f"[coarse] Component {component}: {n_loaded} loaded, {n_total} total")
        ctx.coarse_agg_by_component[component].print_summary()

    # Only save agg if we computed new results
    if computed_any:
        ctx.save_coarse_agg()


@profile("step_fine_activation_patching")
def step_fine_activation_patching(
    ctx: ExperimentContext, try_loading_data: bool = False
) -> None:
    """Run fine-grained activation patching: head-level and neuron-level analysis.

    Performs:
    1. Head-level patching at key attention layers (L24, L21, L19, L29, L30)
    2. MLP neuron analysis at key MLP layers (L31, L24, L28)
    3. Attention pattern analysis for top heads

    Configuration comes from ctx.cfg.fine_patch dict with keys:
    - enabled: bool (default True)
    - head_layers: list[int] (default [24, 21, 19, 29, 30])
    - mlp_layers: list[int] (default [31, 24, 28])
    - n_top_heads: int (default 5)
    - n_top_neurons: int (default 20)
    - source_positions: list[int] (default [86, 87, 88])
    - destination_positions: list[int] (default [143, 144, 145])
    """
    fine_cfg = ctx.cfg.fine_patch if hasattr(ctx.cfg, "fine_patch") else {}
    if not fine_cfg.get("enabled", False):
        log("[fine] Fine patching disabled, skipping")
        return

    if try_loading_data and ctx.load_fine_agg():
        log("[fine] Loaded cached aggregated results")
        return

    # Build FineConfig from config dict
    config = FineConfig(
        head_layers=fine_cfg.get("head_layers", [24, 21, 19, 29, 30]),
        mlp_layers=fine_cfg.get("mlp_layers", [31, 24, 28]),
        n_top_heads=fine_cfg.get("n_top_heads", 5),
        n_top_neurons=fine_cfg.get("n_top_neurons", 20),
        source_positions=fine_cfg.get("source_positions", [86, 87, 88]),
        destination_positions=fine_cfg.get("destination_positions", [143, 144, 145]),
    )

    ctx.fine_results: list[FinePatchingResults] = []

    for pair_idx, pair in enumerate(ctx.pairs):
        log_progress(pair_idx + 1, len(ctx.pairs), "[fine] Processing pair ")

        result = run_fine_patching(ctx.runner, pair, config)
        result.sample_id = pair_idx
        ctx.fine_results.append(result)
        result.print_summary()

    log(f"[fine] Completed fine patching for {len(ctx.fine_results)} pairs")


@profile("step_diffmeans")
def step_diffmeans(
    ctx: ExperimentContext, try_loading_data: bool = False
) -> None:
    """Run difference-in-means analysis on each contrastive pair."""
    diffmeans_cfg = ctx.cfg.diffmeans
    if not diffmeans_cfg.get("enabled", True):
        log("[diffmeans] Diffmeans analysis disabled, skipping")
        return

    # Get additional positions from config (e.g., [86, 87, 88, 145])
    additional_positions = diffmeans_cfg.get("positions", None)

    ctx.diffmeans_agg = DiffMeansAggregatedResults()

    # Detect cached pairs first (unless no_cache is set)
    cached_pair_indices = set()
    use_cache = try_loading_data and not diffmeans_cfg.get("no_cache", False)
    if use_cache:
        cached_pair_indices = set(ctx.detect_cached_diffmeans_pairs())

    # Load all cached pairs
    for pair_idx in sorted(cached_pair_indices):
        if ctx.load_diffmeans_pair(pair_idx):
            result = ctx.diffmeans_patching[pair_idx]
            ctx.diffmeans_agg.add(result)
            log(f"[diffmeans] Loaded cached pair {pair_idx + 1}")

    # Process any new pairs that aren't cached
    for pair_idx, pair in enumerate(ctx.pairs):
        if pair_idx in cached_pair_indices:
            continue

        log_progress(pair_idx + 1, len(ctx.pairs), "[diffmeans] Processing pair ")
        result = run_diffmeans_analysis(
            ctx.runner,
            pair,
            pair_idx=pair_idx,
            additional_positions=additional_positions,
        )
        ctx.diffmeans_patching[pair_idx] = result
        ctx.diffmeans_agg.add(result)
        ctx.save_diffmeans_pair(pair_idx)

    n_loaded = len(cached_pair_indices)
    n_total = ctx.diffmeans_agg.n_pairs
    log(f"[diffmeans] Diffmeans: {n_loaded} loaded from cache, {n_total} total")
    ctx.diffmeans_agg.print_summary()
    ctx.save_diffmeans_agg()


@profile("step_geo")
def step_geo(
    ctx: ExperimentContext, try_loading_data: bool = False
) -> None:
    """Run geometric (PCA) analysis on residual stream activations."""
    geo_cfg = ctx.cfg.geo
    if not geo_cfg.get("enabled", False):
        log("[geo] Geo analysis disabled, skipping")
        return

    ctx.geo_agg = GeoAggregatedResults()

    # Get analysis parameters from config
    positions = geo_cfg.get("positions", None)  # None = last token only
    layers = geo_cfg.get("layers", None)  # None = all layers
    n_components = geo_cfg.get("n_components", 3)

    # Detect cached pairs first (unless no_cache is set)
    cached_pair_indices = set()
    use_cache = try_loading_data and not geo_cfg.get("no_cache", False)
    if use_cache:
        cached_pair_indices = set(ctx.detect_cached_geo_pairs())

    # Load all cached pairs
    for pair_idx in sorted(cached_pair_indices):
        if ctx.load_geo_pair(pair_idx):
            result = ctx.geo_patching[pair_idx]
            ctx.geo_agg.add(result)
            log(f"[geo] Loaded cached pair {pair_idx + 1}")

    # Process any new pairs that aren't cached
    for pair_idx, pair in enumerate(ctx.pairs):
        if pair_idx in cached_pair_indices:
            continue

        log_progress(pair_idx + 1, len(ctx.pairs), "[geo] Processing pair ")
        result = run_geo_analysis(
            ctx.runner,
            pair,
            pair_idx=pair_idx,
            positions=positions,
            layers=layers,
            n_components=n_components,
        )
        ctx.geo_patching[pair_idx] = result
        ctx.geo_agg.add(result)
        ctx.save_geo_pair(pair_idx)

    n_loaded = len(cached_pair_indices)
    n_total = ctx.geo_agg.n_pairs
    log(f"[geo] Geo: {n_loaded} loaded from cache, {n_total} total")
    ctx.geo_agg.print_summary()
    ctx.save_geo_agg()


@profile("step_process_results")
def step_process_results(ctx: ExperimentContext) -> None:
    """Process raw results into structured analysis results.

    This step runs all algorithmic analysis (circuit extraction, redundancy
    analysis, etc.) and stores the results in ctx.processed_results. The
    visualization step then uses these pre-computed results.

    Always recomputes results (no caching) since processing is fast.
    """
    log("[process] Processing results...")
    ctx.processed_results = ProcessedResults()

    # Process coarse patching results if available
    if ctx.coarse_agg_by_component:
        process_coarse_results(ctx)

    # Attribution method agreement analysis (independent of coarse patching)
    if ctx.att_agg and (ctx.att_agg.denoising_agg or ctx.att_agg.noising_agg):
        process_attribution_agreement(ctx)

    if not ctx.processed_results.component_comparison and not ctx.processed_results.attribution_agreement:
        log("[process] No results to process")
        return

    ctx.save_processed_results()
    log("[process] Done processing results")


@profile("step_visualize_results")
def step_visualize_results(
    ctx: ExperimentContext, try_loading_data: bool = False
) -> None:
    """Visualize all patching results."""
    if not ctx.cfg.viz.get("enabled", True):
        log("[viz] Visualization disabled, skipping")
        return

    components = ctx.cfg.coarse_patch.get("components", ["resid_post"])

    # Load cached per-pair results if needed
    if not ctx.coarse_patching and try_loading_data:
        for component in components:
            agg = ctx.coarse_agg_by_component.get(component)
            if agg:
                log(
                    f"[viz] Loading {agg.n_samples} per-pair results for {component}..."
                )
                for pair_idx in range(agg.n_samples):
                    ctx.load_coarse_pair(pair_idx, component)

    # Use shared generate_viz with in-memory data
    only_agg = ctx.cfg.viz.get("only_agg", False)
    generate_viz(
        ctx.output_dir,
        coarse_agg_by_component=ctx.coarse_agg_by_component or None,
        coarse_patching=ctx.coarse_patching or None,
        att_agg=ctx.att_agg,
        att_patching=ctx.att_patching or None,
        fine_agg=ctx.fine_agg,
        fine_patching=ctx.fine_patching or None,
        diffmeans_agg=ctx.diffmeans_agg,
        diffmeans_patching=ctx.diffmeans_patching or None,
        geo_agg=ctx.geo_agg,
        geo_patching=ctx.geo_patching or None,
        processed_results=ctx.processed_results,
        pairs=ctx.pairs if ctx._pairs else None,
        pref_pairs=ctx.pref_pairs if ctx._pref_pairs else None,
        runner=ctx.runner if ctx._runner else None,
        save_token_trees_fn=ctx.save_token_trees,
        components=components,
        only_agg=only_agg,
    )


@profile("run_experiment")
def run_experiment(
    cfg: ExperimentConfig,
    try_loading_data: bool = False,
    output_dir: Path | None = None,
    backend: str | None = None,
) -> ExperimentContext:
    """Run full experiment.

    Args:
        cfg: Experiment configuration
        try_loading_data: If True, try loading cached data before recomputing
        output_dir: Optional custom output directory (overrides default)
        backend: Optional backend override (pyvene, transformerlens, huggingface, nnsight)
    """
    ctx = ExperimentContext(cfg, output_dir=output_dir, backend=backend)

    step_preference_data(ctx, try_loading_data=try_loading_data)

    step_attribution_patching(ctx, try_loading_data=try_loading_data)

    step_coarse_activation_patching(ctx, try_loading_data=try_loading_data)

    step_diffmeans(ctx, try_loading_data=try_loading_data)

    step_geo(ctx, try_loading_data=try_loading_data)

    step_process_results(ctx)

    step_visualize_results(ctx, try_loading_data=try_loading_data)

    return ctx
