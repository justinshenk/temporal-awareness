"""Intertemporal preference experiment orchestration."""

from __future__ import annotations

from pathlib import Path

from ...common import profile
from ...common.device_utils import clear_gpu_memory
from ...common.logging import log, log_progress
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
from .attn import run_attn_analysis, AttnAggregatedResults, AttnAnalysisConfig
from .fine import (
    FineGrainedConfig,
    run_fine_analysis,
    compute_attention_patching_correlation,
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
from .experiment_utils import step_load_cfg, cleanup_pair, get_viz_flags


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
        # Enable cached pairs mode so pair count matches cached data, not config
        ctx.enable_cached_pairs()

    if not ctx.pref_data:
        ctx.pref_data, ctx.prompt_dataset = generate_preference_data(
            model=ctx.cfg.model,
            dataset_config=ctx.cfg.dataset_config,
            max_samples=ctx.cfg.max_samples,
            save_data=True,
        )

    print_analysis(analyze_preferences(ctx.pref_data))

    # Generate analysis (always regenerate to ensure consistency with current pairs)
    ctx.save_all_contrastive_prefs()
    ctx.save_all_position_mappings(skip_viz=ctx.only_viz_agg)
    save_horizon_analysis(
        build_horizon_analysis(ctx.pref_pairs), ctx.get_analysis_dir()
    )
    save_pair_analysis(build_pair_analysis(ctx.pref_pairs), ctx.get_analysis_dir())


@profile("step_attrib")
def step_attrib(ctx: ExperimentContext, try_loading_data: bool = False) -> None:
    """Run attribution patching on each contrastive pair."""
    att_cfg = ctx.cfg.attrib_cfg

    config, viz_only = step_load_cfg(
        ctx,
        "attr",
        att_cfg,
        AttributionSettings,
        ctx.make_attrib_viz,
        ctx.load_attrib_agg,
        ctx.unload_attrib_agg,
    )
    if config is None:
        return

    viz_per_pair, viz_agg = get_viz_flags(att_cfg, ctx.viz_enabled)

    if not viz_only:
        ctx.attrib_agg = AttrPatchAggregatedResults()

    # In viz_only mode, always load cached pairs (no_cache only affects computation)
    # In normal mode, respect no_cache flag
    load_cache = viz_only or (try_loading_data and not att_cfg.get("no_cache", False))
    cached = set(ctx.detect_cached_attrib_pairs()) if load_cache else set()

    for pair_idx in sorted(cached):
        if ctx.load_attrib_pair(pair_idx):
            if not viz_only:
                ctx.attrib_agg.add(ctx.attrib_patching[pair_idx])
            if viz_per_pair:
                ctx.viz_attrib_pair(pair_idx)
            log(f"[attr] Loaded cached pair {pair_idx + 1}")
            # Clean up after viz to free memory
            if pair_idx in ctx.attrib_patching:
                del ctx.attrib_patching[pair_idx]

    for pair_idx, pair in enumerate(ctx.pairs):
        if pair_idx in cached:
            continue
        if viz_only:
            continue
        log_progress(pair_idx + 1, len(ctx.pairs), "[attr] Processing pair ")
        result = attribute_pair(ctx.runner, pair, settings=config)
        ctx.attrib_patching[pair_idx] = result
        ctx.attrib_agg.add(result)
        ctx.save_attrib_pair(pair_idx)
        if viz_per_pair:
            ctx.viz_attrib_pair(pair_idx)
        cleanup_pair(ctx.attrib_patching, pair_idx, pair)

    if viz_only:
        if viz_agg:
            ctx.make_attrib_viz(config)()
            log("[attr] Generated visualizations from cache")
        ctx.unload_attrib_agg()
        return

    log(
        f"[attr] Attribution: {len(cached)} cached, {len(ctx.attrib_agg.denoising) + len(ctx.attrib_agg.noising)} total"
    )
    ctx.attrib_agg.print_summary()
    ctx.save_attrib_agg()

    if ctx.attrib_agg.denoising_agg or ctx.attrib_agg.noising_agg:
        ctx.processed_results = ctx.processed_results or ProcessedResults()
        process_attribution_agreement(ctx)
        ctx.save_processed_results()

    if viz_agg:
        ctx.make_attrib_viz(config)()
        log("[attr] Generated attribution visualizations")
    ctx.unload_attrib_agg()


@profile("step_coarse")
def step_coarse(ctx: ExperimentContext, try_loading_data: bool = False) -> None:
    """Run layer and position sweeps on each contrastive pair for each component."""
    coarse_cfg = ctx.cfg.coarse_cfg

    config, viz_only = step_load_cfg(
        ctx,
        "coarse",
        coarse_cfg,
        CoarsePatchingConfig,
        ctx.make_coarse_viz,
        ctx.load_coarse_agg,
        ctx.unload_coarse_agg,
    )
    if config is None:
        return

    viz_per_pair, viz_agg = get_viz_flags(coarse_cfg, ctx.viz_enabled)

    computed_any = False
    # In viz_only mode, always load cached pairs (no_cache only affects computation)
    use_cache = viz_only or (try_loading_data and not coarse_cfg.get("no_cache", False))
    all_cached_pairs: set[int] = set()

    for component in config.components:
        if not viz_only:
            ctx.coarse_agg_by_component[component] = CoarseActPatchAggregatedResults()
        cached = set(ctx.detect_cached_coarse_pairs(component)) if use_cache else set()
        all_cached_pairs.update(cached)

        for pair_idx in sorted(cached):
            if ctx.load_coarse_pair(pair_idx, component):
                if not viz_only:
                    ctx.coarse_agg_by_component[component].add(
                        ctx.coarse_patching[(pair_idx, component)]
                    )
                if viz_per_pair:
                    ctx.viz_coarse_pair(pair_idx, component)
                log(
                    f"[coarse] Loaded cached pair {pair_idx + 1}, component={component}"
                )
                # Clean up after viz to free memory
                key = (pair_idx, component)
                if key in ctx.coarse_patching:
                    del ctx.coarse_patching[key]

        for pair_idx, pair in enumerate(ctx.pairs):
            if pair_idx in cached:
                continue
            if viz_only:
                continue
            computed_any = True
            log(
                f"[coarse] Processing pair {pair_idx + 1}/{len(ctx.pairs)}, component={component}"
            )
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
            ctx.save_coarse_pair(pair_idx, component)
            if viz_per_pair:
                ctx.viz_coarse_pair(pair_idx, component)
            cleanup_pair(ctx.coarse_patching, (pair_idx, component), pair)

        if not viz_only:
            log(
                f"[coarse] Component {component}: {len(cached)} cached, {ctx.coarse_agg_by_component[component].n_samples} total"
            )
            ctx.coarse_agg_by_component[component].print_summary()

    # Generate per-pair component_comparison viz (requires all components)
    if viz_per_pair and all_cached_pairs:
        for pair_idx in sorted(all_cached_pairs):
            # Load all components for this pair
            for component in config.components:
                ctx.load_coarse_pair(pair_idx, component)
            # Generate component_comparison
            ctx.viz_coarse_pair_component_comparison(pair_idx, config.components)
            # Clean up all components for this pair
            for component in config.components:
                key = (pair_idx, component)
                if key in ctx.coarse_patching:
                    del ctx.coarse_patching[key]
        log(f"[coarse] Generated component_comparison for {len(all_cached_pairs)} pairs")

    if viz_only:
        if viz_agg and ctx.coarse_agg_by_component:
            ctx.make_coarse_viz(config)()
            log("[coarse] Generated visualizations from cache")
        ctx.unload_coarse_agg()
        return

    if computed_any:
        ctx.save_coarse_agg()

    if ctx.coarse_agg_by_component:
        ctx.processed_results = ctx.processed_results or ProcessedResults()
        process_coarse_results(ctx)
        ctx.save_processed_results()

    if viz_agg and ctx.coarse_agg_by_component:
        ctx.make_coarse_viz(config)()
        log("[coarse] Generated coarse patching visualizations")
    ctx.unload_coarse_agg()


@profile("step_diffmeans")
def step_diffmeans(ctx: ExperimentContext, try_loading_data: bool = False) -> None:
    """Run difference-in-means analysis on each contrastive pair."""
    diffmeans_cfg = ctx.cfg.diffmeans_cfg

    config, viz_only = step_load_cfg(
        ctx,
        "diffmeans",
        diffmeans_cfg,
        DiffMeansConfig,
        ctx.make_diffmeans_viz,
        ctx.load_diffmeans_agg,
        ctx.unload_diffmeans_agg,
    )
    if config is None:
        return

    viz_per_pair, viz_agg = get_viz_flags(diffmeans_cfg, ctx.viz_enabled)

    if not viz_only:
        ctx.diffmeans_agg = DiffMeansAggregatedResults()

    # In viz_only mode, always load cached pairs (no_cache only affects computation)
    load_cache = viz_only or (try_loading_data and not diffmeans_cfg.get("no_cache", False))
    cached = set(ctx.detect_cached_diffmeans_pairs()) if load_cache else set()

    for pair_idx in sorted(cached):
        if ctx.load_diffmeans_pair(pair_idx):
            if not viz_only:
                ctx.diffmeans_agg.add(ctx.diffmeans_patching[pair_idx])
            if viz_per_pair:
                ctx.viz_diffmeans_pair(pair_idx)
            log(f"[diffmeans] Loaded cached pair {pair_idx + 1}")
            # Clean up after viz to free memory
            if pair_idx in ctx.diffmeans_patching:
                del ctx.diffmeans_patching[pair_idx]

    for pair_idx, pair in enumerate(ctx.pairs):
        if pair_idx in cached:
            continue
        if viz_only:
            continue
        log_progress(pair_idx + 1, len(ctx.pairs), "[diffmeans] Processing pair ")
        clean_mapping, corrupted_mapping = ctx.position_mappings[pair_idx]
        result = run_diffmeans_analysis(
            ctx.runner,
            pair,
            clean_mapping=clean_mapping,
            corrupted_mapping=corrupted_mapping,
            pair_idx=pair_idx,
            positions=config.positions,
        )
        ctx.diffmeans_patching[pair_idx] = result
        ctx.diffmeans_agg.add(result)
        ctx.save_diffmeans_pair(pair_idx)
        if viz_per_pair:
            ctx.viz_diffmeans_pair(pair_idx)
        cleanup_pair(ctx.diffmeans_patching, pair_idx, pair)

    if viz_only:
        if viz_agg:
            ctx.make_diffmeans_viz(config)()
            log("[diffmeans] Generated visualizations from cache")
        ctx.unload_diffmeans_agg()
        return

    log(
        f"[diffmeans] Diffmeans: {len(cached)} cached, {ctx.diffmeans_agg.n_pairs} total"
    )
    ctx.diffmeans_agg.print_summary()

    if viz_agg:
        ctx.make_diffmeans_viz(config)()
        log("[diffmeans] Generated diffmeans visualizations")
    ctx.save_diffmeans_agg()
    ctx.unload_diffmeans_agg()


@profile("step_mlp")
def step_mlp(ctx: ExperimentContext, try_loading_data: bool = False) -> None:
    """Run MLP neuron analysis at key layers."""
    mlp_cfg = ctx.cfg.mlp_cfg

    config, viz_only = step_load_cfg(
        ctx,
        "mlp",
        mlp_cfg,
        MLPAnalysisConfig,
        ctx.make_mlp_viz,
        ctx.load_mlp_agg,
        ctx.unload_mlp_agg,
    )
    if config is None:
        return

    viz_per_pair, viz_agg = get_viz_flags(mlp_cfg, ctx.viz_enabled)

    if not viz_only:
        ctx.mlp_agg = MLPAggregatedResults(layers_analyzed=config.layers)

    # In viz_only mode, always load cached pairs (no_cache only affects computation)
    load_cache = viz_only or (try_loading_data and not mlp_cfg.get("no_cache", False))
    cached = set(ctx.detect_cached_mlp_pairs()) if load_cache else set()

    for pair_idx in sorted(cached):
        if ctx.load_mlp_pair(pair_idx):
            if not viz_only:
                ctx.mlp_agg.add(ctx.mlp[pair_idx])
            if viz_per_pair:
                ctx.viz_mlp_pair(pair_idx)
            log(f"[mlp] Loaded cached pair {pair_idx + 1}")
            # Clean up after viz to free memory
            if pair_idx in ctx.mlp:
                del ctx.mlp[pair_idx]

    for pair_idx, pair in enumerate(ctx.pairs):
        if pair_idx in cached:
            continue
        if viz_only:
            continue
        log_progress(pair_idx + 1, len(ctx.pairs), "[mlp] Processing pair ")
        result = run_mlp_analysis(
            ctx.runner,
            pair,
            pair_idx=pair_idx,
            layers=config.layers,
            n_top_neurons=config.n_top_neurons,
        )
        ctx.mlp[pair_idx] = result
        ctx.mlp_agg.add(result)
        ctx.save_mlp_pair(pair_idx)
        if viz_per_pair:
            ctx.viz_mlp_pair(pair_idx)
        cleanup_pair(ctx.mlp, pair_idx, pair)

    if viz_only:
        if viz_agg:
            ctx.make_mlp_viz(config)()
            log("[mlp] Generated visualizations from cache")
        ctx.unload_mlp_agg()
        return

    log(f"[mlp] MLP Analysis: {len(cached)} cached, {ctx.mlp_agg.n_pairs} total")
    ctx.mlp_agg.print_summary()

    if viz_agg:
        ctx.make_mlp_viz(config)()
        log("[mlp] Generated MLP analysis visualizations")
    ctx.save_mlp_agg()
    ctx.unload_mlp_agg()


@profile("step_attn")
def step_attn(ctx: ExperimentContext, try_loading_data: bool = False) -> None:
    """Run attention pattern analysis at key layers."""
    attn_cfg = ctx.cfg.attn_cfg

    config, viz_only = step_load_cfg(
        ctx,
        "attn",
        attn_cfg,
        AttnAnalysisConfig,
        ctx.make_attn_viz,
        ctx.load_attn_agg,
        ctx.unload_attn_agg,
    )
    if config is None:
        return

    viz_per_pair, viz_agg = get_viz_flags(attn_cfg, ctx.viz_enabled)

    if not viz_only:
        ctx.attn_agg = AttnAggregatedResults(layers_analyzed=config.layers)

    # In viz_only mode, always load cached pairs (no_cache only affects computation)
    load_cache = viz_only or (try_loading_data and not attn_cfg.get("no_cache", False))
    cached = set(ctx.detect_cached_attn_pairs()) if load_cache else set()

    for pair_idx in sorted(cached):
        if ctx.load_attn_pair(pair_idx):
            if not viz_only:
                ctx.attn_agg.add(ctx.attn[pair_idx])
            if viz_per_pair:
                mapping = ctx.get_position_mapping(pair_idx)
                ctx.viz_attn_pair(pair_idx, mapping)
            log(f"[attn] Loaded cached pair {pair_idx + 1}")
            # Clean up after viz to free memory
            if pair_idx in ctx.attn:
                del ctx.attn[pair_idx]

    for pair_idx, pair in enumerate(ctx.pairs):
        if pair_idx in cached:
            continue
        if viz_only:
            continue
        mapping = ctx.get_position_mapping(pair_idx)
        log_progress(pair_idx + 1, len(ctx.pairs), "[attn] Processing pair ")
        if mapping is None:
            log(f"[attn] Pair {pair_idx}: No position mapping found, skipping")
            continue
        result = run_attn_analysis(
            ctx.runner, pair, mapping, pair_idx=pair_idx, config=config
        )
        ctx.attn[pair_idx] = result
        ctx.attn_agg.add(result)
        ctx.save_attn_pair(pair_idx, store_patterns=config.store_patterns)
        if viz_per_pair:
            ctx.viz_attn_pair(pair_idx, mapping)
        pair.pop_heavy()
        clear_gpu_memory(aggressive=True)

    if viz_only:
        if viz_agg:
            ctx.make_attn_viz(config)()
            log("[attn] Generated visualizations from cache")
        ctx.unload_attn_agg()
        return

    log(
        f"[attn] Attention Analysis: {len(cached)} cached, {ctx.attn_agg.n_pairs} total"
    )
    ctx.attn_agg.print_summary()

    if viz_agg:
        ctx.make_attn_viz(config)()
        log("[attn] Generated attention analysis visualizations")
    ctx.save_attn_agg()
    ctx.unload_attn_agg()


@profile("step_fine")
def step_fine(ctx: ExperimentContext, try_loading_data: bool = False) -> None:
    """Run unified fine-grained patching analysis."""
    fine_cfg = ctx.cfg.fine_cfg

    config, viz_only = step_load_cfg(
        ctx,
        "fine",
        fine_cfg,
        FineGrainedConfig,
        ctx.make_fine_viz,
        ctx.load_fine_for_viz,
        ctx.unload_fine_agg,
    )
    if config is None:
        return

    viz_per_pair, viz_agg = get_viz_flags(fine_cfg, ctx.viz_enabled)

    # In viz_only mode, always load cached pairs (no_cache only affects computation)
    load_cache = viz_only or (try_loading_data and not fine_cfg.get("no_cache", False))
    cached = set(ctx.detect_cached_fine_pairs()) if load_cache else set()

    for pair_idx in sorted(cached):
        if ctx.load_fine_pair(pair_idx):
            if viz_per_pair:
                mapping = ctx.get_position_mapping(pair_idx)
                ctx.viz_fine_pair(pair_idx, mapping)
            log(f"[fine] Loaded cached pair {pair_idx + 1}")
            # Clean up after viz to free memory
            if pair_idx in ctx.fine:
                del ctx.fine[pair_idx]

    for pair_idx, pair in enumerate(ctx.pairs):
        if pair_idx in cached:
            continue
        if viz_only:
            continue
        mapping = ctx.get_position_mapping(pair_idx)
        log_progress(pair_idx + 1, len(ctx.pairs), "[fine] Processing pair ")
        result = run_fine_analysis(ctx.runner, pair, config)
        result.sample_id = pair_idx
        ctx.fine[pair_idx] = result
        ctx.save_fine_pair(pair_idx)
        if viz_per_pair:
            ctx.viz_fine_pair(pair_idx, mapping)
        pair.pop_heavy()
        clear_gpu_memory(aggressive=True)

    if viz_only:
        if viz_agg:
            ctx.make_fine_viz(config)()
            log("[fine] Generated visualizations from cache")
        ctx.unload_fine_agg()
        return

    log(
        f"[fine] Fine patching: {len(cached)} cached, {len(ctx.fine)} total"
    )

    for pair_idx in ctx.fine:
        fine_result = ctx.fine[pair_idx]
        attn_result = ctx.attn.get(pair_idx)
        if fine_result.head_sweep and attn_result:
            fine_result.attention_correlations = compute_attention_patching_correlation(
                fine_result.head_sweep, attn_result, n_heads=10
            )
            ctx.save_fine_pair(pair_idx)
            log(f"[fine] Computed attention-patching correlation for pair {pair_idx}")

    if viz_agg:
        ctx.make_fine_viz(config)()
        log("[fine] Generated fine-grained visualizations")
    ctx.unload_fine_agg()


@profile("run_experiment")
def run_experiment(
    cfg: ExperimentConfig,
    try_loading_data: bool = False,
    output_dir: Path | None = None,
    backend: str | None = None,
) -> ExperimentContext:
    """Run full experiment."""

    ctx = ExperimentContext(cfg, output_dir=output_dir, backend=backend)

    # Save config for future cache runs
    cfg.save(ctx.output_dir)

    step_preference_data(ctx, try_loading_data=True)

    step_attrib(ctx, try_loading_data=try_loading_data)

    step_coarse(ctx, try_loading_data=try_loading_data)

    step_diffmeans(ctx, try_loading_data=try_loading_data)

    step_mlp(ctx, try_loading_data=try_loading_data)

    step_attn(ctx, try_loading_data=try_loading_data)

    step_fine(ctx, try_loading_data=try_loading_data)

    return ctx
