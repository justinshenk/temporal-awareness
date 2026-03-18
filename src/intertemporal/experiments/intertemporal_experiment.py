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
from ...attribution_patching import (
    attribute_pair,
    AttrPatchAggregatedResults,
    AttributionSettings,
)

from ..common import get_pref_dataset_dir
from ..preference import generate_preference_data, load_and_merge_preference_data
from .experiment_context import ExperimentConfig, ExperimentContext
from .intertemporal_viz import generate_viz


def _detect_cached_pairs(output_dir: Path, component: str) -> list[int]:
    """Detect all pair indices that have cached results for a component."""
    cached = []
    pair_idx = 0
    while True:
        pair_dir = output_dir / f"pair_{pair_idx}"
        if not pair_dir.exists():
            break
        results_path = pair_dir / f"sweep_{component}" / "coarse_results.json"
        if results_path.exists():
            cached.append(pair_idx)
        pair_idx += 1
    return cached


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
    ctx.pref_data.print_all()


@profile("step_attribution_patching")
def step_attribution_patching(
    ctx: ExperimentContext, try_loading_data: bool = False
) -> None:
    """Run attribution patching on each contrastive pair."""
    att_cfg = ctx.cfg.att_patch
    if not att_cfg.get("enabled", True):
        log("[attr] Attribution patching disabled, skipping")
        return

    if try_loading_data and ctx.load_att_agg():
        log("[attr] Loaded cached aggregated results")
        ctx.att_agg.print_summary()
        return

    # Build settings from config (only override defaults for fields present in att_cfg)
    settings = AttributionSettings.from_dict(att_cfg)

    ctx.att_agg = AttrPatchAggregatedResults()
    for pair_idx, pair in enumerate(ctx.pairs):
        log_progress(pair_idx + 1, len(ctx.pairs), "[attr] Processing pair ")
        result = attribute_pair(ctx.runner, pair, settings=settings)
        ctx.att_patching[pair_idx] = result
        ctx.att_agg.add(result)

    log()
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

    for component in components:
        ctx.coarse_agg_by_component[component] = CoarseActPatchAggregatedResults()

        # Detect all cached pairs first when loading from cache
        cached_pair_indices = set()
        if try_loading_data:
            cached_pair_indices = set(_detect_cached_pairs(ctx.output_dir, component))

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
    """Run targeted activation patching on decomposed targets for each component."""
    if try_loading_data and ctx.load_fine_agg():
        log("[fine] Loaded cached aggregated results")
        ctx.fine_agg.print_summary()
        return

    ctx.fine_agg = ActPatchAggregatedResult()
    for component in PATCHING_COMPONENTS:
        target = ctx.get_union_target(component=component)
        targets = target.decompose()

        for pair_idx, pair in enumerate(ctx.pairs):
            log(
                f"[fine] Processing pair {pair_idx + 1}/{len(ctx.pairs)}, component={component}",
                gap=1,
            )
            pair_result = patch_pair(ctx.runner, pair, targets)
            pair_result.sample_id = pair_idx
            ctx.fine_patching[pair_idx] = pair_result
            ctx.fine_agg.add(pair_result)

    ctx.fine_agg.print_summary()
    ctx.save_fine_agg()


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
    generate_viz(
        ctx.output_dir,
        coarse_agg_by_component=ctx.coarse_agg_by_component or None,
        coarse_patching=ctx.coarse_patching or None,
        att_agg=ctx.att_agg,
        att_patching=ctx.att_patching or None,
        fine_agg=ctx.fine_agg,
        fine_patching=ctx.fine_patching or None,
        pairs=ctx.pairs if ctx._pairs else None,
        runner=ctx.runner if ctx._runner else None,
        save_token_trees_fn=ctx.save_token_trees,
        components=components,
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

    step_visualize_results(ctx, try_loading_data=try_loading_data)

    return ctx
