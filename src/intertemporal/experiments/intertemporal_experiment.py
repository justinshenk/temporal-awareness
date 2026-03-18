"""Intertemporal preference experiment orchestration."""

from __future__ import annotations

from pathlib import Path

from ...common import profile
from ...common.logging import log, log_progress
from ...inference import COMPONENTS
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
from ..viz import (
    visualize_all_aggregated,
    visualize_att_patching,
    visualize_fine_patching,
    visualize_pair_results,
)
from .experiment_context import ExperimentConfig, ExperimentContext


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

    # Build settings from config
    settings = AttributionSettings(
        components=att_cfg.get("components", ["resid_post"]),
        methods=att_cfg.get("methods", ["standard", "eap"]),
        ig_steps=att_cfg.get("ig_steps", 10),
        grad_at=att_cfg.get("grad_at", "both"),
    )

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

    for component in components:
        ctx.coarse_agg_by_component[component] = CoarseActPatchAggregatedResults()
        all_loaded = True

        for pair_idx, pair in enumerate(ctx.pairs):
            if try_loading_data and ctx.load_coarse_pair(pair_idx, component):
                result = ctx.coarse_patching[(pair_idx, component)]
                ctx.coarse_agg_by_component[component].add(result)
                log(
                    f"[coarse] Loaded cached pair {pair_idx + 1}/{len(ctx.pairs)}, component={component}"
                )
            else:
                all_loaded = False
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

        if all_loaded:
            log(f"[coarse] All pairs loaded from cache for component: {component}")
        ctx.coarse_agg_by_component[component].print_summary()

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
    for component in COMPONENTS:
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

    has_per_pair_results = (
        bool(ctx.att_patching) or bool(ctx.coarse_patching) or bool(ctx.fine_patching)
    )

    # Try loading cached results if needed
    if not has_per_pair_results and try_loading_data:
        for component in components:
            agg = ctx.coarse_agg_by_component.get(component)
            if agg:
                log(
                    f"[viz] Loading {agg.n_samples} per-pair results for {component}..."
                )
                for pair_idx in range(agg.n_samples):
                    if ctx.load_coarse_pair(pair_idx, component):
                        has_per_pair_results = True

    if has_per_pair_results:
        # Determine number of pairs from coarse patching results
        n_pairs = (
            len(ctx.coarse_patching) // len(components) if ctx.coarse_patching else 0
        )

        for pair_idx in range(n_pairs):
            pair_out_dir = ctx.output_dir / f"pair_{pair_idx}"
            pair = ctx.pairs[pair_idx] if pair_idx < len(ctx.pairs) else None

            # Gather coarse results for this pair
            coarse_results = {
                component: ctx.coarse_patching[(pair_idx, component)]
                for component in components
                if (pair_idx, component) in ctx.coarse_patching
            }

            visualize_pair_results(
                pair_idx=pair_idx,
                pair_out_dir=pair_out_dir,
                pair=pair,
                runner=ctx.runner,
                att_result=ctx.att_patching.get(pair_idx),
                coarse_results=coarse_results if coarse_results else None,
                fine_result=ctx.fine_patching.get(pair_idx),
                try_loading_cache=try_loading_data,
                save_token_trees_fn=ctx.save_token_trees,
            )
    else:
        log("[viz] No per-pair results to visualize")

    # Aggregated visualizations with new folder structure
    # Structure: agg/<analysis_slice>/sweep_<component>/... and agg/<slice>/component_comparison/
    agg_out_dir = ctx.output_dir / "agg"
    if ctx.att_agg:
        visualize_att_patching(
            ctx.att_agg.denoising_agg,
            agg_out_dir / "all" / "att_patching" / "denoising",
        )
        visualize_att_patching(
            ctx.att_agg.noising_agg, agg_out_dir / "all" / "att_patching" / "noising"
        )

    # All coarse patching aggregated visualizations (sweep plots + component comparison)
    if ctx.coarse_agg_by_component:
        visualize_all_aggregated(ctx.coarse_agg_by_component, agg_out_dir)

    visualize_fine_patching(ctx.fine_agg, agg_out_dir)


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


def regenerate_all_visualizations(experiments_dir: Path) -> None:
    """Regenerate visualizations for all experiments in the directory.

    Args:
        experiments_dir: Path to the experiments directory containing experiment folders
    """
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.exists():
        log(f"[viz] Experiments directory does not exist: {experiments_dir}")
        return

    # Find all experiment directories (those with pair_0 subdirectory)
    exp_dirs = []
    for d in experiments_dir.iterdir():
        if d.is_dir() and (d / "pair_0").exists():
            exp_dirs.append(d)

    if not exp_dirs:
        log("[viz] No experiment directories found")
        return

    log(f"[viz] Found {len(exp_dirs)} experiment directories")

    for exp_dir in sorted(exp_dirs):
        log(f"\n[viz] Regenerating visualizations for: {exp_dir.name}")
        _regenerate_experiment_visualizations(exp_dir)


def _regenerate_experiment_visualizations(exp_dir: Path) -> None:
    """Regenerate visualizations for a single experiment directory.

    Args:
        exp_dir: Path to the experiment directory
    """
    from ...activation_patching.coarse import CoarseActPatchAggregatedResults

    # Detect available components from cached files
    components = []
    pair_0 = exp_dir / "pair_0"
    if pair_0.exists():
        for d in pair_0.iterdir():
            if d.is_dir() and d.name.startswith("sweep_"):
                comp = d.name.replace("sweep_", "")
                if (d / "coarse_results.json").exists():
                    components.append(comp)

    if not components:
        log(f"[viz] No cached component results found in {exp_dir}")
        return

    log(f"[viz] Found components: {components}")

    # Load aggregated results for each component
    coarse_agg_by_component = {}
    for component in components:
        agg_path = exp_dir / f"coarse_agg_{component}.json"
        if agg_path.exists():
            coarse_agg_by_component[component] = CoarseActPatchAggregatedResults.from_json(
                agg_path
            )
            log(f"[viz] Loaded aggregated results for {component}")

    # Regenerate aggregated visualizations
    if coarse_agg_by_component:
        agg_out_dir = exp_dir / "agg"
        visualize_all_aggregated(coarse_agg_by_component, agg_out_dir)
        log(f"[viz] Regenerated aggregated visualizations")

    # Regenerate per-pair visualizations
    pair_idx = 0
    while True:
        pair_dir = exp_dir / f"pair_{pair_idx}"
        if not pair_dir.exists():
            break

        # Load coarse results for this pair
        from ...activation_patching.coarse import CoarseActPatchResults

        coarse_results = {}
        for component in components:
            results_path = pair_dir / f"sweep_{component}" / "coarse_results.json"
            if results_path.exists():
                coarse_results[component] = CoarseActPatchResults.from_json(results_path)

        if coarse_results:
            visualize_pair_results(
                pair_idx=pair_idx,
                pair_out_dir=pair_dir,
                pair=None,  # No model needed for cache-based regeneration
                runner=None,
                att_result=None,
                coarse_results=coarse_results,
                fine_result=None,
                try_loading_cache=True,
                save_token_trees_fn=None,
            )
            log(f"[viz] Regenerated pair {pair_idx} visualizations")

        pair_idx += 1

    log(f"[viz] Done regenerating {exp_dir.name}")
