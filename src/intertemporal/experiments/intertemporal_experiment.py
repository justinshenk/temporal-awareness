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
from ...attribution_patching import attribute_pair, AttrPatchAggregatedResults
from ...viz.token_coloring import get_token_coloring_for_pair

from ..common import get_pref_dataset_dir
from ..preference import generate_preference_data, load_and_merge_preference_data
from ..viz import (
    visualize_att_patching,
    visualize_coarse_patching,
    visualize_fine_patching,
    visualize_tokenization,
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
    if try_loading_data and ctx.load_att_agg():
        log("[attr] Loaded cached aggregated results")
        ctx.att_agg.print_summary()
        return

    ctx.att_agg = AttrPatchAggregatedResults()
    for pair_idx, pair in enumerate(ctx.pairs):
        log_progress(pair_idx + 1, len(ctx.pairs), "[attr] Processing pair ")
        result = attribute_pair(ctx.runner, pair)
        ctx.att_patching[pair_idx] = result
        ctx.att_agg.add(result)

    log()
    ctx.att_agg.print_summary()
    ctx.save_att_agg()


@profile("step_coarse_activation_patching")
def step_coarse_activation_patching(
    ctx: ExperimentContext, try_loading_data: bool = False
) -> None:
    """Run layer and position sweeps on each contrastive pair."""
    if try_loading_data and ctx.load_coarse_agg():
        log("[coarse] Loaded cached aggregated results")
        ctx.coarse_agg.print_summary()
        return

    ctx.coarse_agg = CoarseActPatchAggregatedResults()
    for pair_idx, pair in enumerate(ctx.pairs):
        log(f"[coarse] Processing pair {pair_idx + 1}/{len(ctx.pairs)}", gap=1)
        result = run_coarse_act_patching(
            ctx.runner,
            pair,
            component=ctx.cfg.coarse_patch.get("component", "resid_post"),
            layer_step_sizes=ctx.cfg.coarse_patch.get("layer_steps"),
            pos_step_sizes=ctx.cfg.coarse_patch.get("pos_steps"),
        )
        result.sample_id = pair_idx
        ctx.coarse_patching[pair_idx] = result
        ctx.coarse_agg.add(result)
        ctx.save_coarse_pair(pair_idx)

    ctx.coarse_agg.print_summary()
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
    has_per_pair_results = (
        bool(ctx.att_patching) or bool(ctx.coarse_patching) or bool(ctx.fine_patching)
    )

    if not has_per_pair_results and try_loading_data and ctx.coarse_agg:
        n_samples = ctx.coarse_agg.n_samples
        log(f"[viz] Loading {n_samples} per-pair results from cache...")
        for pair_idx in range(n_samples):
            if ctx.load_coarse_pair(pair_idx):
                has_per_pair_results = True

    if has_per_pair_results:
        for pair_idx, pair in enumerate(ctx.pairs):
            pair_out_dir = ctx.output_dir / f"pair_{pair_idx}"
            coloring = get_token_coloring_for_pair(pair)
            position_labels = coloring.get_position_labels("short")
            section_markers = coloring.get_section_markers("short")

            ctx.save_token_trees(pair_idx, pair, pair_out_dir)
            visualize_tokenization([pair], ctx.runner, pair_out_dir, max_pairs=1)

            if pair_idx in ctx.att_patching:
                pair_result = ctx.att_patching[pair_idx]
                if pair_result.result.denoising:
                    visualize_att_patching(
                        pair_result.result.denoising,
                        pair_out_dir / "denoising",
                        position_labels,
                        section_markers,
                    )
                if pair_result.result.noising:
                    visualize_att_patching(
                        pair_result.result.noising,
                        pair_out_dir / "noising",
                        position_labels,
                        section_markers,
                    )

            if pair_idx in ctx.coarse_patching:
                visualize_coarse_patching(
                    ctx.coarse_patching[pair_idx], pair_out_dir, coloring, pair=pair
                )

            if pair_idx in ctx.fine_patching:
                visualize_fine_patching(
                    ctx.fine_patching[pair_idx],
                    pair_out_dir,
                    position_labels,
                    section_markers,
                )
    else:
        log("[viz] No per-pair results to visualize")

    agg_out_dir = ctx.output_dir / "agg"
    if ctx.att_agg:
        visualize_att_patching(ctx.att_agg.denoising_agg, agg_out_dir / "denoising")
        visualize_att_patching(ctx.att_agg.noising_agg, agg_out_dir / "noising")
    visualize_coarse_patching(ctx.coarse_agg, agg_out_dir)
    visualize_fine_patching(ctx.fine_agg, agg_out_dir)


@profile("run_experiment")
def run_experiment(
    cfg: ExperimentConfig,
    try_loading_data: bool = False,
    output_dir: Path | None = None,
) -> ExperimentContext:
    """Run full experiment.

    Args:
        cfg: Experiment configuration
        try_loading_data: If True, try loading cached data before recomputing
        output_dir: Optional custom output directory (overrides default)
    """
    ctx = ExperimentContext(cfg, output_dir=output_dir)

    step_preference_data(ctx, try_loading_data=try_loading_data)

    step_coarse_activation_patching(ctx, try_loading_data=try_loading_data)

    step_visualize_results(ctx, try_loading_data=try_loading_data)

    return ctx
