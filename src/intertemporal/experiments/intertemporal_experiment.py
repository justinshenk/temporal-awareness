"""Intertemporal preference experiment orchestration."""

from __future__ import annotations


from ...common import profile
from ...common.profiler import P
from ...inference import COMPONENTS
from ...activation_patching import patch_pair, ActPatchAggregatedResult
from ...activation_patching.coarse import (
    run_coarse_act_patching,
    CoarseActPatchAggregatedResults,
)
from ...attribution_patching import attribute_pair, AttrPatchAggregatedResults

from ..common import get_pref_dataset_dir
from ..preference import generate_preference_data, load_and_merge_preference_data
from ..viz import (
    visualize_att_patching,
    visualize_coarse_patching,
    visualize_fine_patching,
    visualize_tokenization,
)
from ...viz.token_coloring import get_token_coloring_for_pair

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

    print(ctx.pref_data)
    ctx.pref_data.print_summary()


@profile("step_attribution_patching")
def step_attribution_patching(
    ctx: ExperimentContext, try_loading_data: bool = False
) -> None:
    """Run attribution patching on each contrastive pair."""
    if try_loading_data and ctx.load_att_agg():
        print("[attr] Loaded cached aggregated results")
        ctx.att_agg.print_summary()
        return

    ctx.att_agg = AttrPatchAggregatedResults()

    for pair_idx, pair in enumerate(ctx.pairs):
        print(f"\n[attr] Processing pair {pair_idx + 1}/{len(ctx.pairs)}")
        result = attribute_pair(ctx.runner, pair)
        ctx.att_patching[pair_idx] = result
        ctx.att_agg.add(result)

    ctx.att_agg.print_summary()
    ctx.save_att_agg()


@profile("step_coarse_activation_patching")
def step_coarse_activation_patching(
    ctx: ExperimentContext, try_loading_data: bool = False
) -> None:
    """Run layer and position sweeps on each contrastive pair."""
    if try_loading_data and ctx.load_coarse_agg():
        print("[coarse] Loaded cached aggregated results")
        ctx.coarse_agg.print_summary()
        return

    ctx.coarse_agg = CoarseActPatchAggregatedResults()

    for pair_idx, pair in enumerate(ctx.pairs):
        print(f"\n[coarse] Processing pair {pair_idx + 1}/{len(ctx.pairs)}")
        with P("run_coarse_act_patching"):
            result = run_coarse_act_patching(ctx.runner, pair)
        result.sample_id = pair_idx
        ctx.coarse_patching[pair_idx] = result
        ctx.coarse_agg.add(result)
        # Save per-pair results for re-visualization
        with P("save_coarse_pair"):
            ctx.save_coarse_pair(pair_idx)

    with P("print_summary"):
        ctx.coarse_agg.print_summary()
    with P("save_coarse_agg"):
        ctx.save_coarse_agg()


@profile("step_fine_activation_patching")
def step_fine_activation_patching(
    ctx: ExperimentContext, try_loading_data: bool = False
) -> None:
    """Run targeted activation patching on decomposed targets for each component."""
    if try_loading_data and ctx.load_fine_agg():
        print("[fine] Loaded cached aggregated results")
        ctx.fine_agg.print_summary()
        return

    ctx.fine_agg = ActPatchAggregatedResult()

    for component in COMPONENTS:
        target = ctx.get_union_target(component=component)
        targets = target.decompose()

        for pair_idx, pair in enumerate(ctx.pairs):
            print(
                f"\n[fine] Processing pair {pair_idx + 1}/{len(ctx.pairs)}, component={component}"
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

    # Check if we have any per-pair results to visualize
    has_per_pair_results = (
        bool(ctx.att_patching) or bool(ctx.coarse_patching) or bool(ctx.fine_patching)
    )

    # Try to load per-pair results from cache if we have aggregated results
    if not has_per_pair_results and try_loading_data and ctx.coarse_agg:
        n_samples = ctx.coarse_agg.n_samples
        print(f"[viz] Attempting to load {n_samples} per-pair results from cache...")
        for pair_idx in range(n_samples):
            if ctx.load_coarse_pair(pair_idx):
                has_per_pair_results = True

    # Only iterate over pairs if we have per-pair results (avoids expensive pair building)
    if has_per_pair_results:
        for pair_idx, pair in enumerate(ctx.pairs):
            pair_out_dir = ctx.output_dir / f"pair_{pair_idx}"
            with P("get_token_coloring"):
                coloring = get_token_coloring_for_pair(pair)
                position_labels = coloring.get_position_labels("short")
                section_markers = coloring.get_section_markers("short")

            with P("save_token_trees"):
                ctx.save_token_trees(pair_idx, pair, pair_out_dir)

            with P("visualize_tokenization"):
                visualize_tokenization([pair], ctx.runner, pair_out_dir, max_pairs=1)

            # Per-pair patching visualizations
            if pair_idx in ctx.att_patching:
                pair_result = ctx.att_patching[pair_idx]
                if pair_result.result.denoising:
                    with P("visualize_att_patching"):
                        visualize_att_patching(
                            pair_result.result.denoising,
                            pair_out_dir / "denoising",
                            position_labels,
                            section_markers,
                        )
                if pair_result.result.noising:
                    with P("visualize_att_patching"):
                        visualize_att_patching(
                            pair_result.result.noising,
                            pair_out_dir / "noising",
                            position_labels,
                            section_markers,
                        )
            if pair_idx in ctx.coarse_patching:
                with P("visualize_coarse_patching"):
                    visualize_coarse_patching(
                        ctx.coarse_patching[pair_idx], pair_out_dir, coloring, pair=pair
                    )
            if pair_idx in ctx.fine_patching:
                with P("visualize_fine_patching"):
                    visualize_fine_patching(
                        ctx.fine_patching[pair_idx],
                        pair_out_dir,
                        position_labels,
                        section_markers,
                    )
    else:
        print("[viz] No per-pair patching results to visualize (loaded from cache)")

    # Aggregated visualizations
    agg_out_dir = ctx.output_dir / "agg"
    if ctx.att_agg:
        with P("visualize_att_agg"):
            visualize_att_patching(ctx.att_agg.denoising_agg, agg_out_dir / "denoising")
            visualize_att_patching(ctx.att_agg.noising_agg, agg_out_dir / "noising")
    with P("visualize_coarse_agg"):
        visualize_coarse_patching(ctx.coarse_agg, agg_out_dir)
    with P("visualize_fine_agg"):
        visualize_fine_patching(ctx.fine_agg, agg_out_dir)


@profile("run_experiment")
def run_experiment(cfg: ExperimentConfig) -> ExperimentContext:
    """Run full experiment."""
    ctx = ExperimentContext(cfg)
    step_preference_data(ctx, try_loading_data=cfg.try_loading_data)

    # step_attribution_patching(ctx, try_loading_data=cfg.try_loading_data)

    step_coarse_activation_patching(ctx, try_loading_data=cfg.try_loading_data)

    # Only check for pairs if we don't have any results yet (needed for patching)
    if not ctx.coarse_agg and not ctx.att_agg and not ctx.fine_agg:
        if not ctx.pairs:
            print("No preference pairs!")
            return ctx

    # step_fine_activation_patching(ctx, try_loading_data=cfg.try_loading_data)

    step_visualize_results(ctx, try_loading_data=cfg.try_loading_data)

    return ctx
