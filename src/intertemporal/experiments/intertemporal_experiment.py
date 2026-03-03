"""Intertemporal preference experiment orchestration."""

from __future__ import annotations


from ...common import profile
from ...inference import InternalsConfig, COMPONENTS
from ...activation_patching import patch_pair, ActPatchAggregatedResult
from ...activation_patching.coarse_activation_patching import (
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
            internals_config=InternalsConfig.from_dict(ctx.cfg.internals_config)
            if ctx.cfg.internals_config
            else None,
            max_samples=ctx.cfg.max_samples,
        )

    print(ctx.pref_data)
    ctx.pref_data.print_summary()


@profile("step_attribution_patching")
def step_attribution_patching(ctx: ExperimentContext) -> None:
    """Run attribution patching on each contrastive pair."""
    ctx.att_agg = AttrPatchAggregatedResults()

    for pair_idx, pair in enumerate(ctx.pairs):
        print(f"\n[attr] Processing pair {pair_idx + 1}/{len(ctx.pairs)}")
        result = attribute_pair(ctx.runner, pair)
        ctx.att_patching[pair_idx] = result
        ctx.att_agg.add(result)

    ctx.att_agg.print_summary()


@profile("step_coarse_activation_patching")
def step_coarse_activation_patching(ctx: ExperimentContext) -> None:
    """Run layer and position sweeps on each contrastive pair."""
    ctx.coarse_agg = CoarseActPatchAggregatedResults()

    for pair_idx, pair in enumerate(ctx.pairs):
        print(f"\n[coarse] Processing pair {pair_idx + 1}/{len(ctx.pairs)}")
        result = run_coarse_act_patching(ctx.runner, pair)
        result.sample_id = pair_idx
        ctx.coarse_patching[pair_idx] = result
        ctx.coarse_agg.add(result)

    ctx.coarse_agg.print_summary()


@profile("step_fine_activation_patching")
def step_fine_activation_patching(ctx: ExperimentContext) -> None:
    """Run targeted activation patching on decomposed targets for each component."""
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


@profile("step_visualize_results")
def step_visualize_results(ctx: ExperimentContext) -> None:
    """Visualize all patching results."""

    for pair_idx, pair in enumerate(ctx.pairs):
        pair_out_dir = ctx.output_dir / f"pair_{pair_idx}"
        coloring = get_token_coloring_for_pair(pair, ctx.runner)
        position_labels = coloring.get_position_labels("short")
        section_markers = coloring.get_section_markers("short")

        ctx.save_token_trees(pair_idx, pair, pair_out_dir)

        # Tokenization visualization (single pair as list)
        visualize_tokenization([pair], ctx.runner, pair_out_dir, max_pairs=1)

        # Per-pair patching visualizations
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

    # Aggregated visualizations
    agg_out_dir = ctx.output_dir / "agg"
    if ctx.att_agg:
        visualize_att_patching(ctx.att_agg.denoising_agg, agg_out_dir / "denoising")
        visualize_att_patching(ctx.att_agg.noising_agg, agg_out_dir / "noising")
    visualize_coarse_patching(ctx.coarse_agg, agg_out_dir)
    visualize_fine_patching(ctx.fine_agg, agg_out_dir)


@profile("run_experiment")
def run_experiment(cfg: ExperimentConfig) -> ExperimentContext:
    """Run full experiment."""
    ctx = ExperimentContext(cfg)
    step_preference_data(ctx)

    if not ctx.pairs:
        print("No preference pairs!")
        return

    # step_attribution_patching(ctx)

    step_coarse_activation_patching(ctx)

    # step_fine_activation_patching(ctx)

    step_visualize_results(ctx)

    return ctx
