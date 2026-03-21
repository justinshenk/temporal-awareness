"""Activation patching for binary choices."""

from __future__ import annotations

from ..common.profiler import profile
from .act_patch_results import ActPatchPairResult, ActPatchTargetResult
from .intervened_choice import IntervenedChoice
from ..binary_choice import BinaryChoiceRunner
from ..common.contrastive_pair import ContrastivePair
from ..common.device_utils import clear_gpu_memory
from ..common.hook_utils import hook_filter_for_component, hook_filter_exact, hook_name
from ..common.patching_types import PatchingMode
from ..inference.interventions.intervention_target import InterventionTarget


def patch_for_choice(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    target: InterventionTarget,
    mode: PatchingMode,
    alpha: float = 1.0,
    clear_memory: bool = True,
) -> IntervenedChoice:
    """Run activation patching experiment.

    Uses choose() for same-label pairs (faster), multilabel_choose() for
    different labels. Returns IntervenedChoice with choice objects.

    Args:
        runner: BinaryChoiceRunner for model inference
        pair: ContrastivePair (no activation required)
        target: InterventionTarget specifying layers and positions to patch
        mode: Patching mode:
            - "denoising": Run on corrupted text, patch in clean activations
            - "noising": Run on clean text, patch in corrupted activations
        alpha: Interpolation strength (1.0 = full patch, 0.0 = no patch)

    Returns:
        IntervenedChoice with baseline_clean, baseline_corrupted, and intervened.
    """
    component = target.component or "resid_post"
    names_filter = hook_filter_for_component(component)

    # Use simpler choose() when labels are the same (faster path)
    same_labels = pair.clean_labels == pair.corrupted_labels

    if same_labels:
        return _patch_for_choice_single_label(
            runner, pair, target, mode, alpha, clear_memory, names_filter, component
        )
    else:
        return _patch_for_choice_multilabel(
            runner, pair, target, mode, alpha, clear_memory, names_filter, component
        )


@profile
def _patch_for_choice_single_label(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    target: InterventionTarget,
    mode: PatchingMode,
    alpha: float,
    clear_memory: bool,
    names_filter,
    component: str,
) -> IntervenedChoice:
    """Fast path for same-label pairs using choose()."""
    labels = pair.clean_labels  # same as corrupted_labels

    # Get baseline choices
    clean_choice = runner.choose(
        pair.clean_prompt,
        pair.choice_prefix,
        labels,
        with_cache=(mode == "denoising"),
        names_filter=names_filter if mode == "denoising" else None,
    )

    corrupted_choice = runner.choose(
        pair.corrupted_prompt,
        pair.choice_prefix,
        labels,
        with_cache=(mode == "noising"),
        names_filter=names_filter if mode == "noising" else None,
    )

    # Create intervention
    intervention = pair.create_patching_intervention(
        target, mode, clean_choice, corrupted_choice, alpha
    )

    if clear_memory:
        clean_choice.pop_heavy()
        corrupted_choice.pop_heavy()
        clear_gpu_memory()

    # Run with intervention
    run_prompt = pair.corrupted_prompt if mode == "denoising" else pair.clean_prompt

    final_layer_hook = hook_name(runner.n_layers - 1, "resid_post")
    tcb_filter = hook_filter_exact(final_layer_hook)

    intervened_choice = runner.choose(
        run_prompt,
        pair.choice_prefix,
        labels,
        intervention=intervention,
        with_cache=True,
        names_filter=tcb_filter,
    )

    if clear_memory:
        intervened_choice.pop_heavy()
        clear_gpu_memory()

    return IntervenedChoice(
        baseline_clean=clean_choice,
        baseline_corrupted=corrupted_choice,
        intervened=intervened_choice,
        mode=mode,
    )


@profile
def _patch_for_choice_multilabel(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    target: InterventionTarget,
    mode: PatchingMode,
    alpha: float,
    clear_memory: bool,
    names_filter,
    component: str,
) -> IntervenedChoice:
    """Multilabel path for different-label pairs using multilabel_choose()."""
    # Build label list: [clean_labels, corrupted_labels]
    all_labels = [pair.clean_labels, pair.corrupted_labels]

    # Get baseline grouped choices
    clean_grouped = runner.multilabel_choose(
        pair.clean_prompt,
        pair.choice_prefix,
        all_labels,
        with_cache=(mode == "denoising"),
        names_filter=names_filter if mode == "denoising" else None,
    )

    corrupted_grouped = runner.multilabel_choose(
        pair.corrupted_prompt,
        pair.choice_prefix,
        all_labels,
        with_cache=(mode == "noising"),
        names_filter=names_filter if mode == "noising" else None,
    )

    # For intervention, use the "native" choice for each prompt:
    # - clean_prompt with clean_labels (index 0)
    # - corrupted_prompt with corrupted_labels (index 1)
    clean_for_intervention = clean_grouped.choices[0]
    corrupted_for_intervention = corrupted_grouped.choices[1]

    # Create intervention using native label choices
    intervention = pair.create_patching_intervention(
        target, mode, clean_for_intervention, corrupted_for_intervention, alpha
    )

    if clear_memory:
        clean_grouped.pop_heavy()
        corrupted_grouped.pop_heavy()
        clear_gpu_memory()

    # Run with intervention
    run_prompt = pair.corrupted_prompt if mode == "denoising" else pair.clean_prompt

    final_layer_hook = hook_name(runner.n_layers - 1, "resid_post")
    tcb_filter = hook_filter_exact(final_layer_hook)

    intervened_grouped = runner.multilabel_choose(
        run_prompt,
        pair.choice_prefix,
        all_labels,
        intervention=intervention,
        with_cache=True,
        names_filter=tcb_filter,
    )

    if clear_memory:
        intervened_grouped.pop_heavy()
        clear_gpu_memory()

    return IntervenedChoice(
        baseline_clean=clean_grouped,
        baseline_corrupted=corrupted_grouped,
        intervened=intervened_grouped,
        mode=mode,
    )


def patch_target(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    target: InterventionTarget,
    alpha: float = 1.0,
    denoising_target: InterventionTarget | None = None,
    noising_target: InterventionTarget | None = None,
    skip_denoising: bool = False,
    skip_noising: bool = False,
    clear_memory: bool = True,
) -> ActPatchTargetResult:
    """Run patching for a single target in both modes.

    Args:
        runner: BinaryChoiceRunner for model inference
        pair: ContrastivePair with cached activations
        target: Default InterventionTarget (used if mode-specific target not provided)
        alpha: Interpolation strength
        denoising_target: Optional separate target for denoising mode
        noising_target: Optional separate target for noising mode
        skip_denoising: If True, skip denoising mode
        skip_noising: If True, skip noising mode

    Returns:
        ActPatchTargetResult with IntervenedChoice for each mode
    """
    result = ActPatchTargetResult(target=target)

    if not skip_denoising:
        dn_target = denoising_target or target
        result.denoising = patch_for_choice(runner, pair, dn_target, "denoising", alpha)

    if not skip_noising:
        ns_target = noising_target or target
        result.noising = patch_for_choice(runner, pair, ns_target, "noising", alpha)

    if clear_memory:
        result.pop_heavy()
        clear_gpu_memory()

    return result


def patch_pair(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    targets: list[InterventionTarget],
    alpha: float = 1.0,
    clear_memory: bool = True,
) -> ActPatchPairResult:
    """Run patching for all targets on a pair (both denoising and noising modes).

    Args:
        runner: BinaryChoiceRunner for model inference
        pair: ContrastivePair with cached activations
        targets: List of intervention targets to patch
        alpha: Interpolation strength (1.0 = full patch, 0.0 = no patch)

    Returns:
        ActPatchPairResult with patching results for all targets
    """
    result = ActPatchPairResult(sample_id=pair.sample_id)

    for target in targets:
        target_result = patch_target(runner, pair, target, alpha)
        result.by_target[target] = target_result

        if clear_memory:
            result.pop_heavy()
            clear_gpu_memory()

    return result
