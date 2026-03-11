"""Activation patching for binary choices."""

from __future__ import annotations

from .act_patch_results import (
    ActPatchPairResult,
    ActPatchTargetResult,
    IntervenedChoice,
)
from ..binary_choice import BinaryChoiceRunner
from ..common.contrastive_pair import ContrastivePair
from ..common.hook_utils import hook_filter_for_component, hook_filter_exact, hook_name
from ..common.patching_types import PatchingMode
from ..inference.interventions.intervention_target import InterventionTarget


def patch_for_choice(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    target: InterventionTarget,
    mode: PatchingMode,
    alpha: float = 1.0,
) -> IntervenedChoice:
    """Run single activation patching experiment.

    Args:
        runner: BinaryChoiceRunner for model inference
        pair: ContrastivePair (no activation required)
        target: InterventionTarget specifying layers and positions to patch
        mode: Patching mode:
            - "denoising": Run on corrupted text, patch in clean activations (REMOVE noise)
            - "noising": Run on clean text, patch in corrupted activations (ADD noise)
        alpha: Interpolation strength (1.0 = full patch, 0.0 = no patch)

    Returns:
        IntervenedChoice with original and intervened model outputs
    """
    # Build names_filter to capture internals for the component we're patching
    # Only cache the source activations (clean for denoising, corrupted for noising)
    component = target.component or "resid_post"
    names_filter = hook_filter_for_component(component)

    clean_choice = runner.choose(
        pair.clean_prompt,
        pair.choice_prefix,
        pair.clean_labels,
        with_cache=(mode == "denoising"),
        names_filter=names_filter if mode == "denoising" else None,
    )
    corrupted_choice = runner.choose(
        pair.corrupted_prompt,
        pair.choice_prefix,
        pair.corrupted_labels,
        with_cache=(mode == "noising"),
        names_filter=names_filter if mode == "noising" else None,
    )

    intervention = pair.create_patching_intervention(
        target, mode, clean_choice, corrupted_choice, alpha
    )
    if mode == "denoising":
        run_prompt = pair.corrupted_prompt
        run_labels = pair.corrupted_labels
    elif mode == "noising":
        run_prompt = pair.clean_prompt
        run_labels = pair.clean_labels
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Minimal filter for TCB: only capture final layer resid_post
    final_layer_hook = hook_name(runner.n_layers - 1, "resid_post")
    tcb_filter = hook_filter_exact(final_layer_hook)

    # Run intervention with cache to capture internals for TCB computation
    intervened_choice = runner.choose(
        run_prompt,
        pair.choice_prefix,
        run_labels,
        intervention=intervention,
        with_cache=True,
        names_filter=tcb_filter,
    )

    # Strip heavy tensors to save memory (we only need metrics)
    clean_choice.pop_heavy()
    corrupted_choice.pop_heavy()
    intervened_choice.pop_heavy()

    return IntervenedChoice(
        baseline_clean=clean_choice,
        baseline_corrupted=corrupted_choice,
        intervened=intervened_choice,
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
) -> ActPatchTargetResult:
    """Run patching for a single target in both modes.

    Args:
        runner: BinaryChoiceRunner for model inference
        pair: ContrastivePair with cached activations
        target: Default InterventionTarget (used if mode-specific target not provided)
        alpha: Interpolation strength
        denoising_target: Optional separate target for denoising mode (None = use default)
        noising_target: Optional separate target for noising mode (None = use default)
        skip_denoising: If True, skip denoising mode entirely
        skip_noising: If True, skip noising mode entirely

    Returns:
        ActPatchTargetResult with denoising and noising results
    """
    result = ActPatchTargetResult(target=target)

    if not skip_denoising:
        dn_target = denoising_target if denoising_target is not None else target
        result.denoising = patch_for_choice(runner, pair, dn_target, "denoising", alpha)

    if not skip_noising:
        ns_target = noising_target if noising_target is not None else target
        result.noising = patch_for_choice(runner, pair, ns_target, "noising", alpha)

    return result


def patch_pair(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    targets: list[InterventionTarget],
    alpha: float = 1.0,
) -> ActPatchPairResult:
    """Run patching for all targets on a pair (both denoising and noising modes).

    Args:
        runner: BinaryChoiceRunner for model inference
        pair: ContrastivePair with cached activations
        targets: List of intervention targets to patch
        alpha: Interpolation strength (1.0 = full patch, 0.0 = no patch)

    Returns:
        ActPatchPairResult with patching results for all targets and modes
    """
    result = ActPatchPairResult(sample_id=pair.sample_id)

    for target in targets:
        for mode in ("denoising", "noising"):
            choice = patch_for_choice(runner, pair, target, mode, alpha)
            result.add(target, mode, choice)

    return result
