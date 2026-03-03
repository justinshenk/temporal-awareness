"""Activation patching for binary choices."""

from __future__ import annotations

# Toggle between full_text and prompt_text for patching
# full_text includes the model's response in the user message (works correctly)
# prompt_text uses only the prompt without response (gives flat results)
USING_FULL_TEXT = True

from .act_patch_results import (
    ActPatchPairResult,
    ActPatchTargetResult,
    IntervenedChoice,
)
from ..binary_choice import BinaryChoiceRunner
from ..common.contrastive_pair import ContrastivePair
from ..common.types import PatchingMode
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
        pair: ContrastivePair with cached activations for both trajectories
        target: InterventionTarget specifying layers and positions to patch
        mode: Patching mode:
            - "denoising": Run on clean text, patch in corrupted activations
            - "noising": Run on corrupted text, patch in clean activations
        alpha: Interpolation strength (1.0 = full patch, 0.0 = no patch)

    Returns:
        IntervenedChoice with original and intervened model outputs
    """
    # Get base text based on mode
    # denoising: run on clean text, patch corrupted activations
    # noising: run on corrupted text, patch clean activations
    if USING_FULL_TEXT:
        text = pair.clean_text if mode == "denoising" else pair.corrupted_text
    else:
        text = pair.clean_prompt if mode == "denoising" else pair.corrupted_prompt

    layers = target.resolve_layers(pair.available_layers)
    intervention = pair.get_interventions(target, layers, target.component, mode, alpha)

    # Get baseline and intervened choices
    original = runner.choose(text, pair.choice_prefix, pair.labels, intervention=None)
    intervened = runner.choose(
        text, pair.choice_prefix, pair.labels, intervention=intervention
    )

    # Strip heavy tensors to save memory (we only need metrics)
    original.pop_heavy()
    intervened.pop_heavy()

    return IntervenedChoice(original=original, intervened=intervened, mode=mode)


def patch_target(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    target: InterventionTarget,
) -> ActPatchTargetResult:
    """Run patching for a single target in both modes.

    Args:
        runner: BinaryChoiceRunner for model inference
        pair: ContrastivePair with cached activations
        target: InterventionTarget specifying layers and positions

    Returns:
        ActPatchTargetResult with denoising and noising results
    """
    result = ActPatchTargetResult(target=target)
    result.denoising = patch_for_choice(runner, pair, target, "denoising")
    result.noising = patch_for_choice(runner, pair, target, "noising")
    return result


def patch_pair(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    targets: list[InterventionTarget],
    modes: tuple[PatchingMode, ...] | None = None,
    alpha: float = 1.0,
) -> ActPatchPairResult:
    """Run patching for all targets and modes on a pair.

    Args:
        runner: BinaryChoiceRunner for model inference
        pair: ContrastivePair with cached activations
        targets: List of intervention targets to patch
        modes: Tuple of modes to run ("denoising", "noising"). Defaults to both.
        alpha: Interpolation strength (1.0 = full patch, 0.0 = no patch)

    Returns:
        ActPatchPairResult with patching results for all targets and modes
    """
    if modes is None:
        modes = ("denoising", "noising")

    result = ActPatchPairResult(sample_id=pair.sample_id)

    for target in targets:
        for mode in modes:
            choice = patch_for_choice(runner, pair, target, mode, alpha)
            result.add(target, mode, choice)

    return result
