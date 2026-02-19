"""Activation patching for binary choices."""

from __future__ import annotations

from typing import Literal

from ..inference.interventions.intervention_target import InterventionTarget
from .act_patch_results import (
    IntervenedChoice,
    ActPatchTargetResult,
    ActPatchPairResult,
)


from ..binary_choice import BinaryChoiceRunner
from ..common.contrastive_pair import ContrastivePair


def patch_for_choice(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    target: InterventionTarget,
    mode: Literal["noising", "denoising"],
    alpha: float = 1.0,
) -> IntervenedChoice:
    """Run single patching experiment."""
    # Get base text and intervention
    if mode == "denoising":
        text = pair.short_text
    else:
        text = pair.long_text

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
    result = ActPatchTargetResult(target=target)
    result.denoising = patch_for_choice(runner, pair, target, "denoising")
    result.noising = patch_for_choice(runner, pair, target, "noising")
    return result


def patch_pair(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    targets: list[InterventionTarget],
    modes: tuple[Literal["noising", "denoising"]] = None,
    alpha: float = 1.0,
) -> ActPatchPairResult:
    """Run patching for all targets and modes on a pair."""
    if modes is None:
        modes = ("denoising", "noising")

    result = ActPatchPairResult(sample_id=pair.sample_id)

    for target in targets:
        for mode in modes:
            choice = patch_for_choice(runner, pair, target, mode, alpha)
            result.add(target, mode, choice)

    return result
