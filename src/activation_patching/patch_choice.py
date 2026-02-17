"""Activation patching for binary choices."""

from __future__ import annotations

from typing import Literal

from ..binary_choice import BinaryChoiceRunner
from ..common.contrastive_pair import ContrastivePair
from .patching_results import IntervenedChoice, ActivationPatchingResult
from .patching_target import ActivationPatchingTarget


def patch_activation_for_choice(
    runner: BinaryChoiceRunner,
    contrastive_pair: ContrastivePair,
    target: ActivationPatchingTarget | None = None,
    mode: Literal["noising", "denoising"] = "denoising",
) -> ActivationPatchingResult:
    """Run patching experiments and measure effects on binary choice."""
    if target is None:
        target = ActivationPatchingTarget.all()

    # Get base text based on mode
    if mode == "denoising":
        base_text = contrastive_pair.short_text
        seq_len = contrastive_pair.short_length
    else:
        base_text = contrastive_pair.long_text
        seq_len = contrastive_pair.long_length

    # Resolve target to intervention targets and layers
    targets, layers, patch_together = target.to_intervention_targets(
        seq_len=seq_len,
        available_layers=contrastive_pair.available_layers,
    )

    if not layers:
        raise ValueError("No cached activations available for patching")

    # Helper to run choice with intervention
    def choose(intervention):
        return runner.choose(
            base_text,
            contrastive_pair.choice_prefix,
            contrastive_pair.labels,
            intervention=intervention,
        )

    # Get baseline
    original = choose(None)
    results: list[IntervenedChoice] = []

    if patch_together:
        # Patch all layers together for each target
        for intervention_target in targets:
            interventions = contrastive_pair.get_interventions(
                intervention_target, layers, target.component, mode
            )
            if not interventions:
                continue

            results.append(
                IntervenedChoice(
                    target=intervention_target,
                    layer=None,
                    component=target.component,
                    original=original,
                    intervened=choose(interventions),
                    mode=mode,
                )
            )
    else:
        # Patch each layer separately
        for layer in layers:
            for intervention_target in targets:
                intervention = contrastive_pair.get_intervention(
                    intervention_target, layer, target.component, mode
                )
                results.append(
                    IntervenedChoice(
                        target=intervention_target,
                        layer=layer,
                        component=target.component,
                        original=original,
                        intervened=choose(intervention),
                        mode=mode,
                    )
                )

    return ActivationPatchingResult(
        results=results,
        mode=mode,
        patched_layers=layers,
        position_mode=target.position_mode,
    )
