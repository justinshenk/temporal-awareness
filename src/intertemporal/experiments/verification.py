"""Verification utilities for activation patching results.

Provides greedy generation verification to detect whether intervention-induced
choice flips correspond to valid text generation or degenerate outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ...binary_choice import BinaryChoiceRunner
    from ...common.contrastive_pair import ContrastivePair
    from ...activation_patching import ActivationPatchingResult
    from .activation_patching import IntertemporalActivationPatchingConfig


def verify_flipped_choices(
    result: "ActivationPatchingResult",
    runner: "BinaryChoiceRunner",
    contrastive_pair: "ContrastivePair",
    cfg: "IntertemporalActivationPatchingConfig",
    mode: Literal["noising", "denoising"],
) -> None:
    """Verify flipped choices with greedy generation to detect degeneration.

    Uses actual autoregressive generation (not single forward pass) to verify
    that the intervened choice would be produced in a real generation setting.

    Mutates result.results in place, setting:
    - decoding_mismatch: True if generation doesn't match expected choice
    - generation_degenerate: True if neither label appears in output

    Args:
        result: ActivationPatchingResult to verify
        runner: BinaryChoiceRunner for generation
        contrastive_pair: ContrastivePair with text and labels
        cfg: Config with generation parameters
        mode: "denoising" or "noising"
    """
    n_flipped = sum(1 for r in result.results if r.choice_flipped)
    if n_flipped == 0:
        return

    print(f"  Verifying {n_flipped} flipped choices with generation...")

    # Get base text and labels for verification
    prompt_text = _get_prompt_text(contrastive_pair, mode)
    labels = _get_labels(contrastive_pair)
    choice_prefix = contrastive_pair.choice_prefix

    for ic in result.results:
        if not ic.choice_flipped:
            ic.decoding_mismatch = None
            continue

        # Reconstruct intervention
        intervention = _reconstruct_intervention(ic, contrastive_pair, mode)

        # Generate with intervention
        generated_text = runner.generate(
            prompt_text,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            intervention=intervention,
            prefilling=choice_prefix,
        )

        # Determine if generation matches expected choice
        _classify_generation(ic, generated_text, labels)

    _print_verification_summary(result, n_flipped)


# ============================================================================
# Helper functions
# ============================================================================


def _get_prompt_text(
    contrastive_pair: "ContrastivePair",
    mode: Literal["noising", "denoising"],
) -> str:
    """Get the base prompt text for the given mode."""
    if mode == "denoising":
        return contrastive_pair.short_text
    return contrastive_pair.long_text


def _get_labels(contrastive_pair: "ContrastivePair") -> tuple[str, str]:
    """Get (short_label, long_label) tuple."""
    return (
        contrastive_pair.short_label or "",
        contrastive_pair.long_label or "",
    )


def _reconstruct_intervention(ic, contrastive_pair: "ContrastivePair", mode: str):
    """Reconstruct the intervention used for the choice."""
    if ic.layer is None:
        layers = contrastive_pair.available_layers
        return contrastive_pair.get_interventions(
            ic.target, layers, ic.component, mode
        )
    return contrastive_pair.get_intervention(
        ic.target, ic.layer, ic.component, mode
    )


def _classify_generation(ic, generated_text: str, labels: tuple[str, str]) -> None:
    """Classify generated text and set mismatch flags.

    Sets ic.decoding_mismatch and ic.generation_degenerate based on
    whether the expected label appears in the generated text.
    """
    generated_lower = generated_text.lower().strip()
    expected_idx = ic.intervened.choice_idx
    other_idx = 1 - expected_idx

    expected_label = labels[expected_idx]
    other_label = labels[other_idx]

    expected_in_gen = expected_label.lower() in generated_lower
    other_in_gen = other_label.lower() in generated_lower

    if expected_in_gen and not other_in_gen:
        ic.decoding_mismatch = False  # Matches expected
    elif other_in_gen and not expected_in_gen:
        ic.decoding_mismatch = True  # Wrong choice
    elif not expected_in_gen and not other_in_gen:
        ic.decoding_mismatch = True
        ic.generation_degenerate = True  # Neither label found
    else:
        ic.decoding_mismatch = True  # Both labels found (ambiguous)


def _print_verification_summary(result: "ActivationPatchingResult", n_flipped: int) -> None:
    """Print summary of verification results."""
    n_mismatches = sum(1 for r in result.results if r.decoding_mismatch is True)
    n_degenerate = sum(
        1 for r in result.results if getattr(r, "generation_degenerate", False)
    )

    if n_mismatches > 0:
        msg = f"  WARNING: {n_mismatches}/{n_flipped} flips had decoding mismatches"
        if n_degenerate > 0:
            msg += f" ({n_degenerate} degenerate)"
        print(msg)
