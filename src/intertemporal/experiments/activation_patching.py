"""Activation patching experiment for identifying important layers and positions.

This module provides the high-level experiment interface for running activation
patching on intertemporal preference choices, measuring which layers and positions
are most important for the model's time-horizon-dependent decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ...common import BaseSchema, get_device
from ...inference.backends import get_recommended_backend_internals
from ...activation_patching import (
    ActivationPatchingTarget,
    ActivationPatchingResult,
    AggregatedActivationPatchingResult,
    patch_activation_for_choice,
)
from ..common.contrastive_preferences import (
    get_contrastive_preferences,
)
from ...common.contrastive_pair import ContrastivePair

from ...binary_choice import BinaryChoiceRunner
from ...binary_choice.choice_utils import verify_greedy_generation
from ..preference import PreferenceDataset


@dataclass
class DualModeActivationPatchingResult(BaseSchema):
    """Results from running both noising and denoising activation patching."""

    denoising: AggregatedActivationPatchingResult | None = None
    noising: AggregatedActivationPatchingResult | None = None

    def print_summary(self) -> None:
        """Print comparison summary of both modes."""
        print("\n" + "=" * 60)
        print("DUAL MODE ACTIVATION PATCHING COMPARISON")
        print("=" * 60)

        if self.denoising:
            print("\n[DENOISING] (corrupt → clean activations)")
            print(f"  Mean recovery: {self.denoising.mean_recovery:.3f}")
            print(f"  Flip rate: {self.denoising.flip_rate:.1%}")
            best_layer, best_recovery = self.denoising.get_best_layer()
            if best_layer is not None:
                print(f"  Best layer: L{best_layer} (recovery={best_recovery:.3f})")

        if self.noising:
            print("\n[NOISING] (clean → corrupt activations)")
            print(f"  Mean recovery: {self.noising.mean_recovery:.3f}")
            print(f"  Flip rate: {self.noising.flip_rate:.1%}")
            best_layer, best_recovery = self.noising.get_best_layer()
            if best_layer is not None:
                print(f"  Best layer: L{best_layer} (recovery={best_recovery:.3f})")

        if self.denoising and self.noising:
            print("\n[COMPARISON]")
            recovery_diff = self.denoising.mean_recovery - self.noising.mean_recovery
            if abs(recovery_diff) < 0.05:
                print(f"  Similar recovery (diff={recovery_diff:+.3f})")
            elif recovery_diff > 0:
                print(f"  Denoising more effective (+{recovery_diff:.3f})")
            else:
                print(f"  Noising more effective ({recovery_diff:.3f})")

        print("=" * 60)


@dataclass
class IntertemporalActivationPatchingConfig(BaseSchema):
    """Configuration for intertemporal activation patching experiment.

    Attributes:
        n_pairs: Number of contrastive pairs to process
        target: Patching target specification (positions, layers, component)
        mode: "denoising", "noising", or "both" (runs both and compares)
        verify_with_greedy: If True, verify flipped choices with greedy generation
        max_new_tokens: Max tokens to generate during verification
        temperature: Temperature for verification generation (0.0 = greedy)
    """

    n_pairs: int = 1
    target: ActivationPatchingTarget | dict = field(
        default_factory=ActivationPatchingTarget
    )
    mode: Literal["noising", "denoising", "both"] = "both"
    verify_with_greedy: bool = True
    max_new_tokens: int = 10
    temperature: float = 0.0

    def __post_init__(self):
        if isinstance(self.target, dict):
            self.target = ActivationPatchingTarget(**self.target)

    @property
    def modes_to_run(self) -> list[Literal["noising", "denoising"]]:
        """Get list of modes to actually run."""
        if self.mode == "both":
            return ["denoising", "noising"]
        return [self.mode]

    def print_summary(self) -> None:
        print("Activation patching config:")
        print(f"  mode={self.mode}")
        print(
            f"  position_mode={self.target.position_mode}, layers={self.target.layers}"
        )
        print(f"  component={self.target.component}, n_pairs={self.n_pairs}")
        if self.verify_with_greedy:
            print(f"  verify_with_greedy=True (max_tokens={self.max_new_tokens})")


def run_activation_patching(
    pref_dataset: PreferenceDataset,
    cfg: IntertemporalActivationPatchingConfig | None = None,
) -> AggregatedActivationPatchingResult | DualModeActivationPatchingResult:
    """Run full activation patching experiment on preference dataset.

    This function:
    1. Finds contrastive pairs (samples with different time horizons and choices)
    2. For each pair, captures activations and builds ContrastivePair
    3. Runs patching experiments across specified layers and positions
    4. Optionally verifies flipped choices with greedy generation
    5. Aggregates results across all pairs

    Args:
        pref_dataset: PreferenceDataset with samples to analyze
        cfg: Configuration for the experiment

    Returns:
        AggregatedActivationPatchingResult (single mode) or
        DualModeActivationPatchingResult (when mode="both")
    """
    if cfg is None:
        cfg = IntertemporalActivationPatchingConfig()

    all_pairs = get_contrastive_preferences(pref_dataset)
    if not all_pairs:
        print("No contrastive pairs found!")
        return None
    pairs_to_process = all_pairs[: cfg.n_pairs]
    print(f"Found {len(all_pairs)} pairs, processing {len(pairs_to_process)}")

    cfg.print_summary()
    backend = get_recommended_backend_internals()
    runner = BinaryChoiceRunner(
        pref_dataset.model, device=get_device(), backend=backend
    )

    # Get anchor texts from prompt format config for position alignment
    prompt_format = pref_dataset.prompt_format_config
    anchor_texts = prompt_format.get_anchor_texts()

    # Build contrastive pairs once (reused for both modes)
    contrastive_pairs = []
    for i, contrastive_pref in enumerate(pairs_to_process):
        contrastive_pair = contrastive_pref.get_contrastive_pair(
            runner, anchor_texts=anchor_texts
        )
        if contrastive_pair is None:
            print(f"  Skipping pair {i + 1}: invalid sample data")
            continue
        contrastive_pairs.append(contrastive_pair)

    if not contrastive_pairs:
        print("No valid contrastive pairs!")
        return None

    # Run for each mode
    modes_to_run = cfg.modes_to_run
    mode_results: dict[str, AggregatedActivationPatchingResult] = {}

    for mode in modes_to_run:
        print(f"\n{'=' * 40}")
        print(f"Running {mode.upper()} mode")
        print("=" * 40)

        pair_results: list[ActivationPatchingResult] = []
        for i, contrastive_pair in enumerate(contrastive_pairs):
            print(f"\nPair {i + 1}/{len(contrastive_pairs)}")
            contrastive_pair.print_summary()

            result = patch_activation_for_choice(
                runner=runner,
                contrastive_pair=contrastive_pair,
                target=cfg.target,
                mode=mode,
            )

            # Verify flipped choices if enabled
            if cfg.verify_with_greedy:
                _verify_flipped_choices(result, runner, contrastive_pair, cfg, mode)

            pair_results.append(result)
            result.print_summary()

        aggregated = AggregatedActivationPatchingResult.from_results(pair_results)
        mode_results[mode] = aggregated

        print(f"\n{mode.upper()} complete:")
        aggregated.print_summary()

    # Return appropriate result type
    if cfg.mode == "both":
        dual_result = DualModeActivationPatchingResult(
            denoising=mode_results.get("denoising"),
            noising=mode_results.get("noising"),
        )
        dual_result.print_summary()
        return dual_result
    else:
        return mode_results[cfg.mode]


def _verify_flipped_choices(
    result: ActivationPatchingResult,
    runner: BinaryChoiceRunner,
    contrastive_pair: ContrastivePair,
    cfg: IntertemporalActivationPatchingConfig,
    mode: Literal["noising", "denoising"] | None = None,
) -> None:
    """Verify flipped choices with greedy generation to detect degeneration.

    Mutates result.results in place, setting decoding_mismatch on each IntervenedChoice.
    """
    # Use provided mode or fall back to cfg.mode (for backwards compat)
    effective_mode = mode if mode else cfg.mode
    if effective_mode == "both":
        effective_mode = "denoising"  # Shouldn't happen, but default

    # Get base text and labels
    if effective_mode == "denoising":
        prompt_text = contrastive_pair.short_text
    else:
        prompt_text = contrastive_pair.long_text

    short_label = contrastive_pair.short_label or ""
    long_label = contrastive_pair.long_label or ""
    choice_prefix = contrastive_pair.choice_prefix

    n_flipped = sum(1 for r in result.results if r.choice_flipped)
    if n_flipped == 0:
        return

    print(f"  Verifying {n_flipped} flipped choices...")

    for ic in result.results:
        if not ic.choice_flipped:
            # Not flipped, no verification needed
            ic.decoding_mismatch = None
            continue

        # Reconstruct intervention for this result
        if ic.layer is None:
            # All layers patched together
            layers = contrastive_pair.available_layers
            intervention = contrastive_pair.get_interventions(
                ic.target, layers, ic.component, effective_mode
            )
        else:
            intervention = contrastive_pair.get_intervention(
                ic.target, ic.layer, ic.component, effective_mode
            )

        # Generate with intervention
        generated_response = runner.generate(
            prompt_text,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            intervention=intervention,
        )

        # Verify generation matches probability-based choice
        mismatch = verify_greedy_generation(
            ic.intervened,
            generated_response,
            short_label,
            long_label,
            choice_prefix,
            runner=runner,
            prompt=prompt_text,
        )
        ic.decoding_mismatch = mismatch

    n_mismatches = sum(1 for r in result.results if r.decoding_mismatch is True)
    if n_mismatches > 0:
        print(f"  WARNING: {n_mismatches}/{n_flipped} flips had decoding mismatches")


def run_layer_sweep(
    pref_dataset: "PreferenceDataset",
    n_pairs: int = 5,
    component: str = "resid_post",
    mode: Literal["noising", "denoising"] = "denoising",
) -> dict[int, float]:
    """Sweep to find most important layers by patching each layer separately.

    Args:
        pref_dataset: PreferenceDataset with samples
        n_pairs: Number of contrastive pairs to use
        component: Component to patch
        mode: Patching mode

    Returns:
        Dict mapping layer index to mean recovery
    """
    cfg = IntertemporalActivationPatchingConfig(
        n_pairs=n_pairs,
        target=ActivationPatchingTarget(
            position_mode="all",
            layers="each",
            component=component,
        ),
        mode=mode,
    )

    result = run_activation_patching(pref_dataset, cfg)
    return result.get_recovery_by_layer()


def run_position_sweep(
    pref_dataset: "PreferenceDataset",
    layer: int,
    n_pairs: int = 5,
    component: str = "resid_post",
    mode: Literal["noising", "denoising"] = "denoising",
) -> dict[int, float]:
    """Sweep to find most important positions at a specific layer.

    Args:
        pref_dataset: PreferenceDataset with samples
        layer: Layer to patch
        n_pairs: Number of contrastive pairs to use
        component: Component to patch
        mode: Patching mode

    Returns:
        Dict mapping position index to mean recovery
    """
    cfg = IntertemporalActivationPatchingConfig(
        n_pairs=n_pairs,
        target=ActivationPatchingTarget(
            position_mode="each",
            layers=[layer],
            component=component,
        ),
        mode=mode,
    )

    result = run_activation_patching(pref_dataset, cfg)

    position_recoveries: dict[int, list[float]] = {}
    for pr in result.results:
        by_pos = pr.get_results_by_position()
        for pos, results in by_pos.items():
            if pos not in position_recoveries:
                position_recoveries[pos] = []
            position_recoveries[pos].extend(r.recovery for r in results)

    return {
        pos: sum(recoveries) / len(recoveries)
        for pos, recoveries in position_recoveries.items()
    }
