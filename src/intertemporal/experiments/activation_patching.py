"""Activation patching experiment for intertemporal preference analysis.

This module provides the high-level experiment interface for running activation
patching on intertemporal preference choices.

Submodules:
- sweeps.py: Layer and position sweep utilities
- verification.py: Greedy generation verification
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
from ..common.contrastive_preferences import get_contrastive_preferences
from ...binary_choice import BinaryChoiceRunner
from ..preference import PreferenceDataset

from .verification import verify_flipped_choices
from .sweeps import SweepResults, run_layer_sweep, run_progressive_position_patching, run_full_sweep


# ============================================================================
# Result types
# ============================================================================


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
            self._print_mode_summary("DENOISING", "inject long acts -> short prompt", self.denoising)

        if self.noising:
            self._print_mode_summary("NOISING", "inject short acts -> long prompt", self.noising)

        if self.denoising and self.noising:
            self._print_comparison()

        print("=" * 60)

    def _print_mode_summary(self, name: str, desc: str, result: AggregatedActivationPatchingResult) -> None:
        print(f"\n[{name}] ({desc})")
        print(f"  Mean recovery: {result.mean_recovery:.3f}")
        print(f"  Flip rate: {result.flip_rate:.1%}")

        best_layer, best_recovery = result.get_best_layer()
        if best_layer is not None:
            print(f"  Best layer: L{best_layer} (recovery={best_recovery:.3f})")

        self._print_raw_diffs(result)

    def _print_raw_diffs(self, result: AggregatedActivationPatchingResult) -> None:
        if not result.results:
            return
        print("  Raw logprob diffs:")
        for i, pr in enumerate(result.results):
            if not pr.results:
                continue
            r = pr.results[0]
            orig = r.original._divergent_logprobs
            intv = r.intervened._divergent_logprobs
            print(
                f"    Pair {i+1}: baseline=[{orig[0]:.2f}, {orig[1]:.2f}] "
                f"diff={r.original_logprob_diff:.2f} -> "
                f"patched=[{intv[0]:.2f}, {intv[1]:.2f}] "
                f"diff={r.consistent_logprob_diff:.2f}"
            )

    def _print_comparison(self) -> None:
        print("\n[COMPARISON]")
        diff = self.denoising.mean_recovery - self.noising.mean_recovery
        if abs(diff) < 0.05:
            print(f"  Recovery symmetric (diff={diff:+.3f})")
        elif diff > 0:
            print(f"  Denoising stronger (+{diff:.3f})")
        else:
            print(f"  Noising stronger ({diff:.3f})")


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class IntertemporalActivationPatchingConfig(BaseSchema):
    """Configuration for intertemporal activation patching experiment.

    Attributes:
        n_pairs: Number of contrastive pairs to process
        target: Patching target specification (positions, layers, component)
        mode: "denoising", "noising", or "both"
        verify_with_greedy: Verify flipped choices with generation
        max_new_tokens: Max tokens for verification generation
        temperature: Temperature for verification (0.0 = greedy)
        alpha: Interpolation factor (0=source, 1=full replacement)
    """

    n_pairs: int = 1
    target: ActivationPatchingTarget | dict = field(default_factory=ActivationPatchingTarget)
    mode: Literal["noising", "denoising", "both"] = "both"
    verify_with_greedy: bool = True
    max_new_tokens: int = 10
    temperature: float = 0.0
    alpha: float = 1.0

    def __post_init__(self):
        if isinstance(self.target, dict):
            self.target = ActivationPatchingTarget(**self.target)

    @property
    def modes_to_run(self) -> list[Literal["noising", "denoising"]]:
        if self.mode == "both":
            return ["denoising", "noising"]
        return [self.mode]

    def print_summary(self) -> None:
        print("Activation patching config:")
        print(f"  mode={self.mode}")
        print(f"  position_mode={self.target.position_mode}, layers={self.target.layers}")
        print(f"  component={self.target.component}, n_pairs={self.n_pairs}")
        if self.verify_with_greedy:
            print(f"  verify_with_greedy=True (max_tokens={self.max_new_tokens})")


# ============================================================================
# Main experiment runner
# ============================================================================


def run_activation_patching(
    pref_dataset: PreferenceDataset,
    cfg: IntertemporalActivationPatchingConfig | None = None,
) -> AggregatedActivationPatchingResult | DualModeActivationPatchingResult:
    """Run activation patching experiment on preference dataset.

    This function:
    1. Finds contrastive pairs (samples with different time horizons)
    2. Captures activations and builds ContrastivePairs
    3. Runs patching across specified layers and positions
    4. Optionally verifies flipped choices with generation
    5. Aggregates results

    Args:
        pref_dataset: PreferenceDataset with samples
        cfg: Experiment configuration

    Returns:
        AggregatedActivationPatchingResult (single mode) or
        DualModeActivationPatchingResult (mode="both")
    """
    if cfg is None:
        cfg = IntertemporalActivationPatchingConfig()

    # Setup
    contrastive_pairs = _build_contrastive_pairs(pref_dataset, cfg)
    if not contrastive_pairs:
        return None

    runner = _create_runner(pref_dataset)

    # Run for each mode
    mode_results = {}
    for mode in cfg.modes_to_run:
        result = _run_single_mode(runner, contrastive_pairs, cfg, mode)
        mode_results[mode] = result

    # Return appropriate result type
    if cfg.mode == "both":
        dual = DualModeActivationPatchingResult(
            denoising=mode_results.get("denoising"),
            noising=mode_results.get("noising"),
        )
        dual.print_summary()
        return dual

    return mode_results[cfg.mode]


# ============================================================================
# Internal helpers
# ============================================================================


def _build_contrastive_pairs(pref_dataset: PreferenceDataset, cfg):
    """Build contrastive pairs from preference dataset."""
    all_pairs = get_contrastive_preferences(pref_dataset)
    if not all_pairs:
        print("No contrastive pairs found!")
        return []

    pairs_to_process = all_pairs[: cfg.n_pairs]
    print(f"Found {len(all_pairs)} pairs, processing {len(pairs_to_process)}")
    cfg.print_summary()

    # Get anchor texts for position alignment
    prompt_format = pref_dataset.prompt_format_config
    anchor_texts = prompt_format.get_anchor_texts()

    # Create runner for building pairs
    runner = _create_runner(pref_dataset)

    contrastive_pairs = []
    for i, contrastive_pref in enumerate(pairs_to_process):
        pair = contrastive_pref.get_contrastive_pair(runner, anchor_texts=anchor_texts)
        if pair is None:
            print(f"  Skipping pair {i + 1}: invalid sample data")
            continue
        contrastive_pairs.append(pair)

    if not contrastive_pairs:
        print("No valid contrastive pairs!")

    return contrastive_pairs


def _create_runner(pref_dataset: PreferenceDataset) -> BinaryChoiceRunner:
    """Create BinaryChoiceRunner for the dataset's model."""
    backend = get_recommended_backend_internals()
    return BinaryChoiceRunner(
        pref_dataset.model, device=get_device(), backend=backend
    )


def _run_single_mode(runner, contrastive_pairs, cfg, mode) -> AggregatedActivationPatchingResult:
    """Run patching for a single mode (denoising or noising)."""
    print(f"\n{'=' * 40}")
    print(f"Running {mode.upper()} mode")
    print("=" * 40)

    pair_results = []
    for i, pair in enumerate(contrastive_pairs):
        print(f"\nPair {i + 1}/{len(contrastive_pairs)}")
        pair.print_summary()

        result = patch_activation_for_choice(
            runner=runner,
            contrastive_pair=pair,
            target=cfg.target,
            mode=mode,
            alpha=cfg.alpha,
        )

        if cfg.verify_with_greedy:
            verify_flipped_choices(result, runner, pair, cfg, mode)

        pair_results.append(result)
        result.print_summary()

    aggregated = AggregatedActivationPatchingResult.from_results(pair_results)

    print(f"\n{mode.upper()} complete:")
    aggregated.print_summary()

    return aggregated


# ============================================================================
# Re-exports for backwards compatibility
# ============================================================================

# Export sweep functions from sweeps module
__all__ = [
    "run_activation_patching",
    "IntertemporalActivationPatchingConfig",
    "DualModeActivationPatchingResult",
    "SweepResults",
    "run_layer_sweep",
    "run_progressive_position_patching",
    "run_full_sweep",
]
