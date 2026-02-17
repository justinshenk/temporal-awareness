"""Results dataclasses for activation patching experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..common.base_schema import BaseSchema
from ..common.choice import LabeledSimpleBinaryChoice
from ..inference.interventions import InterventionTarget


@dataclass
class IntervenedChoice(BaseSchema):
    """Result of a single patching intervention on a binary choice.

    Attributes:
        target: The intervention target (positions/layers patched)
        layer: The layer that was patched (None = all layers patched together)
        component: The component that was patched (resid_post, etc.)
        original: Choice made without intervention
        intervened: Choice made with intervention
        mode: Whether this was noising or denoising
        decoding_mismatch: Whether greedy generation mismatches choice probabilities.
            None if not verified, True if mismatch detected, False if verified OK.
    """

    target: InterventionTarget
    layer: int | None  # None = all layers patched together
    component: str
    original: LabeledSimpleBinaryChoice
    intervened: LabeledSimpleBinaryChoice
    mode: Literal["noising", "denoising"]
    decoding_mismatch: bool | None = None

    @property
    def original_logprob_diff(self) -> float:
        """Logprob difference (chosen - alternative) without intervention."""
        return self.original.choice_logprob - self.original.alternative_logprob

    @property
    def intervened_logprob_diff(self) -> float:
        """Logprob difference (chosen - alternative) with intervention."""
        return self.intervened.choice_logprob - self.intervened.alternative_logprob

    @property
    def consistent_logprob_diff(self) -> float:
        """Logprob diff using original's sign convention (idx0 - idx1).

        This ensures consistent comparison even when choice flips.
        Positive = favors original choice, negative = favors alternative.
        """
        # Always compute as: logprob[original_choice] - logprob[original_alternative]
        # This uses the original's choice_idx as the reference
        orig_idx = self.original.choice_idx
        if orig_idx == 0:
            # Original chose idx 0, so diff = lp[0] - lp[1]
            return self.intervened._divergent_logprobs[0] - self.intervened._divergent_logprobs[1]
        else:
            # Original chose idx 1, so diff = lp[1] - lp[0]
            return self.intervened._divergent_logprobs[1] - self.intervened._divergent_logprobs[0]

    @property
    def recovery(self) -> float:
        """Compute normalized recovery from logprob diffs.

        For denoising: measures progress toward flipping the choice.
        - 0 = no change from original
        - 1 = fully recovered (logprob diff matches target behavior)
        - >0 = moved in right direction

        Uses consistent sign convention to handle choice flips correctly.
        """
        orig_diff = self.original_logprob_diff  # Always positive (chose what it chose)
        intv_diff = self.consistent_logprob_diff  # Uses original's sign convention

        if orig_diff == 0:
            return 0.0

        if self.mode == "denoising":
            # For denoising, we want to flip: orig_diff positive -> negative
            # Recovery = how much we moved toward negative / how far we need to go
            # If orig_diff = 2 and intv_diff = -2, recovery = 1.0 (fully flipped)
            # If orig_diff = 2 and intv_diff = 0, recovery = 0.5 (halfway)
            # If orig_diff = 2 and intv_diff = 2, recovery = 0.0 (no change)
            change = orig_diff - intv_diff  # Positive when moving toward flip
            recovery = change / (2 * abs(orig_diff))  # Normalize to [0, 1] for full flip
            return max(0.0, min(1.0, recovery))
        else:  # noising
            # For noising, we want to damage: should reduce confidence
            change = orig_diff - intv_diff
            return change / abs(orig_diff)

    @property
    def choice_flipped(self) -> bool:
        """Whether the intervention flipped the model's choice."""
        return self.original.choice_idx != self.intervened.choice_idx

    @property
    def original_chosen_label(self) -> str | None:
        """Label chosen without intervention."""
        return self.original.chosen_label

    @property
    def intervened_chosen_label(self) -> str | None:
        """Label chosen with intervention."""
        return self.intervened.chosen_label

    def _to_dict_hook(self, d: dict) -> dict:
        """Include computed properties in serialization."""
        d["original_logprob_diff"] = self.original_logprob_diff
        d["intervened_logprob_diff"] = self.intervened_logprob_diff
        d["recovery"] = self.recovery
        return d

    def print_summary(self, verbose: bool = False) -> None:
        layer_str = "all" if self.layer is None else f"L{self.layer:2d}"
        pos_str = (
            f"pos {self.target.positions}"
            if self.target.positions
            else self.target.axis
        )

        # Choice transition
        orig_label = self.original_chosen_label or f"idx={self.original.choice_idx}"
        intv_label = self.intervened_chosen_label or f"idx={self.intervened.choice_idx}"

        flip_marker = " [FLIP]" if self.choice_flipped else ""
        if self.decoding_mismatch is True:
            flip_marker += " [MISMATCH]"
        elif self.decoding_mismatch is False:
            flip_marker += " [verified]"

        print(
            f"  {layer_str} @ {pos_str}: {orig_label} -> {intv_label}{flip_marker}"
        )
        print(
            f"    recovery={self.recovery:.4f}, "
            f"logprob_diff: {self.original_logprob_diff:.3f} -> {self.intervened_logprob_diff:.3f}"
        )

        if verbose:
            # Show raw logprobs
            orig_lps = self.original._divergent_logprobs
            intv_lps = self.intervened._divergent_logprobs
            print(f"    original logprobs:   [{orig_lps[0]:.4f}, {orig_lps[1]:.4f}]")
            print(f"    intervened logprobs: [{intv_lps[0]:.4f}, {intv_lps[1]:.4f}]")
            print(f"    consistent_logprob_diff: {self.consistent_logprob_diff:.4f}")


@dataclass
class ActivationPatchingResult(BaseSchema):
    """Results from an activation patching experiment."""

    results: list[IntervenedChoice] = field(default_factory=list)
    metric: Any | None = None
    mode: Literal["noising", "denoising"] = "denoising"
    # Store the target configuration for visualization/reporting
    patched_layers: list[int] | None = None  # Which layers were patched
    position_mode: str | None = None  # "all", "explicit", etc.

    def get_results_by_layer(self) -> dict[int | None, list[IntervenedChoice]]:
        """Group results by layer."""
        by_layer: dict[int, list[IntervenedChoice]] = {}
        for r in self.results:
            if r.layer not in by_layer:
                by_layer[r.layer] = []
            by_layer[r.layer].append(r)
        return by_layer

    def get_results_by_position(self) -> dict[int, list[IntervenedChoice]]:
        """Group results by position (first position if multiple)."""
        by_position: dict[int, list[IntervenedChoice]] = {}
        for r in self.results:
            pos = r.target.positions[0] if r.target.positions else -1
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(r)
        return by_position

    def get_max_recovery(self) -> tuple[IntervenedChoice | None, float]:
        """Get the result with maximum recovery."""
        if not self.results:
            return None, 0.0
        best = max(self.results, key=lambda r: r.recovery)
        return best, best.recovery

    def get_flip_rate(self) -> float:
        """Get the fraction of interventions that flipped the choice."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.choice_flipped) / len(self.results)

    @property
    def n_results(self) -> int:
        return len(self.results)

    @property
    def mean_recovery(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.recovery for r in self.results) / len(self.results)

    @property
    def layers(self) -> list[int | None]:
        return sorted(set(r.layer for r in self.results), key=lambda x: (x is None, x))

    @property
    def n_flipped(self) -> int:
        return sum(1 for r in self.results if r.choice_flipped)

    @property
    def n_degenerate(self) -> int:
        return sum(1 for r in self.results if r.decoding_mismatch is True)

    @property
    def n_valid_flips(self) -> int:
        return sum(
            1 for r in self.results
            if r.choice_flipped and r.decoding_mismatch is False
        )

    def print_summary(self) -> None:
        if not self.results:
            print("No results")
            return

        recoveries = [r.recovery for r in self.results]
        min_rec = min(recoveries)
        max_rec = max(recoveries)

        print(f"  Interventions: {self.n_results}")
        print(f"  Recovery: mean={self.mean_recovery:.4f}, min={min_rec:.4f}, max={max_rec:.4f}")
        print(f"  Flipped: {self.n_flipped}/{self.n_results} ({self.get_flip_rate():.1%})")

        # Show flip validation stats if any flips occurred
        if self.n_flipped > 0:
            n_verified = sum(1 for r in self.results if r.choice_flipped and r.decoding_mismatch is not None)
            if n_verified > 0:
                print(f"    Valid: {self.n_valid_flips}/{n_verified}, Degenerate: {self.n_degenerate}/{n_verified}")

        # Layer breakdown if multiple layers
        by_layer = self.get_results_by_layer()
        if len(by_layer) > 1:
            print("  By layer:")
            for layer, layer_results in sorted(by_layer.items(), key=lambda x: (x[0] is None, x[0])):
                layer_mean = sum(r.recovery for r in layer_results) / len(layer_results)
                layer_flips = sum(1 for r in layer_results if r.choice_flipped)
                layer_label = "all" if layer is None else f"L{layer}"
                print(f"    {layer_label}: recovery={layer_mean:.4f}, flips={layer_flips}")

    def print_detailed(self) -> None:
        """Print detailed per-intervention breakdown."""
        if not self.results:
            print("No results")
            return

        print(f"\n{'='*70}")
        print("DETAILED INTERVENTION ANALYSIS")
        print(f"{'='*70}")

        # Show baseline (original choice)
        first = self.results[0]
        print(f"\nBaseline (no intervention):")
        print(f"  Choice: {first.original_chosen_label} (idx={first.original.choice_idx})")
        orig_lps = first.original._divergent_logprobs
        print(f"  Logprobs: [{orig_lps[0]:.4f}, {orig_lps[1]:.4f}]")
        print(f"  Logprob diff (chosen - alt): {first.original_logprob_diff:.4f}")

        print(f"\n{'-'*70}")
        print("Per-layer interventions:")
        print(f"{'-'*70}")

        for r in self.results:
            r.print_summary(verbose=True)
            print()


@dataclass
class AggregatedActivationPatchingResult(BaseSchema):
    """Aggregated results from multiple activation patching runs.

    Attributes:
        results: List of ActivationPatchingResult, one per run/pair
        aggregate_stats: Summary statistics for recovery across all runs
        n_runs: Number of runs processed
    """

    results: list[ActivationPatchingResult] = field(default_factory=list)
    aggregate_stats: dict[str, float] = field(default_factory=dict)
    n_runs: int = 0

    @classmethod
    def from_results(
        cls, results: list[ActivationPatchingResult]
    ) -> "AggregatedActivationPatchingResult":
        """Create aggregated result with computed statistics."""
        import numpy as np

        all_recoveries = [r.recovery for pr in results for r in pr.results]

        if all_recoveries:
            recoveries_arr = np.array(all_recoveries)
            stats = {
                "mean": float(np.mean(recoveries_arr)),
                "std": float(np.std(recoveries_arr)),
                "median": float(np.median(recoveries_arr)),
                "min": float(np.min(recoveries_arr)),
                "max": float(np.max(recoveries_arr)),
                "n_total": len(all_recoveries),
                "n_positive": int(np.sum(recoveries_arr > 0)),
                "n_flipped": sum(
                    r.choice_flipped for pr in results for r in pr.results
                ),
            }
        else:
            stats = {}

        return cls(results=results, aggregate_stats=stats, n_runs=len(results))

    @property
    def mean_recovery(self) -> float:
        """Mean recovery across all runs."""
        if not self.results:
            return 0.0
        return sum(r.mean_recovery for r in self.results) / len(self.results)

    @property
    def max_recovery(self) -> float:
        """Maximum mean recovery across all runs."""
        if not self.results:
            return 0.0
        return max(r.mean_recovery for r in self.results)

    @property
    def flip_rate(self) -> float:
        """Overall flip rate across all runs."""
        if not self.results:
            return 0.0
        total_flips = sum(r.get_flip_rate() * r.n_results for r in self.results)
        total_results = sum(r.n_results for r in self.results)
        return total_flips / total_results if total_results > 0 else 0.0

    def get_recovery_by_layer(self) -> dict[int | None, float]:
        """Get mean recovery for each layer across all runs."""
        layer_recoveries: dict[int | None, list[float]] = {}

        for pr in self.results:
            for r in pr.results:
                if r.layer not in layer_recoveries:
                    layer_recoveries[r.layer] = []
                layer_recoveries[r.layer].append(r.recovery)

        return {
            layer: sum(recoveries) / len(recoveries)
            for layer, recoveries in layer_recoveries.items()
        }

    def get_best_layer(self) -> tuple[int | None, float]:
        """Get the layer with highest mean recovery."""
        by_layer = self.get_recovery_by_layer()
        if not by_layer:
            return None, 0.0
        best_layer = max(by_layer, key=by_layer.get)
        return best_layer, by_layer[best_layer]

    @property
    def patched_layers(self) -> list[int] | None:
        """Get the layers that were patched (from first result)."""
        if self.results and self.results[0].patched_layers:
            return self.results[0].patched_layers
        return None

    @property
    def position_mode(self) -> str | None:
        """Get the position mode used (from first result)."""
        if self.results and self.results[0].position_mode:
            return self.results[0].position_mode
        return None

    def print_summary(self) -> None:
        print(f"Runs: {self.n_runs}")
        print(f"Mean recovery: {self.mean_recovery:.4f}")
        print(f"Flip rate: {self.flip_rate:.2%}")
        if self.patched_layers:
            print(f"Layers patched: {self.patched_layers}")
        if self.position_mode:
            print(f"Position mode: {self.position_mode}")
        best_layer, best_recovery = self.get_best_layer()
        if best_layer is not None:
            print(f"Best layer: {best_layer} (recovery: {best_recovery:.4f})")
