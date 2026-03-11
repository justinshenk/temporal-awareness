"""Result types for activation patching."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..common.base_schema import BaseSchema
from ..common.choice import LabeledSimpleBinaryChoice
from ..common.math.faithfulness_scores import compute_disruption, compute_recovery
from ..common.patching_types import PatchingMode
from ..inference.interventions.intervention_target import InterventionTarget

# Imported at module level for type resolution in BaseSchema.from_dict()
from .act_patch_metrics import IntervenedChoiceMetrics


@dataclass
class IntervenedChoice(BaseSchema):
    """Single intervention result.

    Stores all three values needed for proper recovery/disruption calculation:
    - baseline_clean: Result from running clean input (no intervention)
    - baseline_corrupted: Result from running corrupted input (no intervention)
    - intervened: Result after applying intervention

    Mode semantics:
    - denoising: Run corrupted text, inject clean activations -> recover clean behavior
    - noising: Run clean text, inject corrupted activations -> disrupt clean behavior

    When loaded from lightweight cache, full choice objects are None and metrics
    are stored directly in _cached_* fields.
    """

    baseline_clean: LabeledSimpleBinaryChoice | None = None
    baseline_corrupted: LabeledSimpleBinaryChoice | None = None
    intervened: LabeledSimpleBinaryChoice | None = None
    mode: PatchingMode = "denoising"
    switched: bool = False

    # Cached metrics for lightweight loading (when full objects are None)
    _cached_recovery: float | None = field(default=None, repr=False)
    _cached_disruption: float | None = field(default=None, repr=False)
    _cached_flipped: bool | None = field(default=None, repr=False)
    _cached_choice_idxs: tuple[int, int, int] | None = field(default=None, repr=False)
    _cached_logprobs: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = field(default=None, repr=False)

    # Tell _canon to call to_dict() directly instead of processing fields
    _use_custom_to_dict: bool = True

    def to_dict(self, **kwargs) -> dict:
        """Convert to lightweight dict with only metrics, not full trees.

        Overrides BaseSchema.to_dict() to skip heavy tree processing.
        """
        # Use cached values if we loaded from lightweight format
        if self.baseline_clean is None:
            return {
                "mode": self.mode.value if hasattr(self.mode, "value") else self.mode,
                "switched": self.switched,
                "recovery": self._cached_recovery,
                "disruption": self._cached_disruption,
                "flipped": self._cached_flipped,
                "baseline_clean_choice_idx": self._cached_choice_idxs[0] if self._cached_choice_idxs else None,
                "baseline_corrupted_choice_idx": self._cached_choice_idxs[1] if self._cached_choice_idxs else None,
                "intervened_choice_idx": self._cached_choice_idxs[2] if self._cached_choice_idxs else None,
                "baseline_clean_logprobs": list(self._cached_logprobs[0]) if self._cached_logprobs else None,
                "baseline_corrupted_logprobs": list(self._cached_logprobs[1]) if self._cached_logprobs else None,
                "intervened_logprobs": list(self._cached_logprobs[2]) if self._cached_logprobs else None,
            }
        return {
            "mode": self.mode.value if hasattr(self.mode, "value") else self.mode,
            "switched": self.switched,
            "recovery": self.recovery,
            "disruption": self.disruption,
            "flipped": self.flipped,
            "baseline_clean_choice_idx": self.baseline_clean.choice_idx,
            "baseline_corrupted_choice_idx": self.baseline_corrupted.choice_idx,
            "intervened_choice_idx": self.intervened.choice_idx,
            "baseline_clean_logprobs": list(self.baseline_clean.divergent_logprobs),
            "baseline_corrupted_logprobs": list(self.baseline_corrupted.divergent_logprobs),
            "intervened_logprobs": list(self.intervened.divergent_logprobs),
        }

    @classmethod
    def from_dict(cls, d: dict) -> IntervenedChoice:
        """Load from dict, handling both full and lightweight formats."""
        # Lightweight format has recovery/disruption directly
        if "recovery" in d and "baseline_clean" not in d:
            return cls(
                baseline_clean=None,
                baseline_corrupted=None,
                intervened=None,
                mode=d.get("mode", "denoising"),
                switched=d.get("switched", False),
                _cached_recovery=d.get("recovery"),
                _cached_disruption=d.get("disruption"),
                _cached_flipped=d.get("flipped"),
                _cached_choice_idxs=(
                    d.get("baseline_clean_choice_idx"),
                    d.get("baseline_corrupted_choice_idx"),
                    d.get("intervened_choice_idx"),
                ),
                _cached_logprobs=(
                    tuple(d.get("baseline_clean_logprobs", (0.0, 0.0))),
                    tuple(d.get("baseline_corrupted_logprobs", (0.0, 0.0))),
                    tuple(d.get("intervened_logprobs", (0.0, 0.0))),
                ),
            )
        # Full format - use default BaseSchema parsing
        return super().from_dict(d)

    def pop_heavy(self) -> None:
        """Remove heavy data (trees) from choices to reduce memory."""
        if self.baseline_clean:
            self.baseline_clean.pop_heavy()
        if self.baseline_corrupted:
            self.baseline_corrupted.pop_heavy()
        if self.intervened:
            self.intervened.pop_heavy()

    def _get_logit_diff(self, choice: LabeledSimpleBinaryChoice) -> float:
        """Get logit difference (A - B) from a choice, respecting switched flag."""
        logits = choice.divergent_logits
        if logits:
            if self.switched:
                return logits[1] - logits[0]
            return logits[0] - logits[1]
        # Fall back to logprobs
        lps = choice.divergent_logprobs
        if self.switched:
            return lps[1] - lps[0]
        return lps[0] - lps[1]

    @property
    def recovery(self) -> float:
        """Recovery: (y_intervened - y_corrupted) / (y_clean - y_corrupted).

        Measures how much intervention moved output from corrupted toward clean.
        - 0.0 = no change (still at corrupted baseline)
        - 1.0 = full recovery (reached clean baseline)
        """
        if self._cached_recovery is not None:
            return self._cached_recovery

        y_clean = self._get_logit_diff(self.baseline_clean)
        y_corrupted = self._get_logit_diff(self.baseline_corrupted)
        y_intervened = self._get_logit_diff(self.intervened)
        return compute_recovery(y_intervened, y_clean, y_corrupted)

    @property
    def disruption(self) -> float:
        """Disruption: (y_clean - y_intervened) / (y_clean - y_corrupted).

        Measures how much intervention moved output from clean toward corrupted.
        - 0.0 = no change (still at clean baseline)
        - 1.0 = full disruption (reached corrupted baseline)
        """
        if self._cached_disruption is not None:
            return self._cached_disruption

        y_clean = self._get_logit_diff(self.baseline_clean)
        y_corrupted = self._get_logit_diff(self.baseline_corrupted)
        y_intervened = self._get_logit_diff(self.intervened)
        return compute_disruption(y_intervened, y_clean, y_corrupted)

    @property
    def effect(self) -> float:
        """Effect metric: recovery for denoising, disruption for noising."""
        return self.recovery if self.mode == "denoising" else self.disruption

    @property
    def flipped(self) -> bool:
        """Whether the intervention flipped the choice."""
        if self._cached_flipped is not None:
            return self._cached_flipped
        if self.mode == "denoising":
            return self.baseline_corrupted.choice_idx != self.intervened.choice_idx
        return self.baseline_clean.choice_idx != self.intervened.choice_idx

    def switch(self) -> IntervenedChoice:
        """Swap clean↔corrupted semantics (e.g., short↔long term perspective).

        Swaps baselines and toggles the switched flag so that logit difference
        direction is reversed.
        """
        new_mode: PatchingMode = "noising" if self.mode == "denoising" else "denoising"
        # Handle lightweight (loaded from cache) format
        if self.baseline_clean is None:
            # Swap recovery/disruption and swap choice_idxs[0] with [1]
            new_choice_idxs = None
            if self._cached_choice_idxs:
                new_choice_idxs = (
                    self._cached_choice_idxs[1],  # old corrupted -> new clean
                    self._cached_choice_idxs[0],  # old clean -> new corrupted
                    self._cached_choice_idxs[2],  # intervened stays same
                )
            new_logprobs = None
            if self._cached_logprobs:
                new_logprobs = (
                    self._cached_logprobs[1],  # old corrupted -> new clean
                    self._cached_logprobs[0],  # old clean -> new corrupted
                    self._cached_logprobs[2],  # intervened stays same
                )
            return IntervenedChoice(
                baseline_clean=None,
                baseline_corrupted=None,
                intervened=None,
                mode=new_mode,
                switched=not self.switched,
                _cached_recovery=self._cached_disruption,  # swap
                _cached_disruption=self._cached_recovery,  # swap
                _cached_flipped=self._cached_flipped,
                _cached_choice_idxs=new_choice_idxs,
                _cached_logprobs=new_logprobs,
            )
        return IntervenedChoice(
            baseline_clean=self.baseline_corrupted,
            baseline_corrupted=self.baseline_clean,
            intervened=self.intervened,
            mode=new_mode,
            switched=not self.switched,
        )


@dataclass
class ActPatchTargetResult(BaseSchema):
    """Results for one target (both modes).

    When pop_heavy() is called, metrics are computed and stored in
    denoising_metrics/noising_metrics before the heavy tree data is removed.
    Visualization should use get_metrics() which returns cached metrics if available.
    """

    target: InterventionTarget
    denoising: IntervenedChoice | None = None
    noising: IntervenedChoice | None = None

    # Cached metrics (populated by pop_heavy before removing tree data)
    denoising_metrics: IntervenedChoiceMetrics | None = None
    noising_metrics: IntervenedChoiceMetrics | None = None

    @property
    def recovery(self) -> float | None:
        """Recovery from denoising mode (how well we recovered clean behavior)."""
        return self.denoising.recovery if self.denoising else None

    @property
    def disruption(self) -> float | None:
        """Disruption from noising mode (how well we disrupted clean behavior)."""
        return self.noising.disruption if self.noising else None

    @property
    def mean_effect(self) -> float:
        """Mean effect across both modes."""
        effects = [r.effect for r in [self.denoising, self.noising] if r]
        return sum(effects) / len(effects) if effects else 0.0

    @property
    def flip_count(self) -> int:
        return sum(1 for r in [self.denoising, self.noising] if r and r.flipped)

    def score(self) -> float:
        """Score for sorting/ranking (higher = more important)."""
        return self.mean_effect

    def pop_heavy(self) -> None:
        """Remove heavy data from choices to reduce memory.

        Computes and caches metrics BEFORE removing tree data, so
        visualization can still access all metrics from cache.
        """
        # Compute metrics before removing tree data
        if self.denoising and self.denoising_metrics is None:
            self.denoising_metrics = IntervenedChoiceMetrics.from_choice(self.denoising)
        if self.noising and self.noising_metrics is None:
            self.noising_metrics = IntervenedChoiceMetrics.from_choice(self.noising)

        # Now remove heavy data
        if self.denoising:
            self.denoising.pop_heavy()
        if self.noising:
            self.noising.pop_heavy()

    def get_denoising_metrics(self) -> IntervenedChoiceMetrics:
        """Get denoising metrics (cached or computed)."""
        if self.denoising_metrics is not None:
            return self.denoising_metrics
        return IntervenedChoiceMetrics.from_choice(self.denoising)

    def get_noising_metrics(self) -> IntervenedChoiceMetrics:
        """Get noising metrics (cached or computed)."""
        if self.noising_metrics is not None:
            return self.noising_metrics
        return IntervenedChoiceMetrics.from_choice(self.noising)

    def switch(self) -> ActPatchTargetResult:
        """Swap clean↔corrupted semantics (e.g., short↔long term perspective).

        After switch:
        - What was denoising (recovering old-clean from old-corrupted) becomes
          noising (disrupting new-clean with new-corrupted)
        - What was noising becomes denoising
        """
        return ActPatchTargetResult(
            target=self.target,
            denoising=self.noising.switch() if self.noising else None,
            noising=self.denoising.switch() if self.denoising else None,
            # Also swap cached metrics
            denoising_metrics=self.noising_metrics,
            noising_metrics=self.denoising_metrics,
        )

    def format_summary(self) -> str:
        """Format result for logging."""
        dn_flip = self.denoising.flipped if self.denoising else False
        ns_flip = self.noising.flipped if self.noising else False
        return f"recovery={self.recovery or 0:.3f} (flip={dn_flip}), disruption={self.disruption or 0:.3f} (flip={ns_flip})"


@dataclass
class ActPatchPairResult(BaseSchema):
    """Results for one contrastive pair."""

    sample_id: int
    by_target: dict[InterventionTarget, ActPatchTargetResult] = field(
        default_factory=dict
    )

    def add(
        self, target: InterventionTarget, mode: str, result: IntervenedChoice
    ) -> None:
        if target not in self.by_target:
            self.by_target[target] = ActPatchTargetResult(target=target)
        if mode == "denoising":
            self.by_target[target].denoising = result
        else:
            self.by_target[target].noising = result

    @property
    def mean_recovery(self) -> float:
        if not self.by_target:
            return 0.0
        recoveries = [r.recovery for r in self.by_target.values() if r.recovery is not None]
        return sum(recoveries) / len(recoveries) if recoveries else 0.0

    @property
    def mean_disruption(self) -> float:
        if not self.by_target:
            return 0.0
        disruptions = [r.disruption for r in self.by_target.values() if r.disruption is not None]
        return sum(disruptions) / len(disruptions) if disruptions else 0.0


@dataclass
class ActPatchAggregatedResult(BaseSchema):
    """Aggregated results across pairs."""

    by_sample: dict[int, ActPatchPairResult] = field(default_factory=dict)

    def add(self, pair: ActPatchPairResult) -> None:
        self.by_sample[pair.sample_id] = pair

    @property
    def n_samples(self) -> int:
        return len(self.by_sample)

    @property
    def mean_recovery(self) -> float:
        if not self.by_sample:
            return 0.0
        return sum(p.mean_recovery for p in self.by_sample.values()) / len(
            self.by_sample
        )

    @property
    def mean_disruption(self) -> float:
        if not self.by_sample:
            return 0.0
        return sum(p.mean_disruption for p in self.by_sample.values()) / len(
            self.by_sample
        )

    def get_recovery_by_layer(self) -> dict[int | None, float]:
        """Mean recovery per layer."""
        by_layer: dict[int | None, list[float]] = {}
        for pair in self.by_sample.values():
            for target, result in pair.by_target.items():
                if result.recovery is None:
                    continue
                layer = (
                    target.layers[0]
                    if target.layers and len(target.layers) == 1
                    else None
                )
                by_layer.setdefault(layer, []).append(result.recovery)
        return {l: sum(r) / len(r) for l, r in by_layer.items()}

    def get_best_layer(self) -> tuple[int | None, float]:
        by_layer = self.get_recovery_by_layer()
        if not by_layer:
            return None, 0.0
        best = max(by_layer.items(), key=lambda x: x[1])
        return best

    def print_summary(self) -> None:
        print(f"Samples: {self.n_samples}, Recovery: {self.mean_recovery:.4f}")
        best_l, best_r = self.get_best_layer()
        if best_l is not None:
            print(f"Best layer: L{best_l} ({best_r:.4f})")
