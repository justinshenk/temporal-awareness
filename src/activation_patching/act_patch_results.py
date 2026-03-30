"""Result types for activation patching."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..common.base_schema import BaseSchema
from ..common.choice.grouped_binary_choice import ForkAggregation
from ..inference.interventions.intervention_target import InterventionTarget
from .intervened_choice import IntervenedChoice
from .act_patch_metrics import IntervenedChoiceMetrics, LabelPerspective


@dataclass
class ActPatchTargetResult(BaseSchema):
    """Results for one target (both modes).

    When choices are GroupedBinaryChoice, use .n_labels and access individual
    label pairs via choice.get_choice(label_idx) for per-label visualization.
    """

    target: InterventionTarget
    denoising: IntervenedChoice | None = None
    noising: IntervenedChoice | None = None

    # Cached metrics (populated by pop_heavy before removing tree data)
    denoising_metrics: IntervenedChoiceMetrics | None = None
    noising_metrics: IntervenedChoiceMetrics | None = None
    # Cached combined metrics for multilabel (label_perspective="combined")
    denoising_combined_metrics: IntervenedChoiceMetrics | None = None
    noising_combined_metrics: IntervenedChoiceMetrics | None = None

    @property
    def n_labels(self) -> int:
        """Number of label pairs."""
        if self.denoising:
            return self.denoising.n_labels
        if self.noising:
            return self.noising.n_labels
        return 1

    @property
    def recovery(self) -> float | None:
        """Recovery from denoising mode."""
        return self.denoising.recovery if self.denoising else None

    @property
    def disruption(self) -> float | None:
        """Disruption from noising mode."""
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
        """Remove heavy data from choices.

        Caches metrics before removing heavy data:
        - clean perspective metrics (default)
        - combined perspective metrics (for multilabel, needs vocab_logits)
        """
        # Cache clean perspective metrics
        if self.denoising and self.denoising_metrics is None:
            self.denoising_metrics = IntervenedChoiceMetrics.from_choice(self.denoising)
        if self.noising and self.noising_metrics is None:
            self.noising_metrics = IntervenedChoiceMetrics.from_choice(self.noising)

        # Cache combined perspective metrics for multilabel (requires vocab_logits)
        if self.n_labels > 1:
            if self.denoising and self.denoising_combined_metrics is None:
                self.denoising_combined_metrics = IntervenedChoiceMetrics.from_choice(
                    self.denoising, label_perspective="combined"
                )
            if self.noising and self.noising_combined_metrics is None:
                self.noising_combined_metrics = IntervenedChoiceMetrics.from_choice(
                    self.noising, label_perspective="combined"
                )

        if self.denoising:
            self.denoising.pop_heavy()
        if self.noising:
            self.noising.pop_heavy()

    def get_denoising_metrics(
        self, label_perspective: LabelPerspective = "clean"
    ) -> IntervenedChoiceMetrics:
        """Get denoising metrics (cached or computed).

        Args:
            label_perspective: Which label system to use:
                - "clean": Use clean labels (default)
                - "corrupted": Use corrupted labels
                - "combined": Aggregate across both label systems

        Note: Cached metrics exist for "clean" and "combined" perspectives.
        For "corrupted", metrics are computed fresh from the choice data.
        """
        if label_perspective == "clean" and self.denoising_metrics is not None:
            return self.denoising_metrics
        if label_perspective == "combined" and self.denoising_combined_metrics is not None:
            return self.denoising_combined_metrics
        return IntervenedChoiceMetrics.from_choice(self.denoising, label_perspective)

    def get_noising_metrics(
        self, label_perspective: LabelPerspective = "clean"
    ) -> IntervenedChoiceMetrics:
        """Get noising metrics (cached or computed).

        Args:
            label_perspective: Which label system to use:
                - "clean": Use clean labels (default)
                - "corrupted": Use corrupted labels
                - "combined": Aggregate across both label systems

        Note: Cached metrics exist for "clean" and "combined" perspectives.
        For "corrupted", metrics are computed fresh from the choice data.
        """
        if label_perspective == "clean" and self.noising_metrics is not None:
            return self.noising_metrics
        if label_perspective == "combined" and self.noising_combined_metrics is not None:
            return self.noising_combined_metrics
        return IntervenedChoiceMetrics.from_choice(self.noising, label_perspective)

    def get_denoising_metrics_by_method(
        self, method: ForkAggregation
    ) -> IntervenedChoiceMetrics:
        """Get denoising metrics using a specific aggregation method."""
        return IntervenedChoiceMetrics.from_choice_aggregated(self.denoising, method)

    def get_noising_metrics_by_method(
        self, method: ForkAggregation
    ) -> IntervenedChoiceMetrics:
        """Get noising metrics using a specific aggregation method."""
        return IntervenedChoiceMetrics.from_choice_aggregated(self.noising, method)

    def get_denoising_metrics_per_fork(self, fork_idx: int) -> IntervenedChoiceMetrics:
        """Get denoising metrics for a specific fork (label pair)."""
        return IntervenedChoiceMetrics.from_choice_per_fork(self.denoising, fork_idx)

    def get_noising_metrics_per_fork(self, fork_idx: int) -> IntervenedChoiceMetrics:
        """Get noising metrics for a specific fork (label pair)."""
        return IntervenedChoiceMetrics.from_choice_per_fork(self.noising, fork_idx)

    def switch(self) -> ActPatchTargetResult:
        """Swap clean↔corrupted semantics."""
        return ActPatchTargetResult(
            target=self.target,
            denoising=self.noising.switch() if self.noising else None,
            noising=self.denoising.switch() if self.denoising else None,
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
        return sum(p.mean_recovery for p in self.by_sample.values()) / len(self.by_sample)

    @property
    def mean_disruption(self) -> float:
        if not self.by_sample:
            return 0.0
        return sum(p.mean_disruption for p in self.by_sample.values()) / len(self.by_sample)

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
