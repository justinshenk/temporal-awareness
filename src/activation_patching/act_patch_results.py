"""Result types for activation patching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..common.base_schema import BaseSchema
from ..common.choice import LabeledSimpleBinaryChoice
from ..inference.interventions.intervention_target import InterventionTarget


@dataclass
class IntervenedChoice(BaseSchema):
    """Single intervention result."""

    original: LabeledSimpleBinaryChoice
    intervened: LabeledSimpleBinaryChoice
    mode: Literal["noising", "denoising"]

    @property
    def recovery(self) -> float:
        """0=no change, 0.5=boundary, 1=full flip."""
        orig = self.original.choice_logprob - self.original.alternative_logprob
        if orig == 0:
            return 0.0
        # Use original's reference frame
        idx = self.original.choice_idx
        lps = self.intervened._divergent_logprobs
        intv = lps[idx] - lps[1 - idx]
        return (orig - intv) / (2 * abs(orig))

    @property
    def flipped(self) -> bool:
        return self.original.choice_idx != self.intervened.choice_idx


@dataclass
class ActPatchTargetResult(BaseSchema):
    """Results for one target (both modes)."""

    target: InterventionTarget
    denoising: IntervenedChoice | None = None
    noising: IntervenedChoice | None = None

    @property
    def recovery(self) -> float:
        recs = [r.recovery for r in [self.denoising, self.noising] if r]
        return sum(recs) / len(recs) if recs else 0.0

    @property
    def flip_count(self) -> int:
        return sum(1 for r in [self.denoising, self.noising] if r and r.flipped)

    def score(self) -> float:
        """Score for sorting/ranking (higher = more important)."""
        return self.recovery


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
        return sum(r.recovery for r in self.by_target.values()) / len(self.by_target)


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

    def get_recovery_by_layer(self) -> dict[int | None, float]:
        """Mean recovery per layer."""
        by_layer: dict[int | None, list[float]] = {}
        for pair in self.by_sample.values():
            for target, result in pair.by_target.items():
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
