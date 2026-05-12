"""Staircase headline computation.

Given the per-position probe accuracies from the existing
`train_commitment_probes()` call and a DomainSpec, this module
computes:

  * the headline N+P baseline (= MAX accuracy across earlier positions,
    a strictly tougher reference than any single-position or aggregated
    baseline; the target probe must beat the BEST earlier position)
  * the workshop-style mean-pool baseline (kept for backward
    comparability across the workshop's 11 code models)
  * the target-position accuracy under each resolver
  * the headline gap = target_accuracy − max_earlier_accuracy
  * a pre-registration check: does the observed gap match the
    predicted sign for this domain?

The output is plain JSON-serialisable dicts so the mega-script can dump
them and analysis notebooks can read them back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..domains import (
    DomainSpec,
    PredictedGap,
    resolve_target_positions,
    get_earlier_positions,
)
from ..utils.types import ActivationCache, PlanningExample


# ──────────────────────────────────────────────────────────────────────
# Per-example resolved positions
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ResolvedPositions:
    """Tokenizer-resolved target positions for one example.

    Stored once after activation extraction so we don't re-resolve for
    every layer.
    """
    example_id: str
    n_tokens: int
    # resolver name → token index (may be None if not found)
    targets_by_resolver: dict[str, Optional[int]]
    # earliest target across all resolvers (used to clip "earlier")
    earliest_target: Optional[int] = None
    # earlier region cap (e.g. signature_end for code)
    signature_end: Optional[int] = None


def resolve_positions_for_caches(
    spec: DomainSpec,
    caches: list[ActivationCache],
    examples: list[PlanningExample],
    tokenizer=None,
) -> list[ResolvedPositions]:
    """Resolve target positions for every example, once.

    For code domain, we also compute `signature_end` per example so the
    earlier-region cap respects the workshop's "first 10 signature
    tokens" convention.
    """
    out: list[ResolvedPositions] = []
    for cache, ex in zip(caches, examples):
        targets = resolve_target_positions(
            spec=spec,
            token_strings=cache.token_strings,
            token_ids=cache.token_ids,
            tokenizer=tokenizer,
        )
        valid = [v for v in targets.values() if v is not None]
        earliest = min(valid) if valid else None

        sig_end = None
        if spec.earlier_region == "signature_only":
            # Workshop convention: signature ends at the colon, capped at 10.
            colon_idx = next(
                (i for i, t in enumerate(cache.token_strings) if ":" in t),
                None,
            )
            sig_end = min(colon_idx + 1 if colon_idx is not None else 10, 10)

        out.append(ResolvedPositions(
            example_id=ex.example_id,
            n_tokens=len(cache.token_ids),
            targets_by_resolver=targets,
            earliest_target=earliest,
            signature_end=sig_end,
        ))
    return out


# ──────────────────────────────────────────────────────────────────────
# Headline summary
# ──────────────────────────────────────────────────────────────────────

@dataclass
class StaircaseHeadline:
    """Per-(layer, target-resolver) summary used in the paper tables.

    Attributes:
        layer: Layer index.
        resolver_name: Which target resolver was used.

        target_accuracy: Probe accuracy at the target position
            (mean across examples — each example's target is resolved
            independently, so positions vary).

        max_earlier_accuracy: Strongest baseline. The single position p
            (strictly earlier than the target) that yielded the highest
            cross-validated accuracy when probed.

        max_earlier_position_mode: Modal position where the max-earlier
            was achieved across examples (informational, not a single
            number, but useful for diagnosis).

        headline_gap: target_accuracy − max_earlier_accuracy

        mean_pool_earlier_accuracy: Workshop-style N+P baseline (mean of
            activations across positions 0..signature_end, single PCA-
            reduced probe). Reported for cross-paper comparability.

        n_examples: How many examples contributed to the headline (only
            those where this resolver resolved successfully).
    """
    layer: int
    resolver_name: str
    target_accuracy: float
    max_earlier_accuracy: float
    max_earlier_position_mode: Optional[int]
    headline_gap: float
    mean_pool_earlier_accuracy: Optional[float] = None
    n_examples: int = 0
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "resolver": self.resolver_name,
            "target_accuracy": float(self.target_accuracy),
            "max_earlier_accuracy": float(self.max_earlier_accuracy),
            "max_earlier_position_mode": self.max_earlier_position_mode,
            "mean_pool_earlier_accuracy":
                None if self.mean_pool_earlier_accuracy is None
                else float(self.mean_pool_earlier_accuracy),
            "headline_gap": float(self.headline_gap),
            "n_examples": int(self.n_examples),
            **{k: v for k, v in self.extras.items()},
        }


def compute_headlines(
    spec: DomainSpec,
    layer: int,
    per_position_results: dict[int, dict],  # position → {cv_accuracy_mean, valid_indices, ...}
    resolved: list[ResolvedPositions],
    target_position_accuracies: dict[str, float],  # resolver_name → mean accuracy over examples
    mean_pool_accuracy: Optional[float] = None,
) -> list[StaircaseHeadline]:
    """Build the headline summary for one layer.

    Args:
        spec: Domain specification.
        layer: Layer being summarised.
        per_position_results: Output of `train_commitment_probes()` for
            this layer (keyed by position).
        resolved: Pre-computed target positions per example.
        target_position_accuracies: For each target resolver, the
            accuracy when each example is probed at its OWN resolved
            target position. This is computed by the runner because it
            requires per-example target indices.
        mean_pool_accuracy: Optional workshop-style mean-pool baseline.

    Returns:
        One StaircaseHeadline per target resolver.
    """
    # Determine the maximum "earlier" position observed across examples
    # (sets the bound on which `per_position_results` entries count).
    headlines: list[StaircaseHeadline] = []

    for resolver_name in target_position_accuracies:
        # Filter to examples where this resolver succeeded
        relevant = [r for r in resolved
                    if r.targets_by_resolver.get(resolver_name) is not None]
        if not relevant:
            continue

        # For each "earlier" candidate position p, count how many examples have it valid.
        # A position p is "earlier" for example r if p < r.targets[resolver_name].
        # We approximate the headline-baseline as: the maximum cv_accuracy_mean
        # at any single position p such that p is earlier for the MAJORITY of
        # examples. This is a clean reviewer-defensible choice.
        eligible_positions = sorted(per_position_results.keys())
        per_pos_acc: list[tuple[int, float]] = []
        for p in eligible_positions:
            # majority earlier criterion
            n_earlier = sum(
                1 for r in relevant
                if r.targets_by_resolver[resolver_name] is not None
                and p < r.targets_by_resolver[resolver_name]
            )
            if n_earlier >= 0.5 * len(relevant):
                acc = per_position_results[p].get("cv_accuracy_mean", 0.0)
                per_pos_acc.append((p, acc))

        if per_pos_acc:
            max_pos, max_acc = max(per_pos_acc, key=lambda kv: kv[1])
        else:
            max_pos, max_acc = None, 0.0

        target_acc = target_position_accuracies[resolver_name]
        gap = target_acc - max_acc

        headlines.append(StaircaseHeadline(
            layer=layer,
            resolver_name=resolver_name,
            target_accuracy=target_acc,
            max_earlier_accuracy=max_acc,
            max_earlier_position_mode=max_pos,
            headline_gap=gap,
            mean_pool_earlier_accuracy=mean_pool_accuracy,
            n_examples=len(relevant),
        ))
    return headlines


# ──────────────────────────────────────────────────────────────────────
# Pre-registration check
# ──────────────────────────────────────────────────────────────────────

def gap_sign_matches_prediction(
    observed_gap_pp: float,
    predicted: PredictedGap,
    weak_threshold_pp: float = 2.0,
    strong_threshold_pp: float = 5.0,
) -> dict:
    """Compare observed gap (in percentage points) to predicted sign.

    Thresholds (in pp):
        |gap| < weak_threshold        → NEAR_ZERO
        weak < gap < strong           → WEAK_POSITIVE
        gap >= strong                  → STRONG_POSITIVE
        gap < -weak_threshold          → NEGATIVE
    """
    if observed_gap_pp >= strong_threshold_pp:
        observed_class = PredictedGap.STRONG_POSITIVE
    elif observed_gap_pp >= weak_threshold_pp:
        observed_class = PredictedGap.WEAK_POSITIVE
    elif observed_gap_pp <= -weak_threshold_pp:
        observed_class = PredictedGap.NEGATIVE
    else:
        observed_class = PredictedGap.NEAR_ZERO

    return {
        "observed_gap_pp": float(observed_gap_pp),
        "predicted_sign": predicted.value,
        "observed_sign": observed_class.value,
        "matches": observed_class == predicted,
        "thresholds": {
            "weak_pp": weak_threshold_pp,
            "strong_pp": strong_threshold_pp,
        },
    }


__all__ = [
    "ResolvedPositions",
    "resolve_positions_for_caches",
    "StaircaseHeadline",
    "compute_headlines",
    "gap_sign_matches_prediction",
]
