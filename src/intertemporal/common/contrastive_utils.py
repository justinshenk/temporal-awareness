"""PrefPairRequirement: requirements for filtering ContrastivePreferences pairs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from ...common.base_schema import BaseSchema
from .contrastive_preferences import ContrastivePreferences
from .preference_types import PreferenceSample

if TYPE_CHECKING:
    from ..preference import PreferenceDataset

log = logging.getLogger(__name__)

GroupByMode = Literal["content", "horizon", "choice"]


@dataclass
class PrefPairRequirement(BaseSchema):
    """Requirements for filtering ContrastivePreferences pairs.

    All fields default to False (no requirement). Set to True to require
    the corresponding property on ContrastivePreferences.
    """

    # Label requirements (both False = no requirement, allows multilabel pairing)
    same_labels: bool = False
    different_labels: bool = False

    # Context requirements
    same_context: bool = False
    different_context: bool = False

    # Formatting requirements
    same_formatting: bool = False
    different_formatting: bool = False

    # Reward requirements
    same_rewards: bool = False
    different_rewards: bool = False

    # Time requirements
    same_times: bool = False
    different_times: bool = False

    # Horizon requirements
    same_horizon: bool = False
    different_horizon: bool = False
    neither_horizon: bool = False
    both_horizon: bool = False
    only_short_horizon: bool = False
    only_long_horizon: bool = False
    only_one_horizon: bool = False

    # Rational choice requirements
    both_rational: bool = False
    neither_rational: bool = False
    only_short_rational: bool = False
    only_long_rational: bool = False
    only_one_rational: bool = False

    # Associated choice requirements
    both_associated: bool = False
    neither_associated: bool = False
    only_short_associated: bool = False
    only_long_associated: bool = False
    only_one_associated: bool = False

    def verify(self) -> None:
        """Verify requirements are logically consistent.

        Raises:
            ValueError: If requirements are contradictory.
        """
        errors = []

        # Check mutually exclusive pairs
        exclusive_pairs = [
            ("same_labels", "different_labels"),
            ("same_context", "different_context"),
            ("same_formatting", "different_formatting"),
            ("same_rewards", "different_rewards"),
            ("same_times", "different_times"),
            ("same_horizon", "different_horizon"),
            ("both_rational", "neither_rational"),
            ("both_associated", "neither_associated"),
        ]
        for a, b in exclusive_pairs:
            if getattr(self, a) and getattr(self, b):
                errors.append(f"{a} and {b} are mutually exclusive")

        # Horizon: mutually exclusive groups
        horizon_exclusive = [
            "neither_horizon",
            "both_horizon",
            "only_short_horizon",
            "only_long_horizon",
        ]
        active_horizon = [h for h in horizon_exclusive if getattr(self, h)]
        if len(active_horizon) > 1:
            errors.append(
                f"Horizon requirements are mutually exclusive: {active_horizon}"
            )

        # same_horizon requires both_horizon (can't be same if one is missing)
        if self.same_horizon and self.neither_horizon:
            errors.append("same_horizon requires both samples to have horizons")
        if self.same_horizon and self.only_one_horizon:
            errors.append("same_horizon is incompatible with only_one_horizon")
        if self.same_horizon and (self.only_short_horizon or self.only_long_horizon):
            errors.append("same_horizon is incompatible with only_short/long_horizon")

        # different_horizon with neither_horizon is impossible
        if self.different_horizon and self.neither_horizon:
            errors.append(
                "different_horizon requires at least one sample to have horizon"
            )

        # Rational: mutually exclusive groups
        rational_exclusive = [
            "both_rational",
            "neither_rational",
            "only_short_rational",
            "only_long_rational",
        ]
        active_rational = [r for r in rational_exclusive if getattr(self, r)]
        if len(active_rational) > 1:
            errors.append(
                f"Rational requirements are mutually exclusive: {active_rational}"
            )

        # Associated: mutually exclusive groups
        associated_exclusive = [
            "both_associated",
            "neither_associated",
            "only_short_associated",
            "only_long_associated",
        ]
        active_associated = [a for a in associated_exclusive if getattr(self, a)]
        if len(active_associated) > 1:
            errors.append(
                f"Associated requirements are mutually exclusive: {active_associated}"
            )

        # only_one_X with specific only_short/long is redundant but not invalid
        # (only_one_horizon with only_short_horizon is just more specific)

        if errors:
            raise ValueError(f"Invalid PrefPairRequirement: {'; '.join(errors)}")

    def passes(self, pair: ContrastivePreferences) -> bool:
        """Check if a ContrastivePreferences pair passes all requirements."""
        self.verify()

        # Label checks
        if self.same_labels and not pair.same_labels:
            return False
        if self.different_labels and pair.same_labels:
            return False

        # Context checks
        if self.same_context and not pair.same_context:
            return False
        if self.different_context and pair.same_context:
            return False

        # Formatting checks
        if self.same_formatting and not pair.same_formatting:
            return False
        if self.different_formatting and pair.same_formatting:
            return False

        # Reward checks
        if self.same_rewards and not pair.same_rewards:
            return False
        if self.different_rewards and pair.same_rewards:
            return False

        # Time checks
        if self.same_times and not pair.same_times:
            return False
        if self.different_times and pair.same_times:
            return False

        # Horizon checks
        if self.same_horizon and not pair.same_horizon:
            return False
        if self.different_horizon and pair.same_horizon:
            return False
        if self.neither_horizon and not pair.neither_horizon:
            return False
        if self.both_horizon and not pair.both_horizon:
            return False
        if self.only_short_horizon and not pair.only_short_horizon:
            return False
        if self.only_long_horizon and not pair.only_long_horizon:
            return False
        if self.only_one_horizon and not pair.only_one_horizon:
            return False

        # Rational checks
        if self.both_rational and not pair.both_rational:
            return False
        if self.neither_rational and not pair.neither_rational:
            return False
        if self.only_short_rational and not pair.only_short_rational:
            return False
        if self.only_long_rational and not pair.only_long_rational:
            return False
        if self.only_one_rational and not pair.only_one_rational:
            return False

        # Associated checks
        if self.both_associated and not pair.both_associated:
            return False
        if self.neither_associated and not pair.neither_associated:
            return False
        if self.only_short_associated and not pair.only_short_associated:
            return False
        if self.only_long_associated and not pair.only_long_associated:
            return False
        if self.only_one_associated and not pair.only_one_associated:
            return False

        return True


def get_contrastive_preferences(
    dataset: PreferenceDataset,
    req: PrefPairRequirement | None = None,
    group_by: GroupByMode = "choice",
    deduplicate: bool = False,
    best_only: bool = False,
    min_confidence: float = 0.0,
) -> list[ContrastivePreferences]:
    """Find pairs of samples with different choices for contrastive analysis.

    Grouping modes:
    - "content": Group by reward/time values. Pairs share same content but differ
      in horizon. Isolates the horizon effect.
    - "horizon": Group by horizon value. Pairs share same horizon but may differ
      in rewards/times. Isolates reward/time sensitivity.
    - "choice": No grouping - pair any short-chooser with any long-chooser.
      Maximum volume for PCA, noisier but reveals dominant separating direction.

    Args:
        dataset: PreferenceDataset containing samples to search
        req: Optional PrefPairRequirement specifying filtering requirements
        group_by: Grouping mode ("content", "horizon", or "choice")
        deduplicate: If True, keep only one pair per unique content×horizon
            combination within each group. Reduces redundancy from formatting
            variations. Recommended for geometry analysis.
        best_only: If True, only keep the single best pair per group (highest
            confidence short + highest confidence long). Creates one high-quality
            pair per content/horizon group instead of all pairwise combinations.
        min_confidence: Minimum choice probability threshold. Pairs where either
            sample has choice_prob below this are filtered out.

    Returns:
        List of ContrastivePreferences pairs sorted by confidence
    """
    if req is None:
        req = PrefPairRequirement()
    req.verify()

    # Collect samples by choice
    short_choosers: list[PreferenceSample] = []
    long_choosers: list[PreferenceSample] = []
    for pref in dataset.preferences:
        if pref.choice_term == "short_term":
            short_choosers.append(pref)
        elif pref.choice_term == "long_term":
            long_choosers.append(pref)

    # Build groups based on mode
    if group_by == "choice":
        # No grouping - all samples in one group
        groups: dict[tuple, tuple[list[PreferenceSample], list[PreferenceSample]]] = {
            (): (short_choosers, long_choosers)
        }
    elif group_by == "horizon":
        # Group by horizon value
        horizon_groups: dict[
            float | None, tuple[list[PreferenceSample], list[PreferenceSample]]
        ] = {}
        for s in short_choosers:
            h = s.time_horizon
            if h not in horizon_groups:
                horizon_groups[h] = ([], [])
            horizon_groups[h][0].append(s)
        for s in long_choosers:
            h = s.time_horizon
            if h not in horizon_groups:
                horizon_groups[h] = ([], [])
            horizon_groups[h][1].append(s)
        groups = {(h,): v for h, v in horizon_groups.items()}
    else:
        # group_by == "content" (default, original behavior)
        # Group by reward/time values
        content_groups: dict[
            tuple, tuple[list[PreferenceSample], list[PreferenceSample]]
        ] = {}
        for s in short_choosers:
            key = (
                s.short_term_reward,
                s.long_term_reward,
                s.short_term_time,
                s.long_term_time,
            )
            if key not in content_groups:
                content_groups[key] = ([], [])
            content_groups[key][0].append(s)
        for s in long_choosers:
            key = (
                s.short_term_reward,
                s.long_term_reward,
                s.short_term_time,
                s.long_term_time,
            )
            if key not in content_groups:
                content_groups[key] = ([], [])
            content_groups[key][1].append(s)
        groups = content_groups

    # Generate pairs within each group
    pairs: list[ContrastivePreferences] = []
    total_candidates = 0
    total_passed = 0

    for group_key, (group_short, group_long) in groups.items():
        if best_only:
            # Only pair the best (highest confidence) short with best long
            sorted_short = sorted(
                group_short, key=lambda s: s.choice_prob, reverse=True
            )
            sorted_long = sorted(
                group_long, key=lambda s: s.choice_prob, reverse=True
            )
            if sorted_short and sorted_long:
                total_candidates += 1
                candidate = ContrastivePreferences(
                    short_term=sorted_short[0],
                    long_term=sorted_long[0],
                )
                if req.passes(candidate) and candidate.min_choice_prob >= min_confidence:
                    total_passed += 1
                    pairs.append(candidate)
        else:
            # All pairwise combinations
            for short_sample in group_short:
                for long_sample in group_long:
                    total_candidates += 1
                    candidate = ContrastivePreferences(
                        short_term=short_sample,
                        long_term=long_sample,
                    )
                    if req.passes(candidate) and candidate.min_choice_prob >= min_confidence:
                        total_passed += 1
                        pairs.append(candidate)

    # Deduplication: keep one pair per unique content×horizon combination
    if deduplicate and not best_only:  # best_only already gives one per group
        seen: set[tuple] = set()
        unique_pairs: list[ContrastivePreferences] = []
        for p in pairs:
            # Key: content (rewards/times) + horizons
            dedup_key = (
                p.short_term.short_term_reward,
                p.short_term.long_term_reward,
                p.short_term.short_term_time,
                p.short_term.long_term_time,
                p.short_term.time_horizon,
                p.long_term.time_horizon,
            )
            if dedup_key not in seen:
                seen.add(dedup_key)
                unique_pairs.append(p)
        log.info(
            f"Deduplication: {len(pairs)} -> {len(unique_pairs)} pairs "
            f"({len(pairs) - len(unique_pairs)} duplicates removed)"
        )
        pairs = unique_pairs

    log.info(
        f"Contrastive pairs (group_by={group_by}, best_only={best_only}): "
        f"{len(short_choosers)} short, {len(long_choosers)} long, "
        f"{total_candidates} candidates, {total_passed} passed, {len(pairs)} final"
    )

    # Sort by minimum choice probability (highest confidence pairs first)
    pairs.sort(key=lambda p: p.min_choice_prob, reverse=True)

    assert len(pairs) != 0, "Bad config! No contrastive pairs pass requirements."

    return pairs
