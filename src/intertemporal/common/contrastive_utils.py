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

    # Order requirements (option ordering: short-term first vs long-term first)
    same_order: bool = False
    different_order: bool = False

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
            ("same_order", "different_order"),
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

        # Order checks
        if self.same_order and not pair.same_order:
            return False
        if self.different_order and pair.same_order:
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


SmartReduceMode = Literal["balanced", "diverse", "minimal"]
SelectionStrategy = Literal["greedy", "round_robin"]


def get_contrastive_preferences(
    dataset: PreferenceDataset,
    req: PrefPairRequirement | None = None,
    group_by: GroupByMode = "choice",
    deduplicate: bool = False,
    best_only: bool = False,
    min_confidence: float = 0.0,
    max_per_sample: int | None = None,
    max_per_horizon_pair: int | None = None,
    max_per_reward_ratio: int | None = None,
    max_per_confidence_bucket: int | None = None,
    smart_reduce: SmartReduceMode | None = None,
    prefer_different_horizon: bool = False,
    target_pairs: int | None = None,
    selection_strategy: SelectionStrategy = "greedy",
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
        max_per_sample: Maximum number of pairs each sample can participate in.
            When set, pairs are selected greedily (highest confidence first) while
            respecting this limit. Ensures balanced coverage without explosion.
            Useful with group_by="choice" to reduce 975 pairs to ~50-100.
        max_per_horizon_pair: Maximum pairs per (short_horizon, long_horizon) combo.
            Ensures diversity across horizon combinations. Applied before
            max_per_sample for stratified reduction.
        max_per_reward_ratio: Maximum pairs per reward ratio (long/short).
            Ensures diversity across different preference strengths.
        max_per_confidence_bucket: Maximum pairs per confidence bucket.
            Buckets are [0.5-0.6), [0.6-0.7), [0.7-0.8), [0.8-0.9), [0.9-1.0].
            Ensures pairs span different confidence levels, not just the highest.
        smart_reduce: Convenience presets for balanced reduction:
            - "balanced": ~60-80 pairs, good signal diversity
            - "diverse": ~40-60 pairs, maximizes dimension coverage
            - "minimal": ~20-30 pairs, aggressive reduction
        prefer_different_horizon: If True, prioritize pairs where short and long
            choosers have different horizons (clearer activation patching signal).
        target_pairs: Target number of output pairs. Auto-calculates max_per_sample
            to achieve this target. Useful when you know how many pairs you want
            rather than tuning max_per_sample manually.
        selection_strategy: How to select pairs when applying limits:
            - "greedy": Take highest confidence pairs first (default)
            - "round_robin": Cycle through horizon combinations to maximize diversity

    Returns:
        List of ContrastivePreferences pairs sorted by confidence
    """
    if req is None:
        req = PrefPairRequirement()
    req.verify()

    # Apply smart_reduce presets (only set parameters not already specified)
    # These are applied sequentially: horizon -> ratio -> sample
    # Use only max_per_sample for simpler, more predictable reduction
    if smart_reduce == "balanced":
        # Target ~60-80 pairs: each sample used ~3x
        max_per_sample = max_per_sample or 3
    elif smart_reduce == "diverse":
        # Target ~40-50 pairs: each sample used ~2x
        max_per_sample = max_per_sample or 2
    elif smart_reduce == "minimal":
        # Target ~25 pairs: each sample used once
        max_per_sample = max_per_sample or 1

    # Collect samples by choice (needed early for target_pairs calculation)
    short_choosers: list[PreferenceSample] = []
    long_choosers: list[PreferenceSample] = []
    for pref in dataset.preferences:
        if pref.choice_term == "short_term":
            short_choosers.append(pref)
        elif pref.choice_term == "long_term":
            long_choosers.append(pref)

    # Target pairs: auto-calculate max_per_sample to achieve target
    if target_pairs is not None and target_pairs > 0 and max_per_sample is None:
        n_short = len(short_choosers)
        n_long = len(long_choosers)
        n_total = n_short + n_long

        if n_total > 0:
            # Each pair uses 2 samples (one short, one long)
            # With max_per_sample=k, we get roughly min(n_short, n_long) * k pairs
            # Solve: target = min(n_short, n_long) * k => k = target / min(n_short, n_long)
            min_side = min(n_short, n_long)
            if min_side > 0:
                estimated_k = max(1, int(target_pairs / min_side + 0.5))
                max_per_sample = estimated_k
                log.info(
                    f"Target pairs ({target_pairs}): estimated max_per_sample={max_per_sample} "
                    f"(n_short={n_short}, n_long={n_long})"
                )

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
            sorted_long = sorted(group_long, key=lambda s: s.choice_prob, reverse=True)
            if sorted_short and sorted_long:
                total_candidates += 1
                candidate = ContrastivePreferences(
                    short_term=sorted_short[0],
                    long_term=sorted_long[0],
                )
                if (
                    req.passes(candidate)
                    and candidate.min_choice_prob >= min_confidence
                ):
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
                    if (
                        req.passes(candidate)
                        and candidate.min_choice_prob >= min_confidence
                    ):
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

    # Prefer different horizon: sort so different-horizon pairs come first
    if prefer_different_horizon:
        from ...common.time_value import parse_horizon_years

        def horizon_priority(p: ContrastivePreferences) -> tuple:
            h_short = parse_horizon_years(p.short_term.time_horizon)
            h_long = parse_horizon_years(p.long_term.time_horizon)
            # 0 = different horizons (preferred), 1 = same or one missing
            same_or_missing = (h_short == h_long) or (h_short is None) or (h_long is None)
            return (same_or_missing, -p.min_choice_prob)

        pairs.sort(key=horizon_priority)
        n_different = sum(
            1 for p in pairs
            if parse_horizon_years(p.short_term.time_horizon) != parse_horizon_years(p.long_term.time_horizon)
            and parse_horizon_years(p.short_term.time_horizon) is not None
            and parse_horizon_years(p.long_term.time_horizon) is not None
        )
        log.info(f"Prefer different horizon: {n_different}/{len(pairs)} pairs have different horizons")

    # Max per horizon pair: stratified selection by horizon combination
    if max_per_horizon_pair is not None and max_per_horizon_pair > 0:
        from ...common.time_value import parse_horizon_years

        # Sort by confidence first
        pairs.sort(key=lambda p: p.min_choice_prob, reverse=True)

        # Group by horizon pair and take top K from each
        horizon_pair_counts: dict[tuple, int] = {}
        stratified_pairs: list[ContrastivePreferences] = []

        for p in pairs:
            h_short = parse_horizon_years(p.short_term.time_horizon)
            h_long = parse_horizon_years(p.long_term.time_horizon)
            key = (h_short, h_long)

            count = horizon_pair_counts.get(key, 0)
            if count < max_per_horizon_pair:
                stratified_pairs.append(p)
                horizon_pair_counts[key] = count + 1

        log.info(
            f"Max per horizon pair ({max_per_horizon_pair}): {len(pairs)} -> {len(stratified_pairs)} pairs "
            f"({len(horizon_pair_counts)} horizon combinations)"
        )
        pairs = stratified_pairs

    # Max per reward ratio: stratified selection by preference strength
    if max_per_reward_ratio is not None and max_per_reward_ratio > 0:
        pairs.sort(key=lambda p: p.min_choice_prob, reverse=True)

        ratio_counts: dict[float, int] = {}
        ratio_pairs: list[ContrastivePreferences] = []

        for p in pairs:
            if p.short_term.short_term_reward and p.short_term.long_term_reward:
                ratio = round(
                    p.short_term.long_term_reward / p.short_term.short_term_reward, 1
                )
            else:
                ratio = 0.0

            count = ratio_counts.get(ratio, 0)
            if count < max_per_reward_ratio:
                ratio_pairs.append(p)
                ratio_counts[ratio] = count + 1

        log.info(
            f"Max per reward ratio ({max_per_reward_ratio}): {len(pairs)} -> {len(ratio_pairs)} pairs "
            f"({len(ratio_counts)} reward ratios)"
        )
        pairs = ratio_pairs

    # Max per confidence bucket: ensure diversity across confidence levels
    if max_per_confidence_bucket is not None and max_per_confidence_bucket > 0:
        # Define buckets: [0.5, 0.6), [0.6, 0.7), [0.7, 0.8), [0.8, 0.9), [0.9, 1.0]
        def get_confidence_bucket(conf: float) -> float:
            if conf < 0.6:
                return 0.5
            elif conf < 0.7:
                return 0.6
            elif conf < 0.8:
                return 0.7
            elif conf < 0.9:
                return 0.8
            else:
                return 0.9

        bucket_counts: dict[float, int] = {}
        bucket_pairs: list[ContrastivePreferences] = []

        # First pass: count pairs per bucket
        for p in pairs:
            bucket = get_confidence_bucket(p.min_choice_prob)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        # Reset and select
        bucket_counts = {}
        for p in pairs:
            bucket = get_confidence_bucket(p.min_choice_prob)
            count = bucket_counts.get(bucket, 0)
            if count < max_per_confidence_bucket:
                bucket_pairs.append(p)
                bucket_counts[bucket] = count + 1

        log.info(
            f"Max per confidence bucket ({max_per_confidence_bucket}): {len(pairs)} -> {len(bucket_pairs)} pairs "
            f"(buckets: {dict(sorted(bucket_counts.items()))})"
        )
        pairs = bucket_pairs

    # Max per sample: limit how many pairs each sample participates in
    if max_per_sample is not None and max_per_sample > 0:
        # Sort by confidence first (highest confidence pairs selected first)
        pairs.sort(key=lambda p: p.min_choice_prob, reverse=True)

        # Track usage count per sample
        sample_usage: dict[int, int] = {}
        limited_pairs: list[ContrastivePreferences] = []

        for p in pairs:
            short_idx = p.short_term.sample_idx
            long_idx = p.long_term.sample_idx

            short_count = sample_usage.get(short_idx, 0)
            long_count = sample_usage.get(long_idx, 0)

            # Only include if both samples are under the limit
            if short_count < max_per_sample and long_count < max_per_sample:
                limited_pairs.append(p)
                sample_usage[short_idx] = short_count + 1
                sample_usage[long_idx] = long_count + 1

        log.info(
            f"Max per sample ({max_per_sample}): {len(pairs)} -> {len(limited_pairs)} pairs"
        )
        pairs = limited_pairs

    # Apply selection strategy for final ordering
    if selection_strategy == "round_robin" and len(pairs) > 0:
        from ...common.time_value import parse_horizon_years

        # Group pairs by horizon combination
        horizon_groups: dict[tuple, list[ContrastivePreferences]] = {}
        for p in pairs:
            h_short = parse_horizon_years(p.short_term.time_horizon)
            h_long = parse_horizon_years(p.long_term.time_horizon)
            key = (h_short, h_long)
            if key not in horizon_groups:
                horizon_groups[key] = []
            horizon_groups[key].append(p)

        # Sort each group by confidence
        for group_pairs in horizon_groups.values():
            group_pairs.sort(key=lambda p: p.min_choice_prob, reverse=True)

        # Round-robin selection across groups
        round_robin_pairs: list[ContrastivePreferences] = []
        group_keys = list(horizon_groups.keys())
        indices = {k: 0 for k in group_keys}

        while len(round_robin_pairs) < len(pairs):
            added_any = False
            for key in group_keys:
                idx = indices[key]
                if idx < len(horizon_groups[key]):
                    round_robin_pairs.append(horizon_groups[key][idx])
                    indices[key] = idx + 1
                    added_any = True
            if not added_any:
                break

        log.info(
            f"Round-robin selection: {len(horizon_groups)} horizon groups, "
            f"interleaved {len(round_robin_pairs)} pairs"
        )
        pairs = round_robin_pairs

    log.info(
        f"Contrastive pairs (group_by={group_by}, best_only={best_only}): "
        f"{len(short_choosers)} short, {len(long_choosers)} long, "
        f"{total_candidates} candidates, {total_passed} passed, {len(pairs)} final"
    )

    # Sort by minimum choice probability (highest confidence pairs first)
    pairs.sort(key=lambda p: p.min_choice_prob, reverse=True)

    if len(pairs) == 0:
        log.warning(
            "No contrastive pairs pass requirements! Check your filters. "
            f"Had {total_candidates} candidates, {total_passed} passed filters."
        )

    return pairs
