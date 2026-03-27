"""Utilities for creating and filtering contrastive preference pairs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ...common.time_value import parse_horizon_years
from .contrastive_preferences import ContrastivePreferences
from .pref_pair_requirement import PrefPairRequirement
from .pref_pair_subsample import PrefPairSubsampleStrategy
from .preference_types import PreferenceSample

if TYPE_CHECKING:
    from ..preference import PreferenceDataset

log = logging.getLogger(__name__)


def get_contrastive_preferences(
    dataset: "PreferenceDataset",
    req: PrefPairRequirement | None = None,
    subsample: PrefPairSubsampleStrategy | None = None,
    **kwargs,
) -> list[ContrastivePreferences]:
    """Find pairs of samples with different choices for contrastive analysis.

    Args:
        dataset: PreferenceDataset containing samples to search
        req: Optional PrefPairRequirement specifying filtering requirements
        subsample: Optional PrefPairSubsampleStrategy for reduction settings.
            If not provided, can pass individual kwargs (legacy interface).
        **kwargs: Legacy interface - individual subsample strategy fields.

    Returns:
        List of ContrastivePreferences pairs sorted by confidence
    """
    if req is None:
        req = PrefPairRequirement()
    req.verify()

    # Build strategy from subsample or kwargs
    if subsample is not None:
        strat = subsample
    elif kwargs:
        strat = PrefPairSubsampleStrategy.from_dict(kwargs)
    else:
        strat = PrefPairSubsampleStrategy()

    # Collect samples by choice
    short_choosers, long_choosers = _collect_samples_by_choice(dataset)
    n_samples = len(short_choosers) + len(long_choosers)

    # Apply smart_reduce preset (only if n_samples > 5)
    strat = strat.apply_smart_reduce(n_samples)

    # Calculate max_per_sample from target_pairs if needed
    max_per_sample = _calculate_max_per_sample(strat, short_choosers, long_choosers)

    # Build groups based on mode
    groups = _build_groups(strat, short_choosers, long_choosers)

    # Generate candidate pairs within each group
    pairs, total_candidates, total_passed = _generate_candidate_pairs(
        groups, req, strat
    )

    # Apply filters in order
    pairs = _apply_deduplication(pairs, strat)
    pairs = _apply_prefer_different_horizon(pairs, strat)
    pairs = _apply_max_per_horizon_pair(pairs, strat)
    pairs = _apply_max_per_reward_ratio(pairs, strat)
    pairs = _apply_max_per_confidence_bucket(pairs, strat)
    pairs = _apply_max_per_sample(pairs, max_per_sample)
    pairs = _apply_selection_strategy(pairs, strat)

    log.info(
        f"Contrastive pairs (group_by={strat.group_by}, best_only={strat.best_only}): "
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


# =============================================================================
# Helper Functions
# =============================================================================


def _collect_samples_by_choice(
    dataset: "PreferenceDataset",
) -> tuple[list[PreferenceSample], list[PreferenceSample]]:
    """Collect samples grouped by their choice (short_term vs long_term)."""
    short_choosers: list[PreferenceSample] = []
    long_choosers: list[PreferenceSample] = []
    for pref in dataset.preferences:
        if pref.choice_term == "short_term":
            short_choosers.append(pref)
        elif pref.choice_term == "long_term":
            long_choosers.append(pref)
    return short_choosers, long_choosers


def _calculate_max_per_sample(
    strat: PrefPairSubsampleStrategy,
    short_choosers: list[PreferenceSample],
    long_choosers: list[PreferenceSample],
) -> int | None:
    """Calculate max_per_sample from target_pairs if specified."""
    max_per_sample = strat.max_per_sample

    if (
        strat.target_pairs is not None
        and strat.target_pairs > 0
        and max_per_sample is None
    ):
        n_short = len(short_choosers)
        n_long = len(long_choosers)
        n_total = n_short + n_long

        if n_total > 0:
            min_side = min(n_short, n_long)
            if min_side > 0:
                estimated_k = max(1, int(strat.target_pairs / min_side + 0.5))
                max_per_sample = estimated_k
                log.info(
                    f"Target pairs ({strat.target_pairs}): estimated max_per_sample={max_per_sample} "
                    f"(n_short={n_short}, n_long={n_long})"
                )

    return max_per_sample


def _build_groups(
    strat: PrefPairSubsampleStrategy,
    short_choosers: list[PreferenceSample],
    long_choosers: list[PreferenceSample],
) -> dict[tuple, tuple[list[PreferenceSample], list[PreferenceSample]]]:
    """Build groups of samples based on grouping mode."""
    if strat.group_by == "choice":
        # No grouping - all samples in one group
        return {(): (short_choosers, long_choosers)}

    elif strat.group_by == "horizon":
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
        return {(h,): v for h, v in horizon_groups.items()}

    else:
        # group_by == "content" - Group by reward/time values
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
        return content_groups


def _generate_candidate_pairs(
    groups: dict[tuple, tuple[list[PreferenceSample], list[PreferenceSample]]],
    req: PrefPairRequirement,
    strat: PrefPairSubsampleStrategy,
) -> tuple[list[ContrastivePreferences], int, int]:
    """Generate candidate pairs within each group."""
    pairs: list[ContrastivePreferences] = []
    total_candidates = 0
    total_passed = 0

    for group_key, (group_short, group_long) in groups.items():
        if strat.best_only:
            # Only pair the best (highest confidence) short with best long
            sorted_short = sorted(
                group_short, key=lambda sample: sample.choice_prob, reverse=True
            )
            sorted_long = sorted(
                group_long, key=lambda sample: sample.choice_prob, reverse=True
            )
            if sorted_short and sorted_long:
                total_candidates += 1
                candidate = ContrastivePreferences(
                    short_term=sorted_short[0],
                    long_term=sorted_long[0],
                )
                if (
                    req.passes(candidate)
                    and candidate.min_choice_prob >= strat.min_confidence
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
                        and candidate.min_choice_prob >= strat.min_confidence
                    ):
                        total_passed += 1
                        pairs.append(candidate)

    return pairs, total_candidates, total_passed


def _apply_deduplication(
    pairs: list[ContrastivePreferences],
    strat: PrefPairSubsampleStrategy,
) -> list[ContrastivePreferences]:
    """Apply deduplication to remove duplicate content×horizon pairs."""
    if not strat.deduplicate or strat.best_only:
        return pairs

    seen: set[tuple] = set()
    unique_pairs: list[ContrastivePreferences] = []
    for p in pairs:
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
    return unique_pairs


def _apply_prefer_different_horizon(
    pairs: list[ContrastivePreferences],
    strat: PrefPairSubsampleStrategy,
) -> list[ContrastivePreferences]:
    """Sort pairs so different-horizon pairs come first."""
    if not strat.prefer_different_horizon:
        return pairs

    def horizon_priority(p: ContrastivePreferences) -> tuple:
        h_short = parse_horizon_years(p.short_term.time_horizon)
        h_long = parse_horizon_years(p.long_term.time_horizon)
        same_or_missing = (h_short == h_long) or (h_short is None) or (h_long is None)
        return (same_or_missing, -p.min_choice_prob)

    pairs.sort(key=horizon_priority)
    n_different = sum(
        1
        for p in pairs
        if parse_horizon_years(p.short_term.time_horizon)
        != parse_horizon_years(p.long_term.time_horizon)
        and parse_horizon_years(p.short_term.time_horizon) is not None
        and parse_horizon_years(p.long_term.time_horizon) is not None
    )
    log.info(
        f"Prefer different horizon: {n_different}/{len(pairs)} pairs have different horizons"
    )
    return pairs


def _apply_max_per_horizon_pair(
    pairs: list[ContrastivePreferences],
    strat: PrefPairSubsampleStrategy,
) -> list[ContrastivePreferences]:
    """Apply max_per_horizon_pair stratified selection."""
    if strat.max_per_horizon_pair is None or strat.max_per_horizon_pair <= 0:
        return pairs

    pairs.sort(key=lambda p: p.min_choice_prob, reverse=True)

    horizon_pair_counts: dict[tuple, int] = {}
    stratified_pairs: list[ContrastivePreferences] = []

    for p in pairs:
        h_short = parse_horizon_years(p.short_term.time_horizon)
        h_long = parse_horizon_years(p.long_term.time_horizon)
        key = (h_short, h_long)

        count = horizon_pair_counts.get(key, 0)
        if count < strat.max_per_horizon_pair:
            stratified_pairs.append(p)
            horizon_pair_counts[key] = count + 1

    log.info(
        f"Max per horizon pair ({strat.max_per_horizon_pair}): {len(pairs)} -> {len(stratified_pairs)} pairs "
        f"({len(horizon_pair_counts)} horizon combinations)"
    )
    return stratified_pairs


def _apply_max_per_reward_ratio(
    pairs: list[ContrastivePreferences],
    strat: PrefPairSubsampleStrategy,
) -> list[ContrastivePreferences]:
    """Apply max_per_reward_ratio stratified selection."""
    if strat.max_per_reward_ratio is None or strat.max_per_reward_ratio <= 0:
        return pairs

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
        if count < strat.max_per_reward_ratio:
            ratio_pairs.append(p)
            ratio_counts[ratio] = count + 1

    log.info(
        f"Max per reward ratio ({strat.max_per_reward_ratio}): {len(pairs)} -> {len(ratio_pairs)} pairs "
        f"({len(ratio_counts)} reward ratios)"
    )
    return ratio_pairs


def _apply_max_per_confidence_bucket(
    pairs: list[ContrastivePreferences],
    strat: PrefPairSubsampleStrategy,
) -> list[ContrastivePreferences]:
    """Apply max_per_confidence_bucket to ensure diversity across confidence levels."""
    if strat.max_per_confidence_bucket is None or strat.max_per_confidence_bucket <= 0:
        return pairs

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

    for p in pairs:
        bucket = get_confidence_bucket(p.min_choice_prob)
        count = bucket_counts.get(bucket, 0)
        if count < strat.max_per_confidence_bucket:
            bucket_pairs.append(p)
            bucket_counts[bucket] = count + 1

    log.info(
        f"Max per confidence bucket ({strat.max_per_confidence_bucket}): {len(pairs)} -> {len(bucket_pairs)} pairs "
        f"(buckets: {dict(sorted(bucket_counts.items()))})"
    )
    return bucket_pairs


def _apply_max_per_sample(
    pairs: list[ContrastivePreferences],
    max_per_sample: int | None,
) -> list[ContrastivePreferences]:
    """Apply max_per_sample to limit how many pairs each sample participates in."""
    if max_per_sample is None or max_per_sample <= 0:
        return pairs

    pairs.sort(key=lambda p: p.min_choice_prob, reverse=True)

    sample_usage: dict[int, int] = {}
    limited_pairs: list[ContrastivePreferences] = []

    for p in pairs:
        short_idx = p.short_term.sample_idx
        long_idx = p.long_term.sample_idx

        short_count = sample_usage.get(short_idx, 0)
        long_count = sample_usage.get(long_idx, 0)

        if short_count < max_per_sample and long_count < max_per_sample:
            limited_pairs.append(p)
            sample_usage[short_idx] = short_count + 1
            sample_usage[long_idx] = long_count + 1

    log.info(
        f"Max per sample ({max_per_sample}): {len(pairs)} -> {len(limited_pairs)} pairs"
    )
    return limited_pairs


def _apply_selection_strategy(
    pairs: list[ContrastivePreferences],
    strat: PrefPairSubsampleStrategy,
) -> list[ContrastivePreferences]:
    """Apply selection strategy for final ordering."""
    if strat.selection_strategy != "round_robin" or len(pairs) == 0:
        return pairs

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
    return round_robin_pairs
