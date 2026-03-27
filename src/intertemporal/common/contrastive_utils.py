"""Utilities for creating and filtering contrastive preference pairs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ...common.profiler import profile

from ...common.time_value import parse_horizon_years
from .contrastive_preferences import ContrastivePreferences
from .pref_pair_requirement import PrefPairRequirement
from .pref_pair_subsample import GroupByMode, PrefPairSubsampleStrategy
from .preference_types import PreferenceSample

if TYPE_CHECKING:
    from ..preference import PreferenceDataset

log = logging.getLogger(__name__)

SAMPLES_THRESH = 4


def get_contrastive_preferences(
    dataset: "PreferenceDataset",
    req: PrefPairRequirement | None = None,
    strat: PrefPairSubsampleStrategy | None = None,
) -> list[ContrastivePreferences]:
    """Find pairs of samples with different choices for contrastive analysis.

    Args:
        dataset: PreferenceDataset containing samples to search
        req: Optional PrefPairRequirement specifying filtering requirements
        subsample: Optional PrefPairSubsampleStrategy for reduction settings.

    Returns:
        List of ContrastivePreferences pairs sorted by confidence
    """

    if req is None:
        req = PrefPairRequirement.Default()
    req.verify()

    # Collect samples by choice
    short_choosers, long_choosers = _collect_samples_by_choice(dataset)
    n_samples = len(short_choosers) + len(long_choosers)

    # Build strategy from subsample
    if n_samples <= SAMPLES_THRESH:
        strat = PrefPairSubsampleStrategy.NoSubsample()
    if strat is None:
        strat = PrefPairSubsampleStrategy.Default()

    # Build groups based on mode
    if strat.skip_filtering:
        groups = _build_groups(short_choosers, long_choosers)
        pairs, _, _ = _generate_candidate_pairs(req, groups, strat)
        return pairs

    # Apply smart_reduce preset (only if n_samples > 5)
    strat = strat.apply_smart_reduce(n_samples)

    # Apply diversity limits based on data dimensions
    all_samples = short_choosers + long_choosers
    n_rewards = len({s.short_term_reward for s in all_samples if s.short_term_reward})
    n_times = len({s.short_term_time for s in all_samples if s.short_term_time})
    n_horizons = len({parse_horizon_years(s.time_horizon) for s in all_samples})
    strat = strat.apply_diversity_limits(n_rewards, n_times)
    strat = strat.apply_balanced_horizon_pairs(n_horizons)

    # Calculate max_per_sample from target_pairs if needed
    max_per_sample = _calculate_max_per_sample(strat, short_choosers, long_choosers)

    groups = _build_groups(short_choosers, long_choosers, strat.group_by)
    pairs, total_candidates, total_passed = _generate_candidate_pairs(
        req, groups, strat
    )

    # Extract no-horizon pairs before filtering (if add_no_horizon is set)
    no_horizon_pairs: list[ContrastivePreferences] = []
    if strat.add_no_horizon:
        no_horizon_pairs = [
            p for p in pairs
            if p.short_term.time_horizon is None and p.long_term.time_horizon is None
        ]

    # Apply filters in order
    # 1. First: reorder for diversity (round_robin cycles through horizon×content groups)
    pairs = _apply_selection_strategy(pairs, strat)
    # 2. Then: apply per-dimension limits on the diverse-ordered list
    pairs = _apply_deduplication(pairs, strat)
    pairs = _apply_prefer_different_horizon(pairs, strat)
    pairs = _apply_max_per_horizon_pair(pairs, strat)
    pairs = _apply_max_per_reward_ratio(pairs, strat)
    pairs = _apply_max_per_short_reward(pairs, strat)
    pairs = _apply_max_per_short_time(pairs, strat)
    pairs = _apply_max_per_content(pairs, strat)
    pairs = _apply_max_per_confidence_bucket(pairs, strat)
    pairs = _apply_max_per_sample(pairs, max_per_sample)
    # 3. Finally: truncate to target
    pairs = _apply_target_pairs(pairs, strat)

    # Add back no-horizon pairs that weren't already included
    if strat.add_no_horizon and no_horizon_pairs:
        existing_pair_keys = {
            (p.short_term.sample_idx, p.long_term.sample_idx) for p in pairs
        }
        # Determine max to add: True means unlimited, int means that many
        max_to_add = len(no_horizon_pairs) if strat.add_no_horizon is True else int(strat.add_no_horizon)
        added = 0
        for p in no_horizon_pairs:
            if added >= max_to_add:
                break
            key = (p.short_term.sample_idx, p.long_term.sample_idx)
            if key not in existing_pair_keys:
                pairs.append(p)
                existing_pair_keys.add(key)
                added += 1
        if added > 0:
            print(f"[add_no_horizon] Added {added} no-horizon pairs")

    print(
        f"[contrastive] {len(short_choosers)} short, {len(long_choosers)} long -> "
        f"{total_candidates} candidates -> {total_passed} passed -> {len(pairs)} final"
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


@profile
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


@profile
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
                print(
                    f"[target_pairs={strat.target_pairs}] -> max_per_sample={max_per_sample} "
                    f"(n_short={n_short}, n_long={n_long})"
                )

    return max_per_sample


@profile
def _build_groups(
    short_choosers: list[PreferenceSample],
    long_choosers: list[PreferenceSample],
    group_mode: GroupByMode = "choice",
) -> dict[tuple, tuple[list[PreferenceSample], list[PreferenceSample]]]:
    """Build groups of samples based on grouping mode."""
    if group_mode == "choice":
        # No grouping - all samples in one group
        return {(): (short_choosers, long_choosers)}

    elif group_mode == "horizon":
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


@profile
def _generate_candidate_pairs(
    req: PrefPairRequirement,
    groups: dict[tuple, tuple[list[PreferenceSample], list[PreferenceSample]]],
    strat: PrefPairSubsampleStrategy | None = None,
) -> tuple[list[ContrastivePreferences], int, int]:
    """Generate candidate pairs within each group."""
    pairs: list[ContrastivePreferences] = []
    total_candidates = 0
    total_passed = 0

    for group_key, (group_short, group_long) in groups.items():
        if strat is not None and strat.best_only:
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


@profile
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


@profile
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


@profile
def _apply_max_per_horizon_pair(
    pairs: list[ContrastivePreferences],
    strat: PrefPairSubsampleStrategy,
) -> list[ContrastivePreferences]:
    """Apply max_per_horizon_pair stratified selection."""
    if strat.max_per_horizon_pair is None or strat.max_per_horizon_pair <= 0:
        return pairs

    # Don't re-sort - preserve round_robin diversity ordering

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


@profile
def _apply_max_per_reward_ratio(
    pairs: list[ContrastivePreferences],
    strat: PrefPairSubsampleStrategy,
) -> list[ContrastivePreferences]:
    """Apply max_per_reward_ratio stratified selection."""
    if strat.max_per_reward_ratio is None or strat.max_per_reward_ratio <= 0:
        return pairs

    # Don't re-sort - preserve round_robin diversity ordering

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


@profile
def _apply_max_per_short_reward(
    pairs: list[ContrastivePreferences],
    strat: PrefPairSubsampleStrategy,
) -> list[ContrastivePreferences]:
    """Apply max_per_short_reward to ensure diversity in reward amounts."""
    if strat.max_per_short_reward is None or strat.max_per_short_reward <= 0:
        return pairs

    # Don't re-sort - preserve round_robin diversity ordering

    reward_counts: dict[float | None, int] = {}
    result: list[ContrastivePreferences] = []

    for p in pairs:
        reward = p.short_term.short_term_reward
        count = reward_counts.get(reward, 0)
        if count < strat.max_per_short_reward:
            result.append(p)
            reward_counts[reward] = count + 1

    print(
        f"[max_per_short_reward={strat.max_per_short_reward}] {len(pairs)} -> {len(result)} pairs "
        f"({len(reward_counts)} rewards)"
    )
    return result


@profile
def _apply_max_per_short_time(
    pairs: list[ContrastivePreferences],
    strat: PrefPairSubsampleStrategy,
) -> list[ContrastivePreferences]:
    """Apply max_per_short_time to ensure diversity in delivery times."""
    if strat.max_per_short_time is None or strat.max_per_short_time <= 0:
        return pairs

    # Don't re-sort - preserve round_robin diversity ordering

    time_counts: dict[float | None, int] = {}
    result: list[ContrastivePreferences] = []

    for p in pairs:
        time = p.short_term.short_term_time
        count = time_counts.get(time, 0)
        if count < strat.max_per_short_time:
            result.append(p)
            time_counts[time] = count + 1

    print(
        f"[max_per_short_time={strat.max_per_short_time}] {len(pairs)} -> {len(result)} pairs "
        f"({len(time_counts)} times)"
    )
    return result


@profile
def _apply_max_per_content(
    pairs: list[ContrastivePreferences],
    strat: PrefPairSubsampleStrategy,
) -> list[ContrastivePreferences]:
    """Apply max_per_content to ensure diversity across all content dimensions."""
    if strat.max_per_content is None or strat.max_per_content <= 0:
        return pairs

    # Don't re-sort - preserve round_robin diversity ordering

    content_counts: dict[tuple, int] = {}
    result: list[ContrastivePreferences] = []

    for p in pairs:
        # Use short_term sample's content (since we're pairing different choices)
        key = (
            p.short_term.short_term_reward,
            p.short_term.long_term_reward,
            p.short_term.short_term_time,
            p.short_term.long_term_time,
        )
        count = content_counts.get(key, 0)
        if count < strat.max_per_content:
            result.append(p)
            content_counts[key] = count + 1

    print(
        f"[max_per_content={strat.max_per_content}] {len(pairs)} -> {len(result)} pairs "
        f"({len(content_counts)} content combos)"
    )
    return result


@profile
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


@profile
def _apply_max_per_sample(
    pairs: list[ContrastivePreferences],
    max_per_sample: int | None,
) -> list[ContrastivePreferences]:
    """Apply max_per_sample to limit how many pairs each sample participates in."""
    if max_per_sample is None or max_per_sample <= 0:
        return pairs

    # Don't re-sort - preserve round_robin diversity ordering

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

    print(
        f"[max_per_sample={max_per_sample}] {len(pairs)} -> {len(limited_pairs)} pairs"
    )
    return limited_pairs


@profile
def _apply_selection_strategy(
    pairs: list[ContrastivePreferences],
    strat: PrefPairSubsampleStrategy,
) -> list[ContrastivePreferences]:
    """Apply selection strategy for final ordering.

    Round-robin selection cycles through multiple dimensions to ensure diversity:
    - Horizon combinations (short_horizon, long_horizon)
    - Content combinations (short_reward, long_reward, short_time, long_time)

    This ensures the final pairs have diversity across all dimensions.
    """
    if strat.selection_strategy != "round_robin" or len(pairs) == 0:
        return pairs

    # Group pairs by (horizon_combo, content_combo) for maximum diversity
    # This ensures we cycle through both horizons AND content
    diversity_groups: dict[tuple, list[ContrastivePreferences]] = {}
    for p in pairs:
        h_short = parse_horizon_years(p.short_term.time_horizon)
        h_long = parse_horizon_years(p.long_term.time_horizon)
        # Use short_term sample's content (reward/time combination)
        content_key = (
            p.short_term.short_term_reward,
            p.short_term.long_term_reward,
            p.short_term.short_term_time,
            p.short_term.long_term_time,
        )
        key = (h_short, h_long, content_key)
        if key not in diversity_groups:
            diversity_groups[key] = []
        diversity_groups[key].append(p)

    # Sort each group by confidence
    for group_pairs in diversity_groups.values():
        group_pairs.sort(key=lambda p: p.min_choice_prob, reverse=True)

    # Round-robin selection across groups
    round_robin_pairs: list[ContrastivePreferences] = []
    group_keys = list(diversity_groups.keys())
    indices = {k: 0 for k in group_keys}

    while len(round_robin_pairs) < len(pairs):
        added_any = False
        for key in group_keys:
            idx = indices[key]
            if idx < len(diversity_groups[key]):
                round_robin_pairs.append(diversity_groups[key][idx])
                indices[key] = idx + 1
                added_any = True
        if not added_any:
            break

    # Count unique dimensions for logging
    unique_horizons = len({(k[0], k[1]) for k in group_keys})
    unique_contents = len({k[2] for k in group_keys})
    print(
        f"[round_robin] {len(diversity_groups)} groups "
        f"({unique_horizons} horizon combos × {unique_contents} content combos)"
    )
    return round_robin_pairs


@profile
def _apply_target_pairs(
    pairs: list[ContrastivePreferences],
    strat: PrefPairSubsampleStrategy,
) -> list[ContrastivePreferences]:
    """Truncate to target_pairs if specified."""
    if strat.target_pairs is None or strat.target_pairs <= 0:
        return pairs
    if len(pairs) <= strat.target_pairs:
        return pairs

    # Already sorted by confidence, just truncate
    truncated = pairs[: strat.target_pairs]
    print(f"[target_pairs={strat.target_pairs}] {len(pairs)} -> {len(truncated)} pairs")
    return truncated
