"""Horizon-based pair categorization.

Creates JSON files categorizing pairs by their horizon characteristics:
- horizon.json: Both pairs have horizon (buckets by comparison)
- half_horizon.json: Only one pair has horizon (clean vs corrupted)
- no_horizon.json: Neither has horizon (by choice made)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ....common.base_schema import BaseSchema
from ....common.file_io import save_json
from ...common.contrastive_preferences import ContrastivePreferences


@dataclass
class HorizonCategories(BaseSchema):
    """Categorization for pairs where both have horizon.

    Compares clean (short_term) horizon vs corrupted (long_term) horizon.
    """

    clean_greater: list[int] = field(default_factory=list)  # clean horizon > corrupted
    corrupted_greater: list[int] = field(
        default_factory=list
    )  # clean horizon < corrupted
    equal: list[int] = field(default_factory=list)  # clean horizon == corrupted

    @property
    def total(self) -> int:
        return len(self.clean_greater) + len(self.corrupted_greater) + len(self.equal)


@dataclass
class HalfHorizonCategories(BaseSchema):
    """Categorization for pairs where only one has horizon."""

    clean_has_horizon: list[int] = field(default_factory=list)  # short_term has horizon
    corrupted_has_horizon: list[int] = field(
        default_factory=list
    )  # long_term has horizon

    @property
    def total(self) -> int:
        return len(self.clean_has_horizon) + len(self.corrupted_has_horizon)


@dataclass
class NoHorizonCategories(BaseSchema):
    """Categorization for pairs where neither has horizon.

    Since neither has horizon info, categorize by the choice that was made.
    """

    # All pairs here have neither horizon, but we track what choices were made
    # Note: In ContrastivePreferences, short_term chose short, long_term chose long
    # So all pairs have both choices represented - we just list pair indices
    pair_indices: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.pair_indices)


@dataclass
class HorizonAnalysis(BaseSchema):
    """Complete horizon analysis for an experiment."""

    horizon: HorizonCategories = field(default_factory=HorizonCategories)
    half_horizon: HalfHorizonCategories = field(default_factory=HalfHorizonCategories)
    no_horizon: NoHorizonCategories = field(default_factory=NoHorizonCategories)

    @property
    def total(self) -> int:
        return self.horizon.total + self.half_horizon.total + self.no_horizon.total


def categorize_pair(
    pair_idx: int,
    pref: ContrastivePreferences,
    analysis: HorizonAnalysis,
) -> None:
    """Categorize a single pair into the appropriate horizon bucket."""
    short_horizon = pref.short_term.time_horizon  # clean trajectory
    long_horizon = pref.long_term.time_horizon  # corrupted trajectory

    if short_horizon is not None and long_horizon is not None:
        # Both have horizon - compare them (values are already in years as floats)
        if short_horizon > long_horizon:
            analysis.horizon.clean_greater.append(pair_idx)
        elif short_horizon < long_horizon:
            analysis.horizon.corrupted_greater.append(pair_idx)
        else:
            analysis.horizon.equal.append(pair_idx)

    elif short_horizon is not None:
        # Only clean (short_term) has horizon
        analysis.half_horizon.clean_has_horizon.append(pair_idx)

    elif long_horizon is not None:
        # Only corrupted (long_term) has horizon
        analysis.half_horizon.corrupted_has_horizon.append(pair_idx)

    else:
        # Neither has horizon
        analysis.no_horizon.pair_indices.append(pair_idx)


def build_horizon_analysis(
    pref_pairs: list[ContrastivePreferences],
) -> HorizonAnalysis:
    """Build complete horizon analysis from preference pairs."""
    analysis = HorizonAnalysis()

    for pair_idx, pref in enumerate(pref_pairs):
        categorize_pair(pair_idx, pref, analysis)

    return analysis


def save_horizon_analysis(
    analysis: HorizonAnalysis,
    output_dir: Path,
) -> None:
    """Save horizon analysis to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each category to its own file
    save_json(analysis.horizon.to_dict(), output_dir / "horizon.json")
    save_json(analysis.half_horizon.to_dict(), output_dir / "half_horizon.json")
    save_json(analysis.no_horizon.to_dict(), output_dir / "no_horizon.json")

    # Also save combined analysis
    save_json(analysis.to_dict(), output_dir / "horizon_analysis.json")

    print(f"[horizon] Saved horizon analysis to {output_dir}")
    print(f"  horizon: {analysis.horizon.total} pairs")
    print(f"    clean > corrupted: {len(analysis.horizon.clean_greater)}")
    print(f"    clean < corrupted: {len(analysis.horizon.corrupted_greater)}")
    print(f"    clean == corrupted: {len(analysis.horizon.equal)}")
    print(f"  half_horizon: {analysis.half_horizon.total} pairs")
    print(f"    clean has horizon: {len(analysis.half_horizon.clean_has_horizon)}")
    print(
        f"    corrupted has horizon: {len(analysis.half_horizon.corrupted_has_horizon)}"
    )
    print(f"  no_horizon: {analysis.no_horizon.total} pairs")


def _categorize_pair_from_summary(
    pair_idx: int,
    summary: dict,
    analysis: HorizonAnalysis,
) -> None:
    """Categorize a single pair from its cached summary dict.

    The summary dict comes from contrastive_preference.json and contains
    time_horizons: [short_term_horizon, long_term_horizon].
    """
    time_horizons = summary.get("time_horizons", [None, None])
    if len(time_horizons) < 2:
        time_horizons = [None, None]

    short_horizon = time_horizons[0]  # clean trajectory (short_term sample)
    long_horizon = time_horizons[1]  # corrupted trajectory (long_term sample)

    if short_horizon is not None and long_horizon is not None:
        # Both have horizon - compare them
        if short_horizon > long_horizon:
            analysis.horizon.clean_greater.append(pair_idx)
        elif short_horizon < long_horizon:
            analysis.horizon.corrupted_greater.append(pair_idx)
        else:
            analysis.horizon.equal.append(pair_idx)

    elif short_horizon is not None:
        # Only clean (short_term) has horizon
        analysis.half_horizon.clean_has_horizon.append(pair_idx)

    elif long_horizon is not None:
        # Only corrupted (long_term) has horizon
        analysis.half_horizon.corrupted_has_horizon.append(pair_idx)

    else:
        # Neither has horizon
        analysis.no_horizon.pair_indices.append(pair_idx)


def build_horizon_analysis_from_cache(pairs_dir: Path) -> HorizonAnalysis | None:
    """Build horizon analysis from cached contrastive_preference.json files.

    Args:
        pairs_dir: Directory containing pair_0/, pair_1/, etc. subdirectories

    Returns:
        HorizonAnalysis built from cached data, or None if no valid pairs found
    """
    pairs_dir = Path(pairs_dir)
    if not pairs_dir.exists():
        return None

    analysis = HorizonAnalysis()
    pair_idx = 0

    # Iterate through pair directories in order
    while True:
        pair_dir = pairs_dir / f"pair_{pair_idx}"
        pref_path = pair_dir / "contrastive_preference.json"

        if not pref_path.exists():
            break

        try:
            with open(pref_path) as f:
                summary = json.load(f)
            _categorize_pair_from_summary(pair_idx, summary, analysis)
        except (json.JSONDecodeError, OSError):
            # Skip corrupted files
            pass

        pair_idx += 1

    if analysis.total == 0:
        return None

    return analysis
