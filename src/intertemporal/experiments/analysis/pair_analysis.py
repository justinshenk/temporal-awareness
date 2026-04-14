"""Pair categorization by label and order.

Creates JSON files categorizing pairs by:
- same_label.json / different_label.json: Whether pairs share label styles
- same_order.json / different_order.json: Whether pairs share option order
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ....common.base_schema import BaseSchema
from ....common.file_io import save_json
from ...common.contrastive_preferences import ContrastivePreferences


@dataclass
class LabelCategories(BaseSchema):
    """Categorization for pairs by label consistency."""
    same_labels: list[int] = field(default_factory=list)
    different_labels: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.same_labels) + len(self.different_labels)


@dataclass
class OrderCategories(BaseSchema):
    """Categorization for pairs by option order consistency."""
    same_order: list[int] = field(default_factory=list)
    different_order: list[int] = field(default_factory=list)
    # More detailed breakdown
    both_short_first: list[int] = field(default_factory=list)
    both_long_first: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.same_order) + len(self.different_order)


@dataclass
class PairAnalysis(BaseSchema):
    """Complete pair analysis for an experiment."""
    labels: LabelCategories = field(default_factory=LabelCategories)
    order: OrderCategories = field(default_factory=OrderCategories)

    @property
    def total(self) -> int:
        return max(self.labels.total, self.order.total)


def categorize_pair(
    pair_idx: int,
    pref: ContrastivePreferences,
    analysis: PairAnalysis,
) -> None:
    """Categorize a single pair into label and order buckets."""
    # Label categorization
    if pref.same_labels:
        analysis.labels.same_labels.append(pair_idx)
    else:
        analysis.labels.different_labels.append(pair_idx)

    # Order categorization
    if pref.same_order:
        analysis.order.same_order.append(pair_idx)
        # Track detailed breakdown
        if pref.both_short_term_first:
            analysis.order.both_short_first.append(pair_idx)
        elif pref.both_long_term_first:
            analysis.order.both_long_first.append(pair_idx)
    elif pref.different_order:
        analysis.order.different_order.append(pair_idx)


def build_pair_analysis(
    pref_pairs: list[ContrastivePreferences],
) -> PairAnalysis:
    """Build complete pair analysis from preference pairs."""
    analysis = PairAnalysis()

    for pair_idx, pref in enumerate(pref_pairs):
        categorize_pair(pair_idx, pref, analysis)

    return analysis


def save_pair_analysis(
    analysis: PairAnalysis,
    output_dir: Path,
) -> None:
    """Save pair analysis to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save label categories (simple lists like horizon.json subcategories)
    save_json(analysis.labels.same_labels, output_dir / "same_label.json")
    save_json(analysis.labels.different_labels, output_dir / "different_label.json")

    # Save order categories
    save_json(analysis.order.same_order, output_dir / "same_order.json")
    save_json(analysis.order.different_order, output_dir / "different_order.json")

    # Also save combined analysis with all details
    save_json(analysis.to_dict(), output_dir / "pair_analysis.json")

    print(f"[pair] Saved pair analysis to {output_dir}")
    print(f"  labels: {analysis.labels.total} pairs")
    print(f"    same_labels: {len(analysis.labels.same_labels)}")
    print(f"    different_labels: {len(analysis.labels.different_labels)}")
    print(f"  order: {analysis.order.total} pairs")
    print(f"    same_order: {len(analysis.order.same_order)}")
    print(f"      both_short_first: {len(analysis.order.both_short_first)}")
    print(f"      both_long_first: {len(analysis.order.both_long_first)}")
    print(f"    different_order: {len(analysis.order.different_order)}")


def _categorize_pair_from_summary(
    pair_idx: int,
    summary: dict,
    analysis: PairAnalysis,
) -> None:
    """Categorize a single pair from its cached summary dict.

    The summary dict comes from contrastive_preference.json and contains
    pre-computed boolean fields like same_labels, same_order, etc.
    """
    # Label categorization
    if summary.get("same_labels", False):
        analysis.labels.same_labels.append(pair_idx)
    else:
        analysis.labels.different_labels.append(pair_idx)

    # Order categorization
    if summary.get("same_order", False):
        analysis.order.same_order.append(pair_idx)
        # Detailed breakdown: both_short_first / both_long_first
        short_term_first_list = summary.get("short_term_first", [])
        if len(short_term_first_list) == 2:
            if short_term_first_list[0] is True and short_term_first_list[1] is True:
                analysis.order.both_short_first.append(pair_idx)
            elif short_term_first_list[0] is False and short_term_first_list[1] is False:
                analysis.order.both_long_first.append(pair_idx)
    else:
        # Check if different_order (not same_order doesn't always mean different)
        short_term_first_list = summary.get("short_term_first", [])
        if len(short_term_first_list) == 2:
            if short_term_first_list[0] is not None and short_term_first_list[1] is not None:
                if short_term_first_list[0] != short_term_first_list[1]:
                    analysis.order.different_order.append(pair_idx)


def build_pair_analysis_from_cache(pairs_dir: Path) -> PairAnalysis | None:
    """Build pair analysis from cached contrastive_preference.json files.

    Args:
        pairs_dir: Directory containing pair_0/, pair_1/, etc. subdirectories

    Returns:
        PairAnalysis built from cached data, or None if no valid pairs found
    """
    pairs_dir = Path(pairs_dir)
    if not pairs_dir.exists():
        return None

    analysis = PairAnalysis()
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
