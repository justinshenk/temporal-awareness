"""Pair categorization by label and order.

Creates JSON files categorizing pairs by:
- same_label.json / different_label.json: Whether pairs share label styles
- same_order.json / different_order.json: Whether pairs share option order
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...common.base_schema import BaseSchema
from ...common.file_io import save_json
from ..common.contrastive_preferences import ContrastivePreferences


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
