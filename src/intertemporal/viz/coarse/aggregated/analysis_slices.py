"""Analysis slices for aggregated visualization.

Defines named slices corresponding to PrefPairRequirement configurations.
Each slice filters pairs based on specific conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

from ....common.contrastive_utils import PrefPairRequirement


@dataclass
class AnalysisSlice:
    """A named analysis slice with filtering requirements."""

    name: str
    req: PrefPairRequirement
    description: str


# Core analysis slices - these are the main ones we visualize
ANALYSIS_SLICES: list[AnalysisSlice] = [
    # No filtering - all pairs
    AnalysisSlice(
        name="all",
        req=PrefPairRequirement(),
        description="All contrastive pairs",
    ),
    # Time horizon filtering - always generated alongside "all"
    AnalysisSlice(
        name="horizon",
        req=PrefPairRequirement(both_horizon=True),
        description="Both samples have time horizon",
    ),
    AnalysisSlice(
        name="no_horizon",
        req=PrefPairRequirement(neither_horizon=True),
        description="Neither sample has time horizon",
    ),
    AnalysisSlice(
        name="half_horizon",
        req=PrefPairRequirement(only_one_horizon=True),
        description="Only one sample has time horizon",
    ),
    # Label matching
    AnalysisSlice(
        name="same_labels",
        req=PrefPairRequirement(same_labels=True),
        description="Pairs with matching label formats",
    ),
    AnalysisSlice(
        name="different_labels",
        req=PrefPairRequirement(different_labels=True),
        description="Pairs with different label formats",
    ),
    # Context matching
    AnalysisSlice(
        name="same_context",
        req=PrefPairRequirement(same_context=True),
        description="Pairs with matching context",
    ),
    AnalysisSlice(
        name="different_context",
        req=PrefPairRequirement(different_context=True),
        description="Pairs with different context",
    ),
    # Formatting matching
    AnalysisSlice(
        name="same_formatting",
        req=PrefPairRequirement(same_formatting=True),
        description="Pairs with matching formatting",
    ),
    AnalysisSlice(
        name="different_formatting",
        req=PrefPairRequirement(different_formatting=True),
        description="Pairs with different formatting",
    ),
    # Combined: same_labels + same_context
    AnalysisSlice(
        name="same_labels-same_context",
        req=PrefPairRequirement(same_labels=True, same_context=True),
        description="Pairs with matching labels and context",
    ),
    # Combined: same_labels + same_formatting
    AnalysisSlice(
        name="same_labels-same_formatting",
        req=PrefPairRequirement(same_labels=True, same_formatting=True),
        description="Pairs with matching labels and formatting",
    ),
    # Combined: same_context + same_formatting
    AnalysisSlice(
        name="same_context-same_formatting",
        req=PrefPairRequirement(same_context=True, same_formatting=True),
        description="Pairs with matching context and formatting",
    ),
    # All matching
    AnalysisSlice(
        name="same_labels-same_context-same_formatting",
        req=PrefPairRequirement(
            same_labels=True, same_context=True, same_formatting=True
        ),
        description="Pairs with matching labels, context, and formatting",
    ),
]


def get_analysis_slice(name: str) -> AnalysisSlice | None:
    """Get an analysis slice by name."""
    for slice_ in ANALYSIS_SLICES:
        if slice_.name == name:
            return slice_
    return None


def get_analysis_slice_names() -> list[str]:
    """Get all analysis slice names."""
    return [s.name for s in ANALYSIS_SLICES]
