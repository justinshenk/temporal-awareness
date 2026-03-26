"""PrefPairSubsampleStrategy: strategy for subsampling contrastive preference pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...common.base_schema import BaseSchema

__all__ = [
    "PrefPairSubsampleStrategy",
    "GroupByMode",
    "SmartReduceMode",
    "SelectionStrategy",
]

GroupByMode = Literal["content", "horizon", "choice"]
SmartReduceMode = Literal["balanced", "diverse", "minimal"]
SelectionStrategy = Literal["greedy", "round_robin"]


@dataclass
class PrefPairSubsampleStrategy(BaseSchema):
    """Strategy for subsampling/reducing contrastive preference pairs.

    Controls how pairs are grouped, deduplicated, and reduced to a manageable size.
    All fields have sensible defaults for the common case (group_by="choice").
    """

    # Grouping mode
    group_by: GroupByMode = "choice"
    """How to group samples before pairing:
    - "choice": No grouping - pair any short-chooser with any long-chooser (default)
    - "horizon": Group by horizon value - pairs share same horizon
    - "content": Group by reward/time values - pairs share same content
    """

    # Deduplication
    deduplicate: bool = False
    """Remove duplicate content×horizon pairs within each group."""

    best_only: bool = False
    """Keep only the single best pair per group (highest confidence)."""

    # Confidence filtering
    min_confidence: float = 0.0
    """Minimum choice probability threshold (0.0-1.0)."""

    # Per-dimension limits (applied in order: horizon -> ratio -> confidence -> sample)
    max_per_sample: int | None = None
    """Maximum pairs each sample can participate in. Core reduction mechanism."""

    max_per_horizon_pair: int | None = None
    """Maximum pairs per (short_horizon, long_horizon) combination."""

    max_per_reward_ratio: int | None = None
    """Maximum pairs per reward ratio (long/short)."""

    max_per_confidence_bucket: int | None = None
    """Maximum pairs per confidence bucket ([0.5-0.6), [0.6-0.7), etc.)."""

    # Convenience presets
    smart_reduce: SmartReduceMode | None = "minimal"
    """Preset that sets max_per_sample:
    - "minimal": max_per_sample=1 (~25 pairs) [DEFAULT]
    - "diverse": max_per_sample=2 (~50 pairs)
    - "balanced": max_per_sample=3 (~75 pairs)
    Only applied when n_samples > 5.
    """

    # Prioritization
    prefer_different_horizon: bool = False
    """Sort different-horizon pairs first before applying limits."""

    # Target-based reduction
    target_pairs: int | None = None
    """Target number of output pairs. Auto-calculates max_per_sample."""

    # Selection strategy
    selection_strategy: SelectionStrategy = "greedy"
    """How to select pairs when applying limits:
    - "greedy": Take highest confidence pairs first (default)
    - "round_robin": Cycle through horizon combinations for diversity
    """

    def apply_smart_reduce(self, n_samples: int) -> "PrefPairSubsampleStrategy":
        """Apply smart_reduce preset to max_per_sample if not already set.

        Args:
            n_samples: Total number of samples. Smart reduce only applied if > 5.

        Returns:
            New strategy with max_per_sample set, or self if not applicable.
        """
        if self.max_per_sample is not None:
            return self  # Don't override explicit setting

        # Only apply smart_reduce for larger datasets
        if n_samples <= 5:
            return self

        if self.smart_reduce == "balanced":
            return PrefPairSubsampleStrategy(
                **{**self.to_dict(), "max_per_sample": 3}
            )
        elif self.smart_reduce == "diverse":
            return PrefPairSubsampleStrategy(
                **{**self.to_dict(), "max_per_sample": 2}
            )
        elif self.smart_reduce == "minimal":
            return PrefPairSubsampleStrategy(
                **{**self.to_dict(), "max_per_sample": 1}
            )
        return self
