"""PrefPairSubsampleStrategy: strategy for subsampling contrastive preference pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...common.base_schema import BaseSchema

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

    # Per-dimension limits (applied in order: horizon -> ratio -> reward -> time -> confidence -> sample)
    max_per_sample: int | None = None
    """Maximum pairs each sample can participate in. Core reduction mechanism."""

    max_per_horizon_pair: int | None = None
    """Maximum pairs per (short_horizon, long_horizon) combination."""

    max_per_reward_ratio: int | None = None
    """Maximum pairs per reward ratio (long/short)."""

    max_per_short_reward: int | None = None
    """Maximum pairs per short_term_reward value. Ensures diversity in reward amounts."""

    max_per_short_time: int | None = None
    """Maximum pairs per short_term_time value. Ensures diversity in delivery times."""

    max_per_content: int | None = None
    """Maximum pairs per (short_reward, long_reward, short_time, long_time) combo."""

    max_per_confidence_bucket: int | None = None
    """Maximum pairs per confidence bucket ([0.5-0.6), [0.6-0.7), etc.)."""

    # Convenience presets
    smart_reduce: SmartReduceMode | None = None
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

    # Diversity across dimensions
    ensure_diversity: bool = False
    """Enable diversity across rewards/times by setting selection_strategy to round_robin.
    Note: For strong diversity guarantees, explicitly set max_per_short_reward and
    max_per_short_time based on your target_pairs and data dimensions.
    """

    # Selection strategy
    selection_strategy: SelectionStrategy = "greedy"
    """How to select pairs when applying limits:
    - "greedy": Take highest confidence pairs first (default)
    - "round_robin": Cycle through horizon combinations for diversity
    """

    skip_filtering: bool = False

    add_no_horizon: bool | int = False
    """Include pairs where both samples have no horizon.
    - False: don't add any
    - True: add all
    - int: add up to this many (that weren't already included)
    """

    @classmethod
    def Default(cls) -> "PrefPairSubsampleStrategy":
        """Return the default subsampling strategy."""
        strategy = cls()
        strategy.min_confidence = 0.8
        strategy.smart_reduce = "balanced"
        strategy.target_pairs = 100
        strategy.selection_strategy = "round_robin"
        strategy.ensure_diversity = True
        strategy.add_no_horizon = 10
        return strategy

    def apply_balanced_horizon_pairs(
        self, n_horizons: int
    ) -> "PrefPairSubsampleStrategy":
        """Calculate max_per_horizon_pair from target_pairs and horizon combos.

        With n_horizons values, there are n_horizons^2 possible (short_h, long_h) combos.
        But not all combos have pairs - estimate ~50% coverage.
        """
        if self.max_per_horizon_pair is not None:
            return self  # Don't override explicit setting
        if self.target_pairs is None or self.target_pairs <= 0:
            return self
        if n_horizons <= 0:
            return self

        # Assume ~50% of combos have pairs, add 2x buffer
        n_combos = n_horizons * n_horizons
        estimated_active_combos = max(1, n_combos // 2)
        max_per_h = max(2, int(self.target_pairs * 2 / estimated_active_combos) + 1)

        return PrefPairSubsampleStrategy(
            **{**self.to_dict(), "max_per_horizon_pair": max_per_h}
        )

    @classmethod
    def NoSubsample(cls) -> "PrefPairSubsampleStrategy":
        """Return a strategy with no subsampling - all valid pairs are kept."""
        return cls(
            min_confidence=0.0,
            max_per_sample=None,
            max_per_horizon_pair=None,
            max_per_reward_ratio=None,
            max_per_short_reward=None,
            max_per_short_time=None,
            max_per_content=None,
            max_per_confidence_bucket=None,
            smart_reduce=None,
            target_pairs=None,
            ensure_diversity=False,
            skip_filtering=True,
        )

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
            return PrefPairSubsampleStrategy(**{**self.to_dict(), "max_per_sample": 3})
        elif self.smart_reduce == "diverse":
            return PrefPairSubsampleStrategy(**{**self.to_dict(), "max_per_sample": 2})
        elif self.smart_reduce == "minimal":
            return PrefPairSubsampleStrategy(**{**self.to_dict(), "max_per_sample": 1})
        return self

    def apply_diversity_limits(
        self,
        n_rewards: int,
        n_times: int,
    ) -> "PrefPairSubsampleStrategy":
        """Apply diversity limits based on target_pairs and data dimensions.

        Uses round-robin selection which works better than hard limits for
        ensuring diversity. Only sets max_per_* limits if explicitly requested.

        Args:
            n_rewards: Number of unique short_term_reward values in the dataset.
            n_times: Number of unique short_term_time values in the dataset.

        Returns:
            New strategy with selection_strategy set to round_robin if ensure_diversity.
        """
        if not self.ensure_diversity:
            return self
        if self.target_pairs is None or self.target_pairs <= 0:
            return self

        # Use round_robin selection which interleaves across dimensions
        # This ensures diversity better than hard per-dimension limits
        # which cascade and over-filter
        if self.selection_strategy != "round_robin":
            return PrefPairSubsampleStrategy(
                **{**self.to_dict(), "selection_strategy": "round_robin"}
            )
        return self
