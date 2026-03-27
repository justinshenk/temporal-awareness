"""PrefPairRequirement: requirements for filtering ContrastivePreferences pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...common.base_schema import BaseSchema

if TYPE_CHECKING:
    from .contrastive_preferences import ContrastivePreferences


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

    # Largest reward choice requirements
    both_largest_reward: bool = False
    neither_largest_reward: bool = False
    only_short_largest_reward: bool = False
    only_long_largest_reward: bool = False
    only_one_largest_reward: bool = False

    # Trajectory length requirements
    same_length: bool = False
    different_length: bool = False

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
            ("same_length", "different_length"),
            ("both_rational", "neither_rational"),
            ("both_associated", "neither_associated"),
            ("both_largest_reward", "neither_largest_reward"),
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

        # Largest reward: mutually exclusive groups
        largest_reward_exclusive = [
            "both_largest_reward",
            "neither_largest_reward",
            "only_short_largest_reward",
            "only_long_largest_reward",
        ]
        active_largest_reward = [r for r in largest_reward_exclusive if getattr(self, r)]
        if len(active_largest_reward) > 1:
            errors.append(
                f"Largest reward requirements are mutually exclusive: {active_largest_reward}"
            )

        if errors:
            raise ValueError(f"Invalid PrefPairRequirement: {'; '.join(errors)}")

    def passes(self, pair: "ContrastivePreferences") -> bool:
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

        # Largest reward checks
        if self.both_largest_reward and not pair.both_largest_reward:
            return False
        if self.neither_largest_reward and not pair.neither_largest_reward:
            return False
        if self.only_short_largest_reward and not pair.only_short_largest_reward:
            return False
        if self.only_long_largest_reward and not pair.only_long_largest_reward:
            return False
        if self.only_one_largest_reward and not pair.only_one_largest_reward:
            return False

        # Length checks
        if self.same_length and not pair.same_length:
            return False
        if self.different_length and not pair.different_length:
            return False

        return True
