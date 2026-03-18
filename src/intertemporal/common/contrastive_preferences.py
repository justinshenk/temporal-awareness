"""ContrastivePreferences: pairs of samples with different time horizon choices."""

from __future__ import annotations

from dataclasses import dataclass

from ...common.base_schema import BaseSchema
from ...common.contrastive_pair import ContrastivePair
from ...binary_choice import BinaryChoiceRunner
from ...common.token_positions import build_position_mapping
from .preference_types import PreferenceSample


@dataclass
class ContrastivePreferences(BaseSchema):
    """A pair of PreferenceSamples that differ in time horizon and choice.

    Attributes:
        short_term: Sample that chose short_term
        long_term: Sample that chose long_term
    """

    short_term: PreferenceSample
    long_term: PreferenceSample

    def get_contrastive_pair(
        self,
        runner: BinaryChoiceRunner,
        anchor_texts: list[str] | None = None,
        first_interesting_marker: str | None = None,
    ) -> ContrastivePair | None:
        """Build a ContrastivePair from the two preference samples.

        Args:
            runner: BinaryChoiceRunner for tokenizer access
            anchor_texts: Text markers for position alignment (defaults to choice labels)
            first_interesting_marker: Marker for first interesting position

        Returns None if either sample fails verification.
        """
        if not self.long_term.verify() or not self.short_term.verify():
            return None

        short_traj = self.short_term.chosen_traj
        long_traj = self.long_term.chosen_traj
        assert short_traj is not None and long_traj is not None

        position_mapping = build_position_mapping(
            runner._tokenizer, short_traj, long_traj, anchor_texts
        )
        position_mapping.first_interesting_marker = first_interesting_marker

        return ContrastivePair(
            clean_traj=short_traj,
            corrupted_traj=long_traj,
            position_mapping=position_mapping,
            full_texts=(self.short_term.full_text, self.long_term.full_text),
            prompt_texts=(self.short_term.prompt_text, self.long_term.prompt_text),
            clean_labels=(
                self.short_term.short_term_label,
                self.short_term.long_term_label,
            ),
            corrupted_labels=(
                self.long_term.short_term_label,
                self.long_term.long_term_label,
            ),
            choice_prefix=self.short_term.choice_prefix,
            prompt_token_counts=(
                self.short_term.prompt_token_count,
                self.long_term.prompt_token_count,
            ),
            choice_divergent_positions=(
                self.short_term.divergent_position,
                self.long_term.divergent_position,
            ),
            time_horizons=(
                self.short_term.time_horizon,
                self.long_term.time_horizon,
            ),
        )

    # =========================================================================
    # Label/Formatting Properties
    # =========================================================================

    @property
    def same_labels(self) -> bool:
        """Check if both samples have the same label text."""
        return (
            self.short_term.short_term_label == self.long_term.short_term_label
            and self.short_term.long_term_label == self.long_term.long_term_label
        )

    @property
    def same_formatting(self) -> bool:
        """Check if both samples have the same formatting_id."""
        return self.short_term.formatting_id == self.long_term.formatting_id

    @property
    def same_context(self) -> bool:
        """Check if both samples have the same context_id."""
        return self.short_term.context_id == self.long_term.context_id

    # =========================================================================
    # Reward/Time Properties
    # =========================================================================

    @property
    def same_rewards(self) -> bool:
        """Check if both samples have the same reward values."""
        return (
            self.short_term.short_term_reward == self.long_term.short_term_reward
            and self.short_term.long_term_reward == self.long_term.long_term_reward
        )

    @property
    def same_times(self) -> bool:
        """Check if both samples have the same time values."""
        return (
            self.short_term.short_term_time == self.long_term.short_term_time
            and self.short_term.long_term_time == self.long_term.long_term_time
        )

    # =========================================================================
    # Horizon Properties
    # =========================================================================

    @property
    def same_horizon(self) -> bool:
        """Check if both samples have exactly the same time horizon."""
        if not self.both_horizon:
            return False
        return self.short_term.time_horizon == self.long_term.time_horizon

    @property
    def neither_horizon(self) -> bool:
        """Neither sample has a time horizon."""
        return (
            self.short_term.time_horizon is None and self.long_term.time_horizon is None
        )

    @property
    def both_horizon(self) -> bool:
        """Both samples have a time horizon."""
        return (
            self.short_term.time_horizon is not None
            and self.long_term.time_horizon is not None
        )

    @property
    def only_short_horizon(self) -> bool:
        """Only short_term sample has a time horizon."""
        return (
            self.short_term.time_horizon is not None
            and self.long_term.time_horizon is None
        )

    @property
    def only_long_horizon(self) -> bool:
        """Only long_term sample has a time horizon."""
        return (
            self.short_term.time_horizon is None
            and self.long_term.time_horizon is not None
        )

    @property
    def only_one_horizon(self) -> bool:
        """Exactly one sample has a time horizon."""
        return self.only_short_horizon or self.only_long_horizon

    # =========================================================================
    # Rational Choice Properties
    # =========================================================================

    @property
    def both_rational(self) -> bool:
        """Both samples match rational choice."""
        return (
            self.short_term.matches_rational is True
            and self.long_term.matches_rational is True
        )

    @property
    def neither_rational(self) -> bool:
        """Neither sample matches rational choice."""
        return (
            self.short_term.matches_rational is False
            and self.long_term.matches_rational is False
        )

    @property
    def only_short_rational(self) -> bool:
        """Only short_term sample matches rational choice."""
        return (
            self.short_term.matches_rational is True
            and self.long_term.matches_rational is False
        )

    @property
    def only_long_rational(self) -> bool:
        """Only long_term sample matches rational choice."""
        return (
            self.short_term.matches_rational is False
            and self.long_term.matches_rational is True
        )

    @property
    def only_one_rational(self) -> bool:
        """Exactly one sample matches rational choice."""
        return self.only_short_rational or self.only_long_rational

    # =========================================================================
    # Associated Choice Properties
    # =========================================================================

    @property
    def both_associated(self) -> bool:
        """Both samples match associated choice."""
        return (
            self.short_term.matches_associated is True
            and self.long_term.matches_associated is True
        )

    @property
    def neither_associated(self) -> bool:
        """Neither sample matches associated choice."""
        return (
            self.short_term.matches_associated is False
            and self.long_term.matches_associated is False
        )

    @property
    def only_short_associated(self) -> bool:
        """Only short_term sample matches associated choice."""
        return (
            self.short_term.matches_associated is True
            and self.long_term.matches_associated is False
        )

    @property
    def only_long_associated(self) -> bool:
        """Only long_term sample matches associated choice."""
        return (
            self.short_term.matches_associated is False
            and self.long_term.matches_associated is True
        )

    @property
    def only_one_associated(self) -> bool:
        """Exactly one sample matches associated choice."""
        return self.only_short_associated or self.only_long_associated

    # =========================================================================
    # Choice Probability Properties
    # =========================================================================

    @property
    def min_choice_prob(self) -> float:
        """Minimum choice probability across both samples."""
        return min(self.short_term.choice_prob, self.long_term.choice_prob)

    @property
    def mean_choice_prob(self) -> float:
        """Mean choice probability across both samples."""
        return (self.short_term.choice_prob + self.long_term.choice_prob) / 2
