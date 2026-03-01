"""ContrastivePreferences: pairs of samples with different time horizon choices."""

from __future__ import annotations

from dataclasses import dataclass

from ...common.base_schema import BaseSchema
from ...common.contrastive_pair import ContrastivePair
from ...binary_choice import BinaryChoiceRunner
from ...inference import GeneratedTrajectory
from ...common.token_positions import build_position_mapping
from ..preference import PreferenceDataset
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
        names_filter: callable | None = None,
        anchor_texts: list[str] | None = None,
        first_interesting_marker: str | None = None,
    ) -> ContrastivePair | None:
        """Build a ContrastivePair using cached activations when available.

        Args:
            runner: BinaryChoiceRunner for inference
            names_filter: Filter for which activations to capture
            anchor_texts: Text markers for position alignment (defaults to choice labels)

        Returns None if either sample fails verification.
        """
        if not self.long_term.verify() or not self.short_term.verify():
            return None

        # Require same formatting - swapped labels cause semantic mismatch in patching
        if not self.same_formatting:
            print("  Skipping pair: different label formatting")
            return None

        long_traj = self._get_trajectory(self.long_term, runner, names_filter)
        short_traj = self._get_trajectory(self.short_term, runner, names_filter)

        position_mapping = build_position_mapping(
            runner._tokenizer, short_traj, long_traj, anchor_texts
        )
        position_mapping.first_interesting_marker = first_interesting_marker

        return ContrastivePair(
            short_traj=short_traj,
            long_traj=long_traj,
            position_mapping=position_mapping,
            full_texts=(self.short_term.full_text, self.long_term.full_text),
            labels=(
                self.short_term.short_term_label,
                self.short_term.long_term_label,
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
        )

    def _get_trajectory(
        self,
        sample: PreferenceSample,
        runner: BinaryChoiceRunner,
        names_filter: callable | None,
    ) -> GeneratedTrajectory:
        """Get trajectory with internals for a sample.

        Attempts in order:
        1. Existing trajectory with required internals
        2. Load internals from disk
        3. Run model forward pass
        """
        traj = sample.chosen_traj
        assert traj is not None

        if traj.has_internals_for(names_filter):
            return traj

        # Try loading from disk
        sample.load_internals_from_disk()
        if traj.has_internals_for(names_filter):
            return traj

        # Run forward pass
        return runner.generate_trajectory_with_cache(traj.token_ids, names_filter)

    @property
    def same_formatting(self) -> bool:
        """Check if both samples use the same label formatting."""
        return (
            self.short_term.short_term_label == self.long_term.short_term_label
            and self.short_term.long_term_label == self.long_term.long_term_label
        )

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

    @property
    def min_choice_prob(self) -> float:
        """Minimum choice probability across both samples.

        Higher values indicate both samples were confident in their choices,
        making this a better pair for activation patching.
        """
        return min(self.short_term.choice_prob, self.long_term.choice_prob)

    @property
    def mean_choice_prob(self) -> float:
        """Mean choice probability across both samples."""
        return (self.short_term.choice_prob + self.long_term.choice_prob) / 2


def get_contrastive_preferences(
    dataset: PreferenceDataset,
    require_same_labels: bool = True,
    debug_by_using_single_sample: bool = False,
) -> list[ContrastivePreferences]:
    """Find pairs of samples that differ primarily by time_horizon with different choices.

    This function groups samples by their content (formatting_id and reward/time values)
    and finds pairs where:
    - One sample chose short_term and one chose long_term
    - The primary difference is the time_horizon (which affects rational choice)

    Args:
        dataset: PreferenceDataset containing samples to search

    Returns:
        List of ContrastivePreferences pairs
    """
    # Group samples by content (same formatting, rewards, times, but potentially different time_horizon)
    # Key: (abs(formatting_id), short_reward, long_reward, short_time, long_time)
    # Use abs(formatting_id) because label swaps (a↔b) result in negated IDs
    content_groups: dict[tuple, list[PreferenceSample]] = {}

    for pref in dataset.preferences:
        # Skip samples with unknown choice
        if pref.choice_term not in ("short_term", "long_term"):
            continue

        # Build content key - use abs(formatting_id) to group label-swapped variants
        key = (
            abs(pref.formatting_id) if pref.formatting_id else 0,
            pref.short_term_reward,
            pref.long_term_reward,
            pref.short_term_time,
            pref.long_term_time,
        )
        if key not in content_groups:
            content_groups[key] = []
        content_groups[key].append(pref)

    # Find pairs with different choices within each group
    pairs: list[ContrastivePreferences] = []

    for key, samples in content_groups.items():
        # Separate by choice
        short_choosers = [s for s in samples if s.choice_term == "short_term"]
        long_choosers = [s for s in samples if s.choice_term == "long_term"]

        # Create pairs
        for short_sample in short_choosers:
            for long_sample in long_choosers:
                # Verify they have different time horizons
                if short_sample.time_horizon == long_sample.time_horizon:
                    continue

                # Verify same label formatting (swapped labels cause semantic mismatch)
                if require_same_labels and (
                    short_sample.short_term_label != long_sample.short_term_label
                    or short_sample.long_term_label != long_sample.long_term_label
                ):
                    continue

                pairs.append(
                    ContrastivePreferences(
                        short_term=short_sample,
                        long_term=long_sample,
                    )
                )

    # Sort by minimum choice probability (highest confidence pairs first)
    # This prioritizes pairs where both samples were confident in their choices
    pairs.sort(key=lambda p: p.min_choice_prob, reverse=True)

    if debug_by_using_single_sample:
        return [
            ContrastivePreferences(
                short_term=pairs[0].short_term,
                long_term=pairs[0].short_term,
            )
        ]

    return pairs
