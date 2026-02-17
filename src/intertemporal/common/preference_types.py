"""Intertemporal preference type definitions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from ...common.base_schema import BaseSchema
from ...common.time_value import TimeValue


@dataclass
class RewardValue(BaseSchema):
    """A reward value with unit."""

    value: float
    unit: str = ""

    def __str__(self) -> str:
        if self.unit:
            return f"{self.value:,.0f} {self.unit}"
        return f"{self.value:,.0f}"


# =============================================================================
# Intertemporal Options
# =============================================================================


@dataclass
class IntertemporalOption(BaseSchema):
    """An intertemporal option: (time, reward)."""

    label: str
    time: TimeValue
    reward: RewardValue


@dataclass
class PreferencePair(BaseSchema):
    """A pair of intertemporal options for comparison."""

    short_term: IntertemporalOption
    long_term: IntertemporalOption


# =============================================================================
# Prompt & Response
# =============================================================================


@dataclass
class Prompt(BaseSchema):
    """Input prompt for preference elicitation."""

    preference_pair: PreferencePair
    time_horizon: Optional[TimeValue] = None
    text: str = ""

    @property
    def expected_rational_choice(self) -> int | None:
        """Return the rational choice given the time horizon.

        Returns:
            0 (short term) if time_horizon < long_term time (won't be around for long term)
            1 (long term) if time_horizon > long_term time (will be around for long term)
            None if no time_horizon or ambiguous
        """
        if not self.time_horizon:
            return None
        horizon_years = self.time_horizon.to_years()
        long_term_years = self.preference_pair.long_term.time.to_years()

        if horizon_years < long_term_years:
            return 0  # Short term is rational
        if horizon_years > long_term_years:
            return 1  # Long term is rational
        return None  # Ambiguous

    @property
    def associated_choice(self) -> int | None:
        """Return the choice whose time is closest to the time horizon.

        Returns:
            0 if short_term time is closer to time_horizon
            1 if long_term time is closer to time_horizon
            None if no time_horizon
        """
        if not self.time_horizon:
            return None
        horizon_years = self.time_horizon.to_years()
        short_years = self.preference_pair.short_term.time.to_years()
        long_years = self.preference_pair.long_term.time.to_years()

        short_dist = abs(horizon_years - short_years)
        long_dist = abs(horizon_years - long_years)

        if short_dist < long_dist:
            return 0
        if long_dist < short_dist:
            return 1
        return None  # Equidistant


# =============================================================================
# Dataset Sample Types
# =============================================================================


@dataclass
class PromptSample(BaseSchema):
    """A complete prompt sample."""

    sample_idx: int
    prompt: Prompt

    formatting_id: int | None = None

    @property
    def expected_rational_choice(self) -> int | None:
        return self.prompt.expected_rational_choice

    @property
    def associated_choice(self) -> int | None:
        return self.prompt.associated_choice


@dataclass
class PreferenceSample(BaseSchema):
    """Single preference result."""

    sample_idx: int
    choice: Any  # BinaryChoice type

    # Sample Info
    time_horizon: Optional[dict] = None
    response_text: str | None = None
    prompt_text: str | None = None

    # Choice Extra Info
    short_term_label: str | None = None
    long_term_label: str | None = None
    short_term_reward: float | None = None
    long_term_reward: float | None = None
    short_term_time: float | None = None
    long_term_time: float | None = None
    choice_prefix: str | None = None  # e.g. "I select: "

    # Extra Info
    internals: Any | None = None
    internals_paths: dict | None = None
    decoding_mismatch: bool | None = None

    formatting_id: int | None = None
    matches_rational: bool | None = None
    matches_associated: bool | None = None

    @property
    def choice_idx(self) -> int:
        """Index of the chosen option (delegates to choice object)."""
        return self.choice.choice_idx

    @property
    def choice_label(self) -> str | None:
        """Label of the chosen option (delegates to choice object)."""
        if hasattr(self.choice, "chosen_label"):
            return self.choice.chosen_label
        return None

    @property
    def alternative_idx(self) -> int:
        """Index of the alternative option (delegates to choice object)."""
        return self.choice.alternative_idx

    @property
    def alternative_label(self) -> str | None:
        """Label of the alternative option (delegates to choice object)."""
        if hasattr(self.choice, "alternative_label"):
            return self.choice.alternative_label
        return None

    @property
    def choice_prob(self) -> float:
        logprob = self.choice.choice_logprob
        if logprob is None:
            return 0.0
        return math.exp(logprob)

    @property
    def alternative_prob(self) -> float:
        logprob = self.choice.alternative_logprob
        if logprob is None:
            return 0.0
        return math.exp(logprob)

    @property
    def choice_term(self) -> str | None:
        """Which term was chosen: 'short_term' or 'long_term'."""
        label = self.choice_label
        if label is None:
            return None
        if label == self.short_term_label:
            return "short_term"
        if label == self.long_term_label:
            return "long_term"
        return None

    @property
    def alternative_term(self) -> str | None:
        """Which term was not chosen: 'short_term' or 'long_term'."""
        label = self.alternative_label
        if label is None:
            return None
        if label == self.short_term_label:
            return "short_term"
        if label == self.long_term_label:
            return "long_term"
        return None

    @property
    def chose_short_term(self) -> bool:
        term = self.choice_term
        if not term:
            return False
        if term == "short_term":
            return True
        return False

    @property
    def chose_long_term(self) -> bool:
        term = self.choice_term
        if not term:
            return False
        if term == "long_term":
            return True
        return False

    @property
    def full_text(self) -> str:
        """Full text: prompt + response."""
        return (self.prompt_text or "") + (self.response_text or "")

    @property
    def chosen_traj(self):
        """Get the chosen trajectory from the choice."""
        if self.choice is None:
            return None
        if hasattr(self.choice, "chosen_traj"):
            return self.choice.chosen_traj
        return None

    @property
    def alternative_traj(self):
        """Get the alternative trajectory from the choice."""
        if self.choice is None:
            return None
        if hasattr(self.choice, "alternative_traj"):
            return self.choice.alternative_traj
        return None

    @property
    def prompt_token_count(self) -> int | None:
        """Get the number of prompt tokens (trunk length) from the choice tree.

        This is the position where the model's response begins.
        Returns None if choice or tree is not available.
        """
        if self.choice is None:
            return None
        if hasattr(self.choice, "tree") and self.choice.tree is not None:
            return self.choice.tree.trunk_length
        return None

    def load_internals_from_disk(self) -> None:
        """Load internals from disk into the trajectory."""
        if self.choice is not None and hasattr(self.choice, "load_internals_from_disk"):
            self.choice.load_internals_from_disk(self.internals_paths)

    def verify(self) -> bool:
        """Check if sample has all required fields for patching experiments."""
        return (
            self.short_term_label is not None
            and self.long_term_label is not None
            and self.prompt_text is not None
            and self.response_text is not None
            and self.choice is not None
        )

    def pop_heavy(self) -> None:
        """Remove heavy data (internals, full_logits) to reduce memory."""
        self.internals = None
        if self.choice:
            self.choice.pop_heavy()

    def to_dict(
        self,
        max_list_length: int | None = None,
        max_string_length: int | None = None,
        without_tree: bool = False,
    ):
        d = super().to_dict(
            max_list_length=max_list_length, max_string_length=max_string_length
        )
        d["choice_idx"] = self.choice_idx
        d["choice_label"] = self.choice_label
        d["choice_prob"] = self.choice_prob
        d["choice_term"] = self.choice_term
        d["alternative_idx"] = self.alternative_idx
        d["alternative_label"] = self.alternative_label
        d["alternative_prob"] = self.alternative_prob
        d["alternative_term"] = self.alternative_term

        d["chose_short_term"] = self.chose_short_term
        d["chose_long_term"] = self.chose_long_term

        if without_tree:
            d["choice"].pop("tree", None)
        return d
