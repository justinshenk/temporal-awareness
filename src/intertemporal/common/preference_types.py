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


@dataclass
class Response(BaseSchema):
    """Model response to a preference prompt."""

    chosen_option: str
    text: str = ""
    reasoning_trace: str = ""


# =============================================================================
# Dataset Sample Types
# =============================================================================


@dataclass
class PromptSample(BaseSchema):
    """A complete prompt sample."""

    sample_idx: int
    prompt: Prompt
    response: Optional[Response] = None


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

    # Extra Info
    internals: Any | None = None
    internals_paths: dict | None = None
    decoding_mismatch: bool | None = None

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

    def to_dict(
        self, max_list_length: int | None = None, max_string_length: int | None = None
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
        return d
