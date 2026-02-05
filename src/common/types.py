"""
Internal type definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schema_utils import SchemaClass


# =============================================================================
# Base Value Types
# =============================================================================


@dataclass
class TimeValue(SchemaClass):
    """
    A time value with unit: t_i

    Attributes:
        value: Numeric time value
        unit: Time unit (e.g., "months", "years")
    """

    value: float
    unit: str = "months"

    def to_months(self) -> float:
        """Convert to months for comparison."""
        if self.unit in ("month", "months"):
            return self.value
        elif self.unit in ("year", "years"):
            return self.value * 12
        elif self.unit in ("week", "weeks"):
            return self.value / 4.345
        elif self.unit in ("day", "days"):
            return self.value / 30
        else:
            raise ValueError(f"Unknown time unit: {self.unit}")

    def to_years(self) -> float:
        """Convert to years."""
        return self.to_months() / 12

    def to_list(self) -> list:
        """Convert to [value, unit] list format for JSON serialization."""
        # Use int if whole number
        val = int(self.value) if self.value == int(self.value) else self.value
        return [val, self.unit]

    def __str__(self) -> str:
        # Format value nicely (no decimal for whole numbers)
        if self.value == int(self.value):
            val_str = str(int(self.value))
        else:
            val_str = f"{self.value:.1f}"

        # Singular/plural unit
        unit = self.unit
        if self.value == 1:
            unit = unit.rstrip("s")  # "years" -> "year"

        return f"{val_str} {unit}"

    @staticmethod
    def parse_time_value(time_data) -> TimeValue:
        """
        Parse time value from various formats.

        Handles:
        - [value, unit] arrays: [5, "months"]
        - "value unit" strings: "5 months"
        - Dict with value/unit keys

        Robust to singular/plural: "1 year" == "1 years"
        """
        if isinstance(time_data, list) and len(time_data) == 2:
            value = float(time_data[0])
            unit = time_data[1]
        elif isinstance(time_data, str):
            parts = time_data.lower().strip().split()
            if len(parts) != 2:
                raise ValueError(f"Invalid time format: {time_data}")
            value = float(parts[0])
            unit = parts[1]
        elif isinstance(time_data, dict):
            value = float(time_data["value"])
            unit = time_data["unit"]
        else:
            raise ValueError(f"Unknown time format: {time_data}")

        # Normalize unit (robust to singular/plural)
        unit_lower = unit.lower()
        if unit_lower in ("month", "months"):
            unit = "months"
        elif unit_lower in ("year", "years"):
            unit = "years"
        elif unit_lower in ("day", "days"):
            unit = "days"
        elif unit_lower in ("week", "weeks"):
            unit = "weeks"
        else:
            unit = unit_lower

        return TimeValue(value=value, unit=unit)


@dataclass
class RewardValue(SchemaClass):
    """
    A reward value with unit: r_i

    Attributes:
        value: Numeric reward value
        unit: Reward unit (e.g., "dollars", "housing units")
    """

    value: float
    unit: str = ""

    def __str__(self) -> str:
        if self.unit:
            return f"{self.value:,.0f} {self.unit}"
        return f"{self.value:,.0f}"


# =============================================================================
# Intertemporal Option & Preference Schemas
# =============================================================================


@dataclass
class IntertemporalOption(SchemaClass):
    """
    An intertemporal option: o_i = (t_i, r_i)

    Represents a reward available at a specific time.

    Attributes:
        label: Option identifier (e.g., "a", "b")
        time: Time value t_i when reward is received
        reward: Reward value r_i
    """

    label: str
    time: TimeValue
    reward: RewardValue


@dataclass
class PreferencePair(SchemaClass):
    """
    A pair of intertemporal options for comparison.

    By convention, short_term has smaller time value than long_term.

    Attributes:
        short_term: The shorter-term option (smaller t_i)
        long_term: The longer-term option (larger t_j)
    """

    short_term: IntertemporalOption
    long_term: IntertemporalOption


# =============================================================================
# Prompt & Response Types
# =============================================================================


@dataclass
class Prompt(SchemaClass):
    """
    Input prompt: x = (t_h, q_x, s_c)

    Attributes:
        question: Preference question q_x (includes pair and time_horizon)
        context: Context string s_c
        text: Full formatted prompt text
        response_format: Expected response format template
    """

    preference_pair: PreferencePair
    time_horizon: Optional[TimeValue] = None  # None = no time horizon constraint
    text: str = ""


@dataclass
class Response(SchemaClass):
    """
    Output response: y = (o_k, s_t)

    Attributes:
        chosen_option: The selected option o_k (label: "a" or "b")
        trace: Strategy/reasoning trace string s_t
        raw_text: Raw response text from model
    """

    chosen_option: str
    text: str = ""
    reasoning_trace: str = ""


# =============================================================================
# Dataset Internal Types
# =============================================================================


@dataclass
class DatasetSample(SchemaClass):
    """
    A complete dataset sample (internal representation).

    Attributes:
        id: Unique sample identifier
        prompt: The full prompt object
        response: Model response (populated after inference)
    """

    sample_id: int
    prompt: Prompt
    response: Optional[Response] = None
