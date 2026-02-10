"""
Internal type definitions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import types
from dataclasses import asdict, dataclass, fields, is_dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

from .io import load_json


# =============================================================================
# Schema utilities (deterministic ID generation)
# =============================================================================


def _qfloat(x: float, places: int = 8) -> float:
    """Stable decimal rounding: converts via str -> Decimal -> quantize."""
    q = Decimal(1) / (Decimal(10) ** places)  # e.g. 1e-8
    d = Decimal(str(x)).quantize(q, rounding=ROUND_HALF_EVEN)
    # normalize -0.0 to 0.0 for stability
    f = float(d)
    return 0.0 if f == 0.0 else f


def _canon(obj: Any, places: int = 8):
    """Canonicalize object for deterministic hashing."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
        return _qfloat(obj, places)
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return _canon(asdict(obj), places)
    if isinstance(obj, dict):
        return {k: _canon(v, places) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v, places) for v in obj]
    return obj


def deterministic_id_from_dataclass(
    data_class_obj: Any, places: int = 8, digest_bytes: int = 16
) -> str:
    """Generate a deterministic ID from a dataclass object."""
    canonical = _canon(data_class_obj, places)
    payload = json.dumps(
        canonical,
        sort_keys=True,  # stable key order
        separators=(",", ":"),  # remove whitespace
        ensure_ascii=False,
        allow_nan=False,
    )
    # fast, strong hash in the stdlib
    h = hashlib.blake2b(payload.encode("utf-8"), digest_size=digest_bytes)
    return h.hexdigest()


# =============================================================================
# Helper dataclass
# =============================================================================


@dataclass
class SchemaClass:
    # Each schema gets unique id based on values
    def get_id(self) -> str:
        return deterministic_id_from_dataclass(self)

    def to_dict(self) -> dict:
        return _canon(self)

    # For logging ease
    def __str__(self) -> str:
        result_dict = self.to_dict()
        return json.dumps(result_dict, indent=4)

    # Each SchemaClass obj should have their own set of params
    # We want to make sure schemas are unique and immutable
    def __post_init__(self):
        for f in fields(self):
            setattr(self, f.name, copy.deepcopy(getattr(self, f.name)))

    def __copy__(self):
        return self.__deepcopy__({})

    def __deepcopy__(self, memo):
        cls = self.__class__
        kwargs = {
            f.name: copy.deepcopy(getattr(self, f.name), memo) for f in fields(self)
        }
        return cls(**kwargs)

    @classmethod
    def _convert_value(cls, val, field_type):
        """Convert a value to the expected field type."""
        # Unwrap Optional[X] / X | None to get X
        origin = get_origin(field_type)
        if origin is Union or isinstance(field_type, types.UnionType):
            args = [a for a in get_args(field_type) if a is not type(None)]
            if len(args) == 1:
                field_type = args[0]

        # Handle None
        if val is None:
            return None

        # Handle Enum
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            return field_type(val) if not isinstance(val, field_type) else val

        # Handle dataclass
        if isinstance(val, dict) and is_dataclass(field_type):
            return field_type.from_dict(val)

        # Handle list[X]
        if get_origin(field_type) is list:
            item_type = get_args(field_type)[0] if get_args(field_type) else None
            if item_type:
                return [cls._convert_value(item, item_type) for item in val]

        # Handle tuple (convert from list, recursively convert items)
        if get_origin(field_type) is tuple:
            item_types = get_args(field_type)
            if item_types:
                return tuple(
                    cls._convert_value(
                        item, item_types[i] if i < len(item_types) else item_types[-1]
                    )
                    for i, item in enumerate(val)
                )
            return tuple(val)

        # Handle dict[K, V]
        if get_origin(field_type) is dict:
            key_type, val_type = (
                get_args(field_type) if get_args(field_type) else (None, None)
            )
            if val_type and is_dataclass(val_type):
                return {k: cls._convert_value(v, val_type) for k, v in val.items()}

        return val

    @classmethod
    def from_dict(cls, d: dict):
        """Recursively construct a dataclass instance from a nested dict."""
        hints = get_type_hints(cls)
        kwargs = {}
        for f in fields(cls):
            if f.name not in d:
                continue  # Let dataclass use its default
            val = d[f.name]
            field_type = hints.get(f.name)
            kwargs[f.name] = cls._convert_value(val, field_type) if field_type else val
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: Path):
        """Load from JSON file. Override from_dict for custom parsing."""
        data = load_json(path)
        return cls.from_dict(data)

    def __setattr__(self, name, value):
        super().__setattr__(name, copy.deepcopy(value))

    def as_base(self):
        cls = type(self).__bases__[0]
        return cls(**{f.name: getattr(self, f.name) for f in fields(cls)})


# =============================================================================
# Base Value Types
# =============================================================================

# Conversion factors to years
TIME_UNIT_TO_YEARS = {
    "years": 1.0,
    "year": 1.0,
    "months": 1.0 / 12.0,
    "month": 1.0 / 12.0,
    "weeks": 1.0 / 52.1429,
    "week": 1.0 / 52.1429,
    "days": 1.0 / 365.25,
    "day": 1.0 / 365.25,
    "hours": 1.0 / (365.25 * 24),
    "hour": 1.0 / (365.25 * 24),
    "decades": 10.0,
    "decade": 10.0,
}

# Available time units for variation
TIME_UNITS = ["years", "months", "weeks", "days", "hours", "decades"]
DEFAULT_TIME_UNIT = "years"


@dataclass
class TimeValue(SchemaClass):
    """
    A time value with unit: t_i

    Attributes:
        value: Numeric time value
        unit: Time unit (e.g., "months", "years")
    """

    value: float
    unit: str = DEFAULT_TIME_UNIT

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
    def parse(time_data) -> TimeValue:
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
class PromptSample(SchemaClass):
    """
    A complete prompt sample (internal representation).

    Attributes:
        sample_idx: Sample index within the dataset
        prompt: The full prompt object
        response: Model response (populated after inference)
    """

    sample_idx: int
    prompt: Prompt
    response: Optional[Response] = None


@dataclass
class CapturedInternals:
    """Captured activations from a forward pass."""

    activations: dict  # name -> tensor
    activation_names: list[str]

    @classmethod
    def from_activation_names(cls, activation_names: list[str], internals: dict):
        activations = {}
        for name in activation_names:
            if name in internals:
                activations[name] = internals[name][0].cpu()
        return CapturedInternals(
            activations=activations,
            activation_names=list(activations.keys()),
        )


@dataclass
class PreferenceSample(SchemaClass):
    """Single preference result."""

    # Minimal Info
    sample_idx: int
    choice: Any  # BinaryChoice type

    # Sample Info
    time_horizon: Optional[dict] = None  # in DEFAULT_TIME_UNIT
    response_text: str | None = None
    prompt_text: str | None = None

    # Choice Extra Info
    short_term_label: str | None = None
    long_term_label: str | None = None
    short_term_reward: float | None = None
    long_term_reward: float | None = None
    short_term_time: float | None = None  # in DEFAULT_TIME_UNIT
    long_term_time: float | None = None  # in DEFAULT_TIME_UNIT

    # Extra Info
    internals: Any | None = None  # CapturedInternals type
    internals_paths: dict | None = None
    decoding_mismatch: bool | None = None

    @property
    def choice_prob(self):
        self.choice.divergent_probs[self.choice.choice_idx]

    @property
    def alternative_prob(self):
        self.choice.divergent_probs[1 - self.choice.choice_idx]

    def to_dict(self):
        d = super().to_dict()
        d["choice_prob"] = self.choice_prob
        d["alternative_prob"] = self.alternative_prob
        return d
