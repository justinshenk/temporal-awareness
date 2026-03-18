"""Config schemas for prompt dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.common import BaseSchema, TimeValue
from ..formatting.prompt_formats import find_prompt_format_config

SCHEMA_VERSION = "1.0"


class StepType(Enum):
    """Stepping type for grid generation."""

    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"


@dataclass
class ContextConfig(BaseSchema):
    """Context configuration for dataset."""

    reward_unit: str = "points"
    role: str = "reasonable decision maker"
    situation: str = ""
    task_in_question: str = "decide between options"
    reasoning_ask: str = "Provide reasoning on why this choice was made."
    domain: str = ""
    extra_situation: str = ""
    labels: tuple[str, str] = ("a)", "b)")

    @classmethod
    def from_dict(cls, d: dict) -> "ContextConfig":
        """Merge with defaults, then parse."""
        defaults = cls().to_dict()
        return super().from_dict(defaults | d)

    def get_context_id(self) -> int:
        """Get a deterministic context ID based on context fields.

        Returns a positive integer ID that uniquely identifies this context
        configuration (excluding labels, which are tracked by formatting_id).
        """
        # Hash all context fields except labels (which are formatting-related)
        hashable = (
            self.reward_unit,
            self.role,
            self.situation,
            self.task_in_question,
            self.reasoning_ask,
            self.domain,
            self.extra_situation,
        )
        return hash(hashable) & 0x7FFFFFFF  # Ensure positive


@dataclass
class OptionRangeConfig(BaseSchema):
    """Option value ranges configuration."""

    reward_range: tuple[float, float]
    time_range: tuple[TimeValue, TimeValue]
    reward_steps: tuple[int, StepType] = (1, StepType.LINEAR)
    time_steps: tuple[int, StepType] = (1, StepType.LINEAR)


@dataclass
class PromptDatasetConfig(BaseSchema):
    """Prompt dataset generation configuration."""

    name: str
    context: ContextConfig
    options: dict[str, OptionRangeConfig]
    time_horizons: list[TimeValue | None]

    add_formatting_noise: bool = False
    do_formatting_variation_grid: bool = False
    do_full_formatting_variation_grid: bool = False
    do_context_variations: bool = False
    prompt_format: str = "default_prompt_format"

    @property
    def prompt_format_config(self):
        """Get the resolved prompt format config object."""
        return find_prompt_format_config(self.prompt_format)

    def get_filename(self) -> str:
        """Get the filename for saving this prompt dataset."""
        return f"{self.name}_{self.get_id()}.json"
