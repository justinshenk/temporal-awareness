"""Config schemas for prompt dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from ..common.schema_utils import SchemaClass
from ..common.io import load_json

if TYPE_CHECKING:
    from ..common.types import TimeValue
    from ..formatting.configs import DefaultPromptFormat


SCHEMA_VERSION = "1.0"


class StepType(Enum):
    """Stepping type for grid generation."""

    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"


@dataclass
class ContextConfig(SchemaClass):
    """Context configuration for dataset."""

    reward_unit: str
    role: str
    situation: str
    task_in_question: str = "decide between options"
    reasoning_ask: str = "Provide reasoning on why this choice was made."
    domain: str = ""
    extra_situation: str = ""
    labels: tuple[str, str] = ("a)", "b)")
    method: str = "grid"


@dataclass
class OptionRangeConfig(SchemaClass):
    """Option value ranges configuration."""

    reward_range: tuple[float, float]
    time_range: tuple[TimeValue, TimeValue]
    reward_steps: tuple[int, StepType] = (1, StepType.LINEAR)
    time_steps: tuple[int, StepType] = (1, StepType.LINEAR)


@dataclass
class PromptDatasetConfig(SchemaClass):
    """Prompt dataset generation configuration."""

    name: str
    context: ContextConfig
    prompt_format: PromptFormatConfig
    options: dict[str, OptionRangeConfig]
    time_horizons: list[TimeValue | None]
    add_formatting_variations: bool = False

    def to_dict(self) -> dict:
        config_dict = asdict(self)
        # Save prompt_format as just the name string
        config_dict["prompt_format"] = self.prompt_format.name
        config_dict["time_horizons"] = [
            t.to_list() if t is not None else None for t in self.time_horizons
        ]
        config_dict["context"]["labels"] = list(self.context.labels)
        for key in ("short_term", "long_term"):
            opt = self.options[key]
            config_dict["options"][key]["time_range"] = [
                opt.time_range[0].to_list(),
                opt.time_range[1].to_list(),
            ]
            config_dict["options"][key]["reward_steps"] = [
                opt.reward_steps[0],
                opt.reward_steps[1].value,
            ]
            config_dict["options"][key]["time_steps"] = [
                opt.time_steps[0],
                opt.time_steps[1].value,
            ]
        return config_dict

    def get_filename(self) -> str:
        """Get the filename for saving this prompt dataset."""
        return f"{self.name}_{self.get_id()}.json"

    @classmethod
    def load_from_json(cls, path: Path) -> "PromptDatasetConfig":
        """Load and parse dataset config from JSON file."""
        data = load_json(path)
        return cls.load_from_dict(data)

    @classmethod
    def load_from_dict(cls, data: dict) -> "PromptDatasetConfig":
        """
        Parse dataset config from dictionary.

        Sample dict:
        {
            "name": "cityhousing",
            "context": {
                "reward_unit": "housing units",
                "role": "the city administration",
                "situation": "Plan for housing in the city.",
                "task_in_question": "build",
                "reasoning_ask": "why choice was made",
                "domain": "housing",
                "labels": ["a)", "b)"],
                "method": "grid",
            },
            "options": {
                "short_term": {
                    "reward_range": [2000, 5000],
                    "time_range": [[3, "months"], [1, "years"]],
                    "reward_steps": [1, "linear"],
                    "time_steps": [1, "linear"]
                },
                "long_term": { ... }
            },
            "time_horizons": [[5, "months"], [15, "years"]]
        }
        """
        from ..common.types import TimeValue
        from ..formatting.prompt_formats import find_prompt_format_config

        # Parse context (includes labels, method)
        ctx = data["context"]
        labels = ctx.get("labels", ["a)", "b)"])
        if isinstance(labels, list):
            labels = tuple(labels)

        context = ContextConfig(
            reward_unit=ctx["reward_unit"],
            role=ctx["role"],
            situation=ctx["situation"],
            task_in_question=ctx.get("task_in_question", "to decide between"),
            reasoning_ask=ctx.get("reasoning_ask", "why this choice was made"),
            domain=ctx.get("domain", ""),
            extra_situation=ctx.get("extra_situation", ""),
            labels=labels,
            method=ctx.get("method", "grid"),
        )

        prompt_format_name = data.get("prompt_format", "default_prompt_format")
        prompt_format = find_prompt_format_config(prompt_format_name)

        # Parse options
        options = {}
        for key in ("short_term", "long_term"):
            opt = data["options"][key]
            options[key] = OptionRangeConfig(
                reward_range=tuple(opt["reward_range"]),
                time_range=(
                    TimeValue.parse_time_value(opt["time_range"][0]),
                    TimeValue.parse_time_value(opt["time_range"][1]),
                ),
                reward_steps=(
                    opt["reward_steps"][0],
                    StepType(opt["reward_steps"][1]),
                ),
                time_steps=(
                    opt["time_steps"][0],
                    StepType(opt["time_steps"][1]),
                ),
            )

        # Parse time horizons (null/None = no time horizon constraint)
        time_horizons = [
            TimeValue.parse_time_value(t) if t is not None else None
            for t in data["time_horizons"]
        ]

        # Parse optional formatting variations flag
        add_formatting_variations = data.get("add_formatting_variations", False)

        return cls(
            name=data["name"],
            context=context,
            prompt_format=prompt_format,
            options=options,
            time_horizons=time_horizons,
            add_formatting_variations=add_formatting_variations,
        )
