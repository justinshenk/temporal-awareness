"""Config schemas for dataset generation."""

from dataclasses import dataclass, asdict
from enum import Enum

from ..common.schema_utils import SchemaClass
from ..common.types import TimeValue


SCHEMA_VERSION = "1.0"


class StepType(Enum):
    """Stepping type for grid generation."""

    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"


@dataclass(kw_only=True)
class PromptFormatConfig(SchemaClass):
    """Prompt formatting configuration."""

    name: str
    question_template: str
    response_template: str
    const_keywords: dict
    keywords: list
    var_keywords: list


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
class DatasetConfig(SchemaClass):
    """Dataset generation configuration."""

    name: str
    context: ContextConfig
    prompt_format: PromptFormatConfig
    options: dict[str, OptionRangeConfig]
    time_horizons: list[TimeValue | None]
    add_formatting_variations: bool = False

    def to_dict(self) -> dict:
        config_dict = asdict(self)
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
