"""Generate prompt records from the feature-geometry parameter grid."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypedDict

try:
    from .paramterized_dataset import tasks, templates, values
except ImportError:
    from paramterized_dataset import tasks, templates, values


class TemplateConfig(TypedDict):
    id: str
    template: str


class PromptRecord(TypedDict):
    text: str
    template_id: str
    task: str
    base_value: int
    base_unit: str
    unit_variant: Literal["original", "smaller"]
    number_format: Literal["numeric", "words"]
    value: int
    value_text: str
    unit: str


NumberFormat = Literal["numeric", "words"]
UnitVariant = Literal["original", "smaller"]

NUMBER_FORMATS: tuple[NumberFormat, ...] = ("numeric", "words")

SINGULAR_UNITS = {
    "minutes": "minute",
    "hours": "hour",
    "days": "day",
    "weeks": "week",
    "months": "month",
    "years": "year",
    "decades": "decade",
    "centuries": "century",
    "millennia": "millennium",
}

NEXT_SMALLEST_UNIT = {
    "hours": ("minutes", 60),
    "days": ("hours", 24),
    "weeks": ("days", 7),
    "months": ("weeks", 4),
    "years": ("months", 12),
    "decades": ("years", 10),
    "centuries": ("decades", 10),
    "millennia": ("centuries", 10),
}

ONES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def number_to_words(value: int) -> str:
    """Return a lowercase English rendering for non-negative integers."""
    if value < 0:
        raise ValueError(f"Unsupported number for word rendering: {value}")
    if value < 20:
        return ONES[value]
    if value < 100:
        tens = value // 10 * 10
        remainder = value % 10
        if remainder == 0:
            return TENS[tens]
        return f"{TENS[tens]} {ONES[remainder]}"
    if value < 1000:
        hundreds = value // 100
        remainder = value % 100
        if remainder == 0:
            return f"{ONES[hundreds]} hundred"
        return f"{ONES[hundreds]} hundred {number_to_words(remainder)}"
    if value < 1_000_000:
        thousands = value // 1000
        remainder = value % 1000
        if remainder == 0:
            return f"{number_to_words(thousands)} thousand"
        return f"{number_to_words(thousands)} thousand {number_to_words(remainder)}"
    raise ValueError(f"Unsupported number for word rendering: {value}")


def singularize_unit(unit: str) -> str:
    """Return the singular form for a time unit."""
    if unit in SINGULAR_UNITS:
        return SINGULAR_UNITS[unit]
    if unit.endswith("s"):
        return unit[:-1]
    return unit


def render_unit(value: int, unit: str) -> str:
    """Return the unit with singular form when the value is one."""
    return singularize_unit(unit) if value == 1 else unit


def smaller_unit_value(value: int, unit: str) -> tuple[int, str] | None:
    """Convert a value to the next smallest configured time unit."""
    conversion = NEXT_SMALLEST_UNIT.get(unit)
    if conversion is None:
        return None

    smaller_unit, multiplier = conversion
    return value * multiplier, smaller_unit


def make_prompt_record(
    template: str,
    template_id: str,
    task: str,
    base_value: int,
    base_unit: str,
    value: int,
    unit: str,
    unit_variant: UnitVariant,
    number_format: NumberFormat,
) -> PromptRecord:
    """Return one formatted prompt plus the parameters that generated it."""
    rendered_unit = render_unit(value, unit)
    value_text = str(value) if number_format == "numeric" else number_to_words(value)

    return {
        "text": template.format(task=task, vlaue=value_text, unit=rendered_unit),
        "template_id": template_id,
        "task": task,
        "base_value": base_value,
        "base_unit": base_unit,
        "unit_variant": unit_variant,
        "number_format": number_format,
        "value": value,
        "value_text": value_text,
        "unit": rendered_unit,
    }


def append_prompt_variants(
    records: list[PromptRecord],
    template: str,
    template_id: str,
    task: str,
    base_value: int,
    base_unit: str,
    value: int,
    unit: str,
    unit_variant: UnitVariant,
) -> None:
    """Append numeric and word-number records for one prompt."""
    for number_format in NUMBER_FORMATS:
        records.append(
            make_prompt_record(
                template=template,
                template_id=template_id,
                task=task,
                base_value=base_value,
                base_unit=base_unit,
                value=value,
                unit=unit,
                unit_variant=unit_variant,
                number_format=number_format,
            )
        )


def generate_task_dataset(
    template_list: Iterable[TemplateConfig] = templates,  # type: ignore
    task_units: dict[str, set[str]] = tasks,
    time_values: Iterable[int] = values,
    output_path: str | Path | None = None,
) -> list[PromptRecord]:
    """Return every formatted prompt record from the configured templates and tasks."""
    records: list[PromptRecord] = []

    for template_config in template_list:
        template_id = template_config["id"]
        template = template_config["template"]
        for task, units in sorted(task_units.items()):
            for unit in sorted(units):
                for value in time_values:
                    append_prompt_variants(
                        records=records,
                        template=template,
                        template_id=template_id,
                        task=task,
                        base_value=value,
                        base_unit=unit,
                        value=value,
                        unit=unit,
                        unit_variant="original",
                    )

                    smaller = smaller_unit_value(value, unit)
                    if smaller is not None:
                        smaller_value, smaller_unit = smaller
                        append_prompt_variants(
                            records=records,
                            template=template,
                            template_id=template_id,
                            task=task,
                            base_value=value,
                            base_unit=unit,
                            value=smaller_value,
                            unit=smaller_unit,
                            unit_variant="smaller",
                        )

    if output_path is not None:
        Path(output_path).write_text(json.dumps(records, indent=2), encoding="utf-8")

    return records


dataset: list[PromptRecord] = generate_task_dataset()


if __name__ == "__main__":
    print(json.dumps(dataset, indent=2))
