"""
Formatting variations for dataset generation.

Provides randomized variations for:
- Labels: different styles (a/b, x/y, [i]/[ii], OPTION_ONE/OPTION_TWO, etc.)
- Label order: which option appears first
- Time units: years, months, days, weeks, hours, decades
- Number format: numerical vs spelled out
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional
from itertools import product
from src.common import TimeValue, TIME_UNIT_TO_YEARS, TIME_UNITS


# =============================================================================
# Label Variations
# =============================================================================

# Different label pair styles
SELECT_LABEL_STYLES: list[tuple[str, str]] = [
    ("a)", "b)"),
    ("x)", "y)"),
    ("[i]", "[ii]"),
    ("[I]", "[II]"),
    ("first_option", "second_option"),
]
MORE_LABEL_STYLES: list[tuple[str, str]] = [
    ("A)", "B)"),
    ("a.", "b."),
    ("A.", "B."),
    ("X)", "Y)"),
    ("[a]", "[b]"),
    ("[A]", "[B]"),
    ("[1]", "[2]"),
    ("(1)", "(2)"),
    ("(a)", "(b)"),
    ("(A)", "(B)"),
    ("Option A:", "Option B:"),
    ("Option 1:", "Option 2:"),
    ("Choice A:", "Choice B:"),
    ("Choice 1:", "Choice 2:"),
    ("OPTION_ONE:", "OPTION_TWO:"),
    ("FIRST:", "SECOND:"),
]
LABEL_STYLES: list[tuple[str, str]] = SELECT_LABEL_STYLES + MORE_LABEL_STYLES


def get_formatting_id(label_a: str, label_b: str) -> int:
    """
    Get a formatting ID from label pair.

    The ID is deterministic: same labels always produce the same ID.
    Swapped labels produce the negated ID: (a, b) -> +N, (b, a) -> -N.

    Args:
        label_a: First label
        label_b: Second label

    Returns:
        Signed integer ID (positive for canonical order, negative for swapped)
    """
    # Create canonical ordering (alphabetically first label comes first)
    if label_a <= label_b:
        canonical = (label_a, label_b)
        sign = 1
    else:
        canonical = (label_b, label_a)
        sign = -1

    # Hash the canonical pair to get a positive ID
    base_id = hash(canonical) & 0x7FFFFFFF  # Ensure positive by masking sign bit

    return sign * base_id


def get_random_labels() -> tuple[str, str]:
    """Get a random label pair."""
    return random.choice(LABEL_STYLES)


def get_select_label_styles() -> list[tuple[str, str]]:
    """Get all available label styles."""
    return SELECT_LABEL_STYLES.copy()


def get_all_label_styles() -> list[tuple[str, str]]:
    """Get all available label styles."""
    return LABEL_STYLES.copy()


# =============================================================================
# Time Unit Conversions
# =============================================================================


def convert_time_value(tv: TimeValue, target_unit: str) -> TimeValue:
    """
    Convert a TimeValue to a different unit.

    Args:
        tv: Original TimeValue
        target_unit: Target unit (years, months, weeks, days, hours, decades)

    Returns:
        New TimeValue in target unit
    """
    # Convert to years first
    years = tv.to_years()

    # Convert from years to target unit
    if target_unit not in TIME_UNIT_TO_YEARS:
        raise ValueError(f"Unknown time unit: {target_unit}")

    target_value = years / TIME_UNIT_TO_YEARS[target_unit]

    return TimeValue(value=target_value, unit=target_unit)


def get_sensible_units_for_time(tv: TimeValue) -> list[str]:
    """
    Get list of sensible units for a given time value.

    Filters out units that would result in very small or very large numbers.

    Args:
        tv: TimeValue to find sensible units for

    Returns:
        List of sensible unit names
    """
    years = tv.to_years()
    sensible = []

    for unit in TIME_UNITS:
        converted_value = years / TIME_UNIT_TO_YEARS[unit]

        # Filter out extreme values
        if 0.1 <= converted_value <= 10000:
            sensible.append(unit)

    # Always include original unit if not already there
    if tv.unit not in sensible:
        sensible.append(tv.unit)

    return sensible


def get_random_time_unit(tv: TimeValue) -> str:
    """
    Get a random sensible unit for the given time value.

    Args:
        tv: TimeValue to convert

    Returns:
        Random sensible unit name
    """
    sensible_units = get_sensible_units_for_time(tv)
    return random.choice(sensible_units)


def convert_to_random_unit(tv: TimeValue) -> TimeValue:
    """
    Convert TimeValue to a random sensible unit.

    Args:
        tv: Original TimeValue

    Returns:
        TimeValue in a randomly chosen sensible unit
    """
    target_unit = get_random_time_unit(tv)
    return convert_time_value(tv, target_unit)


# =============================================================================
# Number Spelling
# =============================================================================

# Basic number words
NUMBER_WORDS = {
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
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
    100: "one hundred",
    1000: "one thousand",
}

# Fractional expressions
FRACTION_WORDS = {
    0.1: "a tenth of a",
    0.2: "a fifth of a",
    0.25: "a quarter of a",
    0.5: "half a",
    0.75: "three quarters of a",
    1.5: "one and a half",
    2.5: "two and a half",
}


def spell_number(n: float) -> Optional[str]:
    """
    Spell out a number if it has a clean word representation.

    Args:
        n: Number to spell

    Returns:
        Spelled out string, or None if no clean representation
    """
    # Check exact matches first
    if n in NUMBER_WORDS:
        return NUMBER_WORDS[n]

    # Check fractions
    if n in FRACTION_WORDS:
        return FRACTION_WORDS[n]

    # Handle integers up to 99
    if isinstance(n, int) or (isinstance(n, float) and n == int(n)):
        n_int = int(n)
        if 21 <= n_int <= 99:
            tens = (n_int // 10) * 10
            ones = n_int % 10
            if tens in NUMBER_WORDS and ones in NUMBER_WORDS:
                return f"{NUMBER_WORDS[tens]}-{NUMBER_WORDS[ones]}"

    return None


def format_time_spelled(tv: TimeValue) -> Optional[str]:
    """
    Format a TimeValue with spelled-out numbers.

    Args:
        tv: TimeValue to format

    Returns:
        Spelled-out string, or None if not applicable
    """
    spelled = spell_number(tv.value)
    if spelled is None:
        return None

    unit = tv.unit
    # Handle singular/plural
    if tv.value == 1 or spelled in (
        "a tenth of a",
        "a fifth of a",
        "a quarter of a",
        "half a",
    ):
        # Use singular
        unit = unit.rstrip("s")

    # Special handling for fractions
    if spelled in FRACTION_WORDS.values():
        return f"{spelled} {unit}"

    return f"{spelled} {unit}"


def format_time_value(tv: TimeValue, spell_out: bool = False) -> str:
    """
    Format a TimeValue, optionally spelling out numbers.

    Args:
        tv: TimeValue to format
        spell_out: Whether to attempt spelling out numbers

    Returns:
        Formatted string
    """
    if spell_out:
        spelled = format_time_spelled(tv)
        if spelled:
            return spelled

    # Default numerical format
    value = tv.value
    unit = tv.unit

    # Round to reasonable precision
    if value == int(value):
        value_str = str(int(value))
    elif value < 10:
        value_str = f"{value:.2f}".rstrip("0").rstrip(".")
    else:
        value_str = f"{value:.1f}".rstrip("0").rstrip(".")

    return f"{value_str} {unit}"


# =============================================================================
# Combined Variation
# =============================================================================


@dataclass
class FormattingVariation:
    """Configuration for a specific formatting variation."""

    labels: tuple[str, str]  # Label pair to use
    flip_order: bool  # Whether to flip short/long term order
    time_unit_variation: bool  # Whether to vary time units
    spell_numbers: bool  # Whether to spell out numbers

    @classmethod
    def random(cls) -> "FormattingVariation":
        return cls(
            labels=get_random_labels(),
            flip_order=random.choice([True, False]),
            time_unit_variation=random.choice([True, False]),
            spell_numbers=random.choice(
                [True, False, False, False]
            ),  # Less likely to spell
        )

    @classmethod
    def get_grid(cls) -> list["FormattingVariation"]:
        labels_grid = get_select_label_styles()
        flip_grid = [True, False]

        # Not variations on these yet
        random_unit_grid = [False]
        spell_grid = [False]

        full_grid = product(labels_grid, flip_grid, random_unit_grid, spell_grid)
        return [cls(*p) for p in full_grid]

    @classmethod
    def default(cls) -> "FormattingVariation":
        """Create default (no variation) formatting."""
        return cls(
            labels=get_all_label_styles()[0],
            flip_order=False,
            time_unit_variation=False,
            spell_numbers=False,
        )


def apply_time_variation(
    tv: TimeValue,
    variation: FormattingVariation,
) -> tuple[TimeValue, str]:
    """
    Apply time variation to a TimeValue.

    Args:
        tv: Original TimeValue
        variation: Formatting variation config

    Returns:
        Tuple of (possibly converted TimeValue, formatted string)
    """
    result_tv = tv

    # Apply unit variation
    if variation.time_unit_variation:
        result_tv = convert_to_random_unit(tv)

    # Format with or without spelling
    formatted = format_time_value(result_tv, spell_out=variation.spell_numbers)

    return result_tv, formatted
