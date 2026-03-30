"""Time value representation with unit conversions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base_schema import BaseSchema

# Astronomical year = 365.25 days
DAYS_PER_YEAR = 365.25
HOURS_PER_YEAR = DAYS_PER_YEAR * 24
MINUTES_PER_YEAR = HOURS_PER_YEAR * 60
SECONDS_PER_YEAR = MINUTES_PER_YEAR * 60

# Canonical units with their conversion factor to years
# Format: (canonical_name, years_per_unit, [aliases...])
_UNIT_DEFINITIONS: list[tuple[str, float, list[str]]] = [
    ("nanoseconds", 1.0 / (SECONDS_PER_YEAR * 1e9), ["nanosecond", "ns"]),
    ("microseconds", 1.0 / (SECONDS_PER_YEAR * 1e6), ["microsecond", "us", "μs"]),
    ("milliseconds", 1.0 / (SECONDS_PER_YEAR * 1e3), ["millisecond", "ms"]),
    ("seconds", 1.0 / SECONDS_PER_YEAR, ["second", "sec", "secs", "s"]),
    ("minutes", 1.0 / MINUTES_PER_YEAR, ["minute", "min", "mins"]),
    ("hours", 1.0 / HOURS_PER_YEAR, ["hour", "hr", "hrs", "h"]),
    ("days", 1.0 / DAYS_PER_YEAR, ["day", "d"]),
    ("weeks", 7.0 / DAYS_PER_YEAR, ["week", "wk", "wks"]),
    ("fortnights", 14.0 / DAYS_PER_YEAR, ["fortnight"]),
    ("months", 1.0 / 12.0, ["month", "mo", "mos"]),
    ("quarters", 0.25, ["quarter", "qtr", "qtrs", "q"]),
    ("semesters", 0.5, ["semester"]),
    ("years", 1.0, ["year", "yr", "yrs", "y"]),
    ("decades", 10.0, ["decade"]),
    ("scores", 20.0, ["score"]),
    ("generations", 25.0, ["generation", "gen", "gens"]),
    ("centuries", 100.0, ["century"]),
    ("millennia", 1000.0, ["millennium", "millenia", "millenium"]),
    ("eons", 1e9, ["eon", "aeons", "aeon"]),
]

# Build lookup tables from definitions
TIME_UNITS = [name for name, _, _ in _UNIT_DEFINITIONS]
TIME_UNIT_TO_YEARS: dict[str, float] = {}
_UNIT_TO_CANONICAL: dict[str, str] = {}

for canonical, years, aliases in _UNIT_DEFINITIONS:
    TIME_UNIT_TO_YEARS[canonical] = years
    _UNIT_TO_CANONICAL[canonical] = canonical
    for alias in aliases:
        TIME_UNIT_TO_YEARS[alias] = years
        _UNIT_TO_CANONICAL[alias] = canonical

DEFAULT_TIME_UNIT = "years"


def canonicalize_unit(unit: str) -> str:
    """Convert a time unit to its canonical plural form."""
    unit_lower = unit.lower().strip()
    if unit_lower in _UNIT_TO_CANONICAL:
        return _UNIT_TO_CANONICAL[unit_lower]
    raise ValueError(f"Unknown time unit: {unit}")


@dataclass
class TimeValue(BaseSchema):
    """A time value with unit."""

    value: float
    unit: str = DEFAULT_TIME_UNIT

    def to_years(self) -> float:
        """Convert to years."""
        unit_lower = self.unit.lower()
        if unit_lower not in TIME_UNIT_TO_YEARS:
            raise ValueError(f"Unknown time unit: {self.unit}")
        return self.value * TIME_UNIT_TO_YEARS[unit_lower]

    def to_months(self) -> float:
        return self.to_years() * 12

    def to_days(self) -> float:
        return self.to_years() * DAYS_PER_YEAR

    def to_hours(self) -> float:
        return self.to_years() * HOURS_PER_YEAR

    def to_minutes(self) -> float:
        return self.to_years() * MINUTES_PER_YEAR

    def to_seconds(self) -> float:
        return self.to_years() * SECONDS_PER_YEAR

    def to_unit(self, target_unit: str) -> float:
        """Convert to any supported unit."""
        target_lower = target_unit.lower()
        if target_lower not in TIME_UNIT_TO_YEARS:
            raise ValueError(f"Unknown target unit: {target_unit}")
        return self.to_years() / TIME_UNIT_TO_YEARS[target_lower]

    def convert(self, target_unit: str) -> TimeValue:
        """Return a new TimeValue converted to the target unit."""
        return TimeValue(
            value=self.to_unit(target_unit), unit=canonicalize_unit(target_unit)
        )

    def __str__(self) -> str:
        val_str = (
            str(int(self.value))
            if self.value == int(self.value)
            else f"{self.value:.1f}"
        )
        unit = self.unit
        if self.value == 1:
            if unit.endswith("ies"):
                unit = unit[:-3] + "y"
            elif unit.endswith("ia"):
                unit = unit[:-1] + "um"
            elif unit.endswith("s") and not unit.endswith("ss"):
                unit = unit[:-1]
        return f"{val_str} {unit}"

    def __lt__(self, other: TimeValue) -> bool:
        return self.to_years() < other.to_years()

    def __le__(self, other: TimeValue) -> bool:
        return self.to_years() <= other.to_years()

    def __gt__(self, other: TimeValue) -> bool:
        return self.to_years() > other.to_years()

    def __ge__(self, other: TimeValue) -> bool:
        return self.to_years() >= other.to_years()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeValue):
            return NotImplemented
        return abs(self.to_years() - other.to_years()) < 1e-10

    def __hash__(self) -> int:
        return hash(round(self.to_years(), 10))

    @classmethod
    def from_dict(cls, data) -> TimeValue:
        """Create TimeValue from dict or list format."""
        return cls.parse(data)

    @staticmethod
    def parse(time_data) -> TimeValue:
        """Parse time value from various formats.

        Accepts:
        - List: [value, unit] e.g., [5, "years"]
        - String: "5 years" or "5years"
        - Dict: {"value": 5, "unit": "years"}
        - TimeValue: returns as-is
        - int/float: assumes years
        """
        if isinstance(time_data, TimeValue):
            return time_data

        # Plain number assumes years
        if isinstance(time_data, (int, float)):
            return TimeValue(value=float(time_data), unit="years")

        if isinstance(time_data, list) and len(time_data) == 2:
            value, unit = float(time_data[0]), time_data[1]
        elif isinstance(time_data, str):
            time_str = time_data.lower().strip()
            parts = time_str.split()
            if len(parts) == 2:
                value, unit = float(parts[0]), parts[1]
            elif len(parts) == 1:
                match = re.match(r"^([\d.]+)\s*([a-zA-Zμ]+)$", time_str)
                if match:
                    value, unit = float(match.group(1)), match.group(2)
                else:
                    raise ValueError(f"Invalid time format: {time_data}")
            else:
                raise ValueError(f"Invalid time format: {time_data}")
        elif isinstance(time_data, dict):
            value, unit = float(time_data["value"]), time_data["unit"]
        else:
            raise ValueError(f"Unknown time format: {time_data}")

        return TimeValue(value=value, unit=canonicalize_unit(unit))


def parse_horizon_years(horizon) -> float | None:
    """Parse horizon to years, handling all formats including None.

    Accepts:
    - None: returns None
    - int/float: assumes years, returns as float
    - Dict with value=None: returns None
    - Dict/str/list: parses via TimeValue and converts to years
    """
    if horizon is None:
        return None
    if isinstance(horizon, (int, float)):
        return float(horizon)
    if isinstance(horizon, dict) and horizon.get("value") is None:
        return None
    try:
        return TimeValue.parse(horizon).to_years()
    except (ValueError, KeyError, TypeError):
        return None
