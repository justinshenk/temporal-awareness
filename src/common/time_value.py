"""Time value representation with unit conversions."""

from __future__ import annotations

from dataclasses import dataclass

from .base_schema import BaseSchema


# =============================================================================
# Time Constants
# =============================================================================

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

TIME_UNITS = ["years", "months", "weeks", "days", "hours", "decades"]
DEFAULT_TIME_UNIT = "years"


# =============================================================================
# TimeValue
# =============================================================================


@dataclass
class TimeValue(BaseSchema):
    """A time value with unit."""

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
        if self.value == int(self.value):
            val_str = str(int(self.value))
        else:
            val_str = f"{self.value:.1f}"

        unit = self.unit
        if self.value == 1:
            unit = unit.rstrip("s")

        return f"{val_str} {unit}"

    @classmethod
    def from_dict(cls, data) -> TimeValue:
        """Create TimeValue from dict or list format (for BaseSchema compatibility)."""
        return cls.parse(data)

    @staticmethod
    def parse(time_data) -> TimeValue:
        """Parse time value from various formats."""
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
