"""Base class for sample position mappings.

This module provides generic position mapping types that can be used
across different domains. Domain-specific builders should be in their
respective modules (e.g., src/intertemporal/common/sample_position_mapping.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .base_schema import BaseSchema
from .position_info import TokenPositionInfo


@dataclass
class SamplePositionMappingBase(BaseSchema):
    """Base class for sample position mappings (generic, no build methods).

    Maps every absolute token position to its semantic meaning.

    Attributes:
        sample_idx: Index of the sample
        prompt_len: Number of tokens in the prompt
        full_len: Total number of tokens (prompt + response)
        positions: List of TokenPositionInfo, indexed by abs_pos
        named_positions: Dict mapping format_pos names to list of abs_pos indices
    """

    sample_idx: int = 0
    prompt_len: int = 0
    full_len: int = 0
    positions: list[TokenPositionInfo] = field(default_factory=list)
    named_positions: dict[str, list[int]] = field(default_factory=dict)

    def get_position(self, abs_pos: int) -> TokenPositionInfo | None:
        """Get position info by absolute position."""
        if 0 <= abs_pos < len(self.positions):
            return self.positions[abs_pos]
        return None

    def get_positions_by_name(self, format_pos: str) -> list[TokenPositionInfo]:
        """Get all positions for a named format position."""
        if format_pos not in self.named_positions:
            return []
        return [self.positions[i] for i in self.named_positions[format_pos]]

    def get_format_pos_names(self) -> list[str]:
        """Get list of all format position names in this sample."""
        return list(self.named_positions.keys())

    def get_format_pos_label(self, abs_pos: int, fallback: str = "") -> str:
        """Get format_pos name or fallback.

        Args:
            abs_pos: Absolute position to look up
            fallback: Fallback string if position not found or has no format_pos.
                     If empty, uses "P{abs_pos}" as fallback.

        Returns:
            The format_pos name if found, otherwise the fallback string.
        """
        pos_info = self.get_position(abs_pos)
        if pos_info and pos_info.format_pos:
            return pos_info.format_pos
        return fallback or f"P{abs_pos}"


@dataclass
class DatasetPositionMappingBase(BaseSchema):
    """Base class for position mappings across all samples in a dataset."""

    mappings: list[SamplePositionMappingBase] = field(default_factory=list)

    def add(self, mapping: SamplePositionMappingBase) -> None:
        """Add a sample mapping."""
        self.mappings.append(mapping)

    def get(self, sample_idx: int) -> SamplePositionMappingBase | None:
        """Get mapping by sample index."""
        for m in self.mappings:
            if m.sample_idx == sample_idx:
                return m
        return None

    def __len__(self) -> int:
        return len(self.mappings)

    def __iter__(self):
        return iter(self.mappings)

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DatasetPositionMappingBase":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
