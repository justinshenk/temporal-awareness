"""Configuration for difference-in-means analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from ....common.base_schema import BaseSchema


@dataclass
class DiffMeansConfig(BaseSchema):
    """Configuration for difference-in-means analysis.

    Attributes:
        enabled: Whether to run diffmeans analysis
        positions: Additional positions to analyze (semantic names or absolute)
        no_cache: Skip loading from cache
    """

    enabled: bool = True
    positions: list[str | int] | None = None
    no_cache: bool = False
