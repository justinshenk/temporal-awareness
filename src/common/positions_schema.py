"""Standardized positions schema for cross-script interoperability.

This defines the format for positions.json files that can be shared between scripts:
- activation_patching outputs positions.json
- steering_vectors reads positions.json to know where to intervene
- probes reads positions.json to know where to extract activations
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import json


@dataclass
class PositionSpec:
    """A single position specification."""

    position: int  # Token position index
    token: str  # Token string at this position
    score: float  # Importance score (method-dependent)
    layer: Optional[int] = None  # Layer index (None = all layers)
    section: Optional[str] = None  # Section name (e.g., "choices", "time_horizon")

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove None values for cleaner JSON
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class PositionsFile:
    """Standardized positions file format."""

    model: str  # Model name/path
    method: (
        str  # Method that generated this (e.g., "activation_patching", "attribution")
    )
    positions: list[PositionSpec]

    # Optional metadata
    dataset_id: Optional[str] = None
    threshold: Optional[float] = None
    component: Optional[str] = None  # e.g., "resid_post", "attn_out"

    def to_dict(self) -> dict:
        d = {
            "model": self.model,
            "method": self.method,
            "positions": [p.to_dict() for p in self.positions],
        }
        if self.dataset_id:
            d["dataset_id"] = self.dataset_id
        if self.threshold is not None:
            d["threshold"] = self.threshold
        if self.component:
            d["component"] = self.component
        return d

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved positions: {path}")

    @classmethod
    def load(cls, path: Path) -> "PositionsFile":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)

        positions = [
            PositionSpec(
                position=p["position"],
                token=p["token"],
                score=p.get("score", p.get("recovery", 0.0)),  # Handle both keys
                layer=p.get("layer"),
                section=p.get("section"),
            )
            for p in data["positions"]
        ]

        return cls(
            model=data.get("model", "unknown"),
            method=data.get("method", "unknown"),
            positions=positions,
            dataset_id=data.get("dataset_id"),
            threshold=data.get("threshold"),
            component=data.get("component"),
        )

    def filter_by_score(self, min_score: float) -> "PositionsFile":
        """Return new PositionsFile with positions above threshold."""
        filtered = [p for p in self.positions if p.score >= min_score]
        return PositionsFile(
            model=self.model,
            method=self.method,
            positions=filtered,
            dataset_id=self.dataset_id,
            threshold=min_score,
            component=self.component,
        )

    def get_positions(self, layer: Optional[int] = None) -> list[int]:
        """Get position indices, optionally filtered by layer."""
        if layer is None:
            return [p.position for p in self.positions]
        return [
            p.position for p in self.positions if p.layer is None or p.layer == layer
        ]

    def get_top_n(self, n: int) -> list[PositionSpec]:
        """Get top N positions by score."""
        sorted_pos = sorted(self.positions, key=lambda p: p.score, reverse=True)
        return sorted_pos[:n]
