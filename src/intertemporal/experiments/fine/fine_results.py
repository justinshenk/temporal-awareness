"""Result dataclasses for fine-grained patching analysis.

NOTE: Head attribution and position patching results are now in attn_head_attribution.py
NOTE: Neuron attribution results are now in mlp_analysis_results.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ....common.base_schema import BaseSchema


@dataclass
class PathPatchingResult(BaseSchema):
    """Result for path patching between source and destination components.

    Attributes:
        source_layer: Source layer
        source_head: Source head (or -1 for MLP)
        dest_layer: Destination layer
        dest_head: Destination head (or -1 for MLP)
        dest_component: "attn" or "mlp"
        effect: Path patching effect (how much source affects dest)
    """

    source_layer: int
    source_head: int
    dest_layer: int
    dest_head: int = -1  # -1 for MLP
    dest_component: str = "mlp"
    effect: float = 0.0

    @property
    def source_label(self) -> str:
        if self.source_head >= 0:
            return f"L{self.source_layer}.H{self.source_head}"
        return f"L{self.source_layer}.MLP"

    @property
    def dest_label(self) -> str:
        if self.dest_head >= 0:
            return f"L{self.dest_layer}.H{self.dest_head}"
        return f"L{self.dest_layer}.MLP"


@dataclass
class MultiSiteResult(BaseSchema):
    """Result for multi-site interaction patching.

    Measures interaction effect = joint - individual_A - individual_B

    Attributes:
        component_a: First component label (e.g., "L24.H7")
        component_b: Second component label
        individual_a: Effect of patching A alone
        individual_b: Effect of patching B alone
        joint: Effect of patching A and B together
        interaction: interaction = joint - individual_a - individual_b
    """

    component_a: str
    component_b: str
    individual_a: float = 0.0
    individual_b: float = 0.0
    joint: float = 0.0

    @property
    def interaction(self) -> float:
        """Interaction effect: joint - sum of individuals."""
        return self.joint - self.individual_a - self.individual_b


@dataclass
class LayerPositionResult(BaseSchema):
    """Result for layer x position fine patching.

    Stores the full 2D grid of patching effects.

    Attributes:
        component: Component name ("attn_out" or "mlp_out")
        layers: List of layers analyzed
        positions: List of positions analyzed
        denoising_grid: [n_layers, n_positions] array
        noising_grid: [n_layers, n_positions] array
    """

    component: str
    layers: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    denoising_grid: np.ndarray | None = None
    noising_grid: np.ndarray | None = None

    def to_dict(self, **kwargs) -> dict:
        """Custom serialization to handle numpy arrays."""
        d = {
            "component": self.component,
            "layers": self.layers,
            "positions": self.positions,
        }
        if self.denoising_grid is not None:
            d["denoising_grid"] = self.denoising_grid.tolist()
        if self.noising_grid is not None:
            d["noising_grid"] = self.noising_grid.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LayerPositionResult":
        """Custom deserialization to handle numpy arrays."""
        result = cls(
            component=d.get("component", ""),
            layers=d.get("layers", []),
            positions=d.get("positions", []),
        )
        if "denoising_grid" in d and d["denoising_grid"] is not None:
            grid = d["denoising_grid"]
            # Handle string representation (legacy format)
            if isinstance(grid, str):
                grid = np.fromstring(grid.replace("[", "").replace("]", "").replace("\n", " "), sep=" ")
                n_layers = len(d.get("layers", []))
                n_pos = len(d.get("positions", []))
                if n_layers > 0 and n_pos > 0:
                    grid = grid.reshape(n_layers, n_pos)
            result.denoising_grid = np.array(grid, dtype=float)
        if "noising_grid" in d and d["noising_grid"] is not None:
            grid = d["noising_grid"]
            # Handle string representation (legacy format)
            if isinstance(grid, str):
                grid = np.fromstring(grid.replace("[", "").replace("]", "").replace("\n", " "), sep=" ")
                n_layers = len(d.get("layers", []))
                n_pos = len(d.get("positions", []))
                if n_layers > 0 and n_pos > 0:
                    grid = grid.reshape(n_layers, n_pos)
            result.noising_grid = np.array(grid, dtype=float)
        return result


@dataclass
class FineResults(BaseSchema):
    """Aggregated results from fine-grained analyses.

    Contains path patching and multi-site interaction results.

    NOTE: Layer-position patching is now in attn (for attn_out) and mlp (for mlp_out).

    Attributes:
        sample_id: Sample identifier
        path_to_mlp: Head-to-MLP path patching results
        path_to_head: Head-to-head path patching results
        cross_layer_paths: Cross-layer head-to-head path patching
        multi_site: Multi-site interaction results
    """

    sample_id: int | None = None

    # Path patching results
    path_to_mlp: list[PathPatchingResult] = field(default_factory=list)
    path_to_head: list[PathPatchingResult] = field(default_factory=list)
    cross_layer_paths: list[PathPatchingResult] = field(default_factory=list)

    # Multi-site interaction
    multi_site: list[MultiSiteResult] = field(default_factory=list)

    # Metadata
    n_layers: int = 0
    n_heads: int = 0

    def print_summary(self) -> None:
        """Print summary of results."""
        print(f"[fine] Sample {self.sample_id}")
        if self.path_to_mlp:
            print(f"  Path to MLP: {len(self.path_to_mlp)} paths")
        if self.path_to_head:
            print(f"  Path to head: {len(self.path_to_head)} paths")
        if self.cross_layer_paths:
            print(f"  Cross-layer paths: {len(self.cross_layer_paths)} paths")
        if self.multi_site:
            print(f"  Multi-site: {len(self.multi_site)} interactions")

    def to_dict(self, **kwargs) -> dict:
        """Custom serialization to handle nested dataclasses."""
        d = {
            "sample_id": self.sample_id,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "path_to_mlp": [r.to_dict() for r in self.path_to_mlp],
            "path_to_head": [r.to_dict() for r in self.path_to_head],
            "cross_layer_paths": [r.to_dict() for r in self.cross_layer_paths],
            "multi_site": [r.to_dict() for r in self.multi_site],
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FineResults":
        """Custom deserialization to handle nested dataclasses."""
        return cls(
            sample_id=d.get("sample_id"),
            n_layers=d.get("n_layers", 0),
            n_heads=d.get("n_heads", 0),
            path_to_mlp=[PathPatchingResult.from_dict(r) for r in d.get("path_to_mlp", [])],
            path_to_head=[PathPatchingResult.from_dict(r) for r in d.get("path_to_head", [])],
            cross_layer_paths=[PathPatchingResult.from_dict(r) for r in d.get("cross_layer_paths", [])],
            multi_site=[MultiSiteResult.from_dict(r) for r in d.get("multi_site", [])],
        )


@dataclass
class FineAggregatedResults(BaseSchema):
    """Aggregated fine-grained analysis results across all pairs."""

    pair_results: list[FineResults] = field(default_factory=list)

    @property
    def n_pairs(self) -> int:
        return len(self.pair_results)

    def add(self, result: FineResults) -> None:
        """Add a pair result."""
        self.pair_results.append(result)

    def print_summary(self) -> None:
        """Print summary of aggregated results."""
        print(f"[fine] Aggregated: {self.n_pairs} pairs")
        total_path_to_mlp = sum(len(r.path_to_mlp) for r in self.pair_results)
        total_path_to_head = sum(len(r.path_to_head) for r in self.pair_results)
        total_cross_layer = sum(len(r.cross_layer_paths) for r in self.pair_results)
        total_multi_site = sum(len(r.multi_site) for r in self.pair_results)
        if total_path_to_mlp:
            print(f"  Total path to MLP: {total_path_to_mlp}")
        if total_path_to_head:
            print(f"  Total path to head: {total_path_to_head}")
        if total_cross_layer:
            print(f"  Total cross-layer paths: {total_cross_layer}")
        if total_multi_site:
            print(f"  Total multi-site: {total_multi_site}")

    def to_dict(self, **kwargs) -> dict:
        """Custom serialization."""
        return {
            "pair_results": [r.to_dict() for r in self.pair_results],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FineAggregatedResults":
        """Custom deserialization."""
        return cls(
            pair_results=[FineResults.from_dict(r) for r in d.get("pair_results", [])],
        )
