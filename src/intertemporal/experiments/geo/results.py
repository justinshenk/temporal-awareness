"""Result dataclasses for geometric (PCA) analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ....common.base_schema import BaseSchema
from ....common.file_io import save_json


@dataclass
class GeoPCALayerResult(BaseSchema):
    """PCA results for a single layer.

    Attributes:
        layer: Layer index
        explained_variance_ratio: Variance explained by each PC (top N)
        clean_mean_pc: Mean PC coordinates for clean samples [n_components]
        corrupted_mean_pc: Mean PC coordinates for corrupted samples [n_components]
        separation_distance: Euclidean distance between clean/corrupted means in PC space
        separation_pc1: Signed distance along PC1 (clean - corrupted)
        logit_diff_alignment: Cosine similarity between PC1 and logit diff direction
    """
    layer: int
    explained_variance_ratio: list[float] = field(default_factory=list)
    clean_mean_pc: list[float] = field(default_factory=list)
    corrupted_mean_pc: list[float] = field(default_factory=list)
    separation_distance: float = 0.0
    separation_pc1: float = 0.0
    logit_diff_alignment: float | None = None


@dataclass
class GeoPCAPositionResult(BaseSchema):
    """PCA results for a single position across all layers.

    Attributes:
        position: Position index
        position_label: Optional human-readable position label
        layer_results: PCA results per layer
    """
    position: int
    position_label: str | None = None
    layer_results: list[GeoPCALayerResult] = field(default_factory=list)

    def get_separation_trajectory(self) -> tuple[list[int], list[float]]:
        """Get separation distance trajectory across layers."""
        layers = [r.layer for r in self.layer_results]
        distances = [r.separation_distance for r in self.layer_results]
        return layers, distances

    def get_variance_trajectory(self, pc_idx: int = 0) -> tuple[list[int], list[float]]:
        """Get explained variance ratio trajectory for a specific PC."""
        layers = []
        ratios = []
        for r in self.layer_results:
            if pc_idx < len(r.explained_variance_ratio):
                layers.append(r.layer)
                ratios.append(r.explained_variance_ratio[pc_idx])
        return layers, ratios


@dataclass
class GeoPairResult(BaseSchema):
    """Geometric analysis results for a single pair.

    Attributes:
        pair_idx: Pair index
        position_results: PCA results per position
    """
    pair_idx: int = 0
    position_results: list[GeoPCAPositionResult] = field(default_factory=list)

    def get_position_result(self, position: int) -> GeoPCAPositionResult | None:
        """Get result for a specific position."""
        for r in self.position_results:
            if r.position == position:
                return r
        return None


@dataclass
class GeoAggregatedResults(BaseSchema):
    """Aggregated geometric analysis results across pairs.

    Attributes:
        pair_results: Results per pair
        n_pairs: Number of pairs
        positions_analyzed: List of positions that were analyzed
        n_components: Number of PCA components tracked
    """
    pair_results: list[GeoPairResult] = field(default_factory=list)
    n_pairs: int = 0
    positions_analyzed: list[int] = field(default_factory=list)
    n_components: int = 3

    def add(self, result: GeoPairResult) -> None:
        """Add a pair result."""
        self.pair_results.append(result)
        self.n_pairs = len(self.pair_results)
        # Update positions_analyzed from the result
        for pos_result in result.position_results:
            if pos_result.position not in self.positions_analyzed:
                self.positions_analyzed.append(pos_result.position)
        self.positions_analyzed.sort()

    def get_mean_separation_trajectory(
        self, position: int
    ) -> tuple[list[int], list[float], list[float]]:
        """Get mean and std separation distance trajectory for a position.

        Returns:
            Tuple of (layers, means, stds)
        """
        if not self.pair_results:
            return [], [], []

        # Collect separations per layer
        layer_separations: dict[int, list[float]] = {}
        for pr in self.pair_results:
            pos_result = pr.get_position_result(position)
            if pos_result:
                for lr in pos_result.layer_results:
                    if lr.layer not in layer_separations:
                        layer_separations[lr.layer] = []
                    layer_separations[lr.layer].append(lr.separation_distance)

        layers = sorted(layer_separations.keys())
        means = [float(np.mean(layer_separations[l])) for l in layers]
        stds = [float(np.std(layer_separations[l])) for l in layers]
        return layers, means, stds

    def get_mean_variance_trajectory(
        self, position: int, pc_idx: int = 0
    ) -> tuple[list[int], list[float], list[float]]:
        """Get mean and std explained variance ratio trajectory.

        Returns:
            Tuple of (layers, means, stds)
        """
        if not self.pair_results:
            return [], [], []

        # Collect variances per layer
        layer_variances: dict[int, list[float]] = {}
        for pr in self.pair_results:
            pos_result = pr.get_position_result(position)
            if pos_result:
                for lr in pos_result.layer_results:
                    if pc_idx < len(lr.explained_variance_ratio):
                        if lr.layer not in layer_variances:
                            layer_variances[lr.layer] = []
                        layer_variances[lr.layer].append(lr.explained_variance_ratio[pc_idx])

        layers = sorted(layer_variances.keys())
        # Filter out string NaN values and compute stats
        means = []
        stds = []
        for l in layers:
            values = [v for v in layer_variances[l] if isinstance(v, (int, float)) and not np.isnan(v)]
            if values:
                means.append(float(np.mean(values)))
                stds.append(float(np.std(values)))
            else:
                means.append(0.0)
                stds.append(0.0)
        return layers, means, stds

    def save(self, output_dir: Path) -> None:
        """Save results to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.to_dict(), output_dir / "geo_agg.json")

    @classmethod
    def load(cls, output_dir: Path) -> "GeoAggregatedResults | None":
        """Load results from directory."""
        path = Path(output_dir) / "geo_agg.json"
        if path.exists():
            return cls.from_json(path)
        return None

    def print_summary(self) -> None:
        """Print summary of results."""
        print(f"[geo] Aggregated results: {self.n_pairs} pairs")
        print(f"  Positions analyzed: {self.positions_analyzed}")
        print(f"  Components tracked: {self.n_components}")
