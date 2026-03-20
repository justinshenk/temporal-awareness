"""Result dataclasses for difference-in-means analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ....common.base_schema import BaseSchema
from ....common.file_io import save_json


@dataclass
class DiffMeansLayerResult(BaseSchema):
    """Difference-in-means results for a single layer.

    Attributes:
        layer: Layer index
        cosine_to_next: Cosine similarity to next layer's diff direction
        cosine_to_prev: Cosine similarity to previous layer's diff direction
        diff_norm: L2 norm of the difference vector
        attn_rotation_angle: Angle (degrees) of rotation from attention
        mlp_rotation_angle: Angle (degrees) of rotation from MLP
        total_rotation_angle: Total rotation angle from previous layer
        cosine_to_logit: Cosine similarity to logit direction
        cosine_to_initial: Cosine similarity to initial layer's diff direction (cumulative drift)
        attn_out_diff_norm: L2 norm of attention output difference
        mlp_out_diff_norm: L2 norm of MLP output difference
    """
    layer: int
    cosine_to_next: float | None = None
    cosine_to_prev: float | None = None
    diff_norm: float = 0.0
    attn_rotation_angle: float | None = None
    mlp_rotation_angle: float | None = None
    total_rotation_angle: float | None = None
    cosine_to_logit: float | None = None
    cosine_to_initial: float | None = None
    attn_out_diff_norm: float | None = None
    mlp_out_diff_norm: float | None = None


@dataclass
class SVDResult(BaseSchema):
    """SVD analysis results for a layer.

    Attributes:
        layer: Layer index
        singular_values: Top-k singular values (normalized)
        explained_variance_ratio: Variance explained by each component
        effective_rank: Effective dimensionality (sum of normalized singular values squared)
        top1_ratio: Ratio of top singular value to sum (1.0 = rank-1)
    """
    layer: int
    singular_values: list[float] = field(default_factory=list)
    explained_variance_ratio: list[float] = field(default_factory=list)
    effective_rank: float = 0.0
    top1_ratio: float = 0.0


@dataclass
class DiffMeansPairResult(BaseSchema):
    """Difference-in-means results for a single pair.

    Attributes:
        pair_idx: Pair index
        layer_results: Results per layer (for primary position)
        positions_analyzed: Number of positions analyzed
        position_results: Per-position layer results (position -> layer results)
        primary_position: The main position analyzed (e.g., last token)
    """
    pair_idx: int = 0
    layer_results: list[DiffMeansLayerResult] = field(default_factory=list)
    positions_analyzed: int = 0
    position_results: dict[int, list[DiffMeansLayerResult]] = field(default_factory=dict)
    primary_position: int | None = None

    def get_layer_result(self, layer: int) -> DiffMeansLayerResult | None:
        """Get result for a specific layer."""
        for r in self.layer_results:
            if r.layer == layer:
                return r
        return None

    def get_cosine_trajectory(self) -> tuple[list[int], list[float]]:
        """Get cosine similarity trajectory across layers."""
        layers = []
        cosines = []
        for r in self.layer_results:
            if r.cosine_to_next is not None:
                layers.append(r.layer)
                cosines.append(r.cosine_to_next)
        return layers, cosines

    def get_rotation_trajectory(self) -> dict[str, tuple[list[int], list[float]]]:
        """Get rotation angle trajectories."""
        layers = []
        attn_angles = []
        mlp_angles = []
        total_angles = []

        for r in self.layer_results:
            if r.total_rotation_angle is not None:
                layers.append(r.layer)
                attn_angles.append(r.attn_rotation_angle or 0.0)
                mlp_angles.append(r.mlp_rotation_angle or 0.0)
                total_angles.append(r.total_rotation_angle)

        return {
            "layers": (layers, layers),
            "attn": (layers, attn_angles),
            "mlp": (layers, mlp_angles),
            "total": (layers, total_angles),
        }

    def get_cosine_to_logit_trajectory(self) -> tuple[list[int], list[float]]:
        """Get cosine similarity to logit direction trajectory."""
        layers = []
        cosines = []
        for r in self.layer_results:
            if r.cosine_to_logit is not None:
                layers.append(r.layer)
                cosines.append(r.cosine_to_logit)
        return layers, cosines

    def get_cosine_to_initial_trajectory(self) -> tuple[list[int], list[float]]:
        """Get cosine similarity to initial direction (cumulative drift) trajectory."""
        layers = []
        cosines = []
        for r in self.layer_results:
            if r.cosine_to_initial is not None:
                layers.append(r.layer)
                cosines.append(r.cosine_to_initial)
        return layers, cosines

    def get_component_norm_trajectory(self) -> dict[str, tuple[list[int], list[float]]]:
        """Get attention and MLP output difference norm trajectories."""
        layers = []
        attn_norms = []
        mlp_norms = []

        for r in self.layer_results:
            if r.attn_out_diff_norm is not None and r.mlp_out_diff_norm is not None:
                layers.append(r.layer)
                attn_norms.append(r.attn_out_diff_norm)
                mlp_norms.append(r.mlp_out_diff_norm)

        return {
            "attn": (layers, attn_norms),
            "mlp": (layers, mlp_norms),
        }

    def get_position_diff_norm_trajectory(
        self, position: int
    ) -> tuple[list[int], list[float]]:
        """Get diff norm trajectory for a specific position."""
        if position not in self.position_results:
            return [], []

        layers = []
        norms = []
        for r in self.position_results[position]:
            layers.append(r.layer)
            norms.append(r.diff_norm)
        return layers, norms


@dataclass
class DiffMeansAggregatedResults(BaseSchema):
    """Aggregated difference-in-means results across pairs.

    Attributes:
        pair_results: Results per pair
        svd_results: SVD analysis per layer (computed across all pairs)
        n_pairs: Number of pairs
    """
    pair_results: list[DiffMeansPairResult] = field(default_factory=list)
    svd_results: list[SVDResult] = field(default_factory=list)
    n_pairs: int = 0

    def add(self, result: DiffMeansPairResult) -> None:
        """Add a pair result."""
        self.pair_results.append(result)
        self.n_pairs = len(self.pair_results)

    def get_mean_cosine_trajectory(self) -> tuple[list[int], list[float], list[float]]:
        """Get mean and std cosine similarity trajectory."""
        if not self.pair_results:
            return [], [], []

        # Collect all cosines per layer
        layer_cosines: dict[int, list[float]] = {}
        for pr in self.pair_results:
            for lr in pr.layer_results:
                if lr.cosine_to_next is not None:
                    if lr.layer not in layer_cosines:
                        layer_cosines[lr.layer] = []
                    layer_cosines[lr.layer].append(float(lr.cosine_to_next))

        layers = sorted(layer_cosines.keys())
        means = [float(np.mean(layer_cosines[l])) for l in layers]
        stds = [float(np.std(layer_cosines[l])) for l in layers]
        return layers, means, stds

    def get_mean_rotation_trajectory(self) -> dict[str, tuple[list[int], list[float], list[float]]]:
        """Get mean and std rotation trajectories."""
        if not self.pair_results:
            return {}

        # Collect rotations per layer
        layer_attn: dict[int, list[float]] = {}
        layer_mlp: dict[int, list[float]] = {}
        layer_total: dict[int, list[float]] = {}

        for pr in self.pair_results:
            for lr in pr.layer_results:
                if lr.total_rotation_angle is not None:
                    layer = lr.layer
                    if layer not in layer_attn:
                        layer_attn[layer] = []
                        layer_mlp[layer] = []
                        layer_total[layer] = []
                    layer_attn[layer].append(float(lr.attn_rotation_angle or 0.0))
                    layer_mlp[layer].append(float(lr.mlp_rotation_angle or 0.0))
                    layer_total[layer].append(float(lr.total_rotation_angle))

        layers = sorted(layer_attn.keys())
        return {
            "attn": (layers, [float(np.mean(layer_attn[l])) for l in layers], [float(np.std(layer_attn[l])) for l in layers]),
            "mlp": (layers, [float(np.mean(layer_mlp[l])) for l in layers], [float(np.std(layer_mlp[l])) for l in layers]),
            "total": (layers, [float(np.mean(layer_total[l])) for l in layers], [float(np.std(layer_total[l])) for l in layers]),
        }

    def get_svd_effective_rank_trajectory(self) -> tuple[list[int], list[float]]:
        """Get effective rank trajectory from SVD."""
        layers = [s.layer for s in self.svd_results]
        ranks = [s.effective_rank for s in self.svd_results]
        return layers, ranks

    def get_svd_top1_ratio_trajectory(self) -> tuple[list[int], list[float]]:
        """Get top-1 ratio trajectory (higher = more rank-1)."""
        layers = [s.layer for s in self.svd_results]
        ratios = [s.top1_ratio for s in self.svd_results]
        return layers, ratios

    def get_mean_cosine_to_logit_trajectory(
        self,
    ) -> tuple[list[int], list[float], list[float]]:
        """Get mean and std cosine to logit direction trajectory."""
        if not self.pair_results:
            return [], [], []

        layer_cosines: dict[int, list[float]] = {}
        for pr in self.pair_results:
            for lr in pr.layer_results:
                if lr.cosine_to_logit is not None:
                    if lr.layer not in layer_cosines:
                        layer_cosines[lr.layer] = []
                    layer_cosines[lr.layer].append(float(lr.cosine_to_logit))

        layers = sorted(layer_cosines.keys())
        means = [float(np.mean(layer_cosines[l])) for l in layers]
        stds = [float(np.std(layer_cosines[l])) for l in layers]
        return layers, means, stds

    def get_mean_cosine_to_initial_trajectory(
        self,
    ) -> tuple[list[int], list[float], list[float]]:
        """Get mean and std cosine to initial direction trajectory."""
        if not self.pair_results:
            return [], [], []

        layer_cosines: dict[int, list[float]] = {}
        for pr in self.pair_results:
            for lr in pr.layer_results:
                if lr.cosine_to_initial is not None:
                    if lr.layer not in layer_cosines:
                        layer_cosines[lr.layer] = []
                    layer_cosines[lr.layer].append(float(lr.cosine_to_initial))

        layers = sorted(layer_cosines.keys())
        means = [float(np.mean(layer_cosines[l])) for l in layers]
        stds = [float(np.std(layer_cosines[l])) for l in layers]
        return layers, means, stds

    def get_mean_component_norm_trajectory(
        self,
    ) -> dict[str, tuple[list[int], list[float], list[float]]]:
        """Get mean and std attention/MLP output difference norm trajectories."""
        if not self.pair_results:
            return {}

        layer_attn: dict[int, list[float]] = {}
        layer_mlp: dict[int, list[float]] = {}

        for pr in self.pair_results:
            for lr in pr.layer_results:
                if lr.attn_out_diff_norm is not None:
                    if lr.layer not in layer_attn:
                        layer_attn[lr.layer] = []
                    layer_attn[lr.layer].append(float(lr.attn_out_diff_norm))
                if lr.mlp_out_diff_norm is not None:
                    if lr.layer not in layer_mlp:
                        layer_mlp[lr.layer] = []
                    layer_mlp[lr.layer].append(float(lr.mlp_out_diff_norm))

        layers = sorted(layer_attn.keys())
        return {
            "attn": (
                layers,
                [float(np.mean(layer_attn[l])) for l in layers],
                [float(np.std(layer_attn[l])) for l in layers],
            ),
            "mlp": (
                layers,
                [float(np.mean(layer_mlp[l])) for l in layers],
                [float(np.std(layer_mlp[l])) for l in layers],
            ),
        }

    def get_mean_diff_norm_trajectory(
        self,
    ) -> tuple[list[int], list[float], list[float]]:
        """Get mean and std difference norm trajectory."""
        if not self.pair_results:
            return [], [], []

        layer_norms: dict[int, list[float]] = {}
        for pr in self.pair_results:
            for lr in pr.layer_results:
                if lr.layer not in layer_norms:
                    layer_norms[lr.layer] = []
                layer_norms[lr.layer].append(float(lr.diff_norm))

        layers = sorted(layer_norms.keys())
        means = [float(np.mean(layer_norms[l])) for l in layers]
        stds = [float(np.std(layer_norms[l])) for l in layers]
        return layers, means, stds

    def get_mean_position_diff_norm_trajectory(
        self, position: int
    ) -> tuple[list[int], list[float], list[float]]:
        """Get mean and std diff norm trajectory for a specific position."""
        if not self.pair_results:
            return [], [], []

        layer_norms: dict[int, list[float]] = {}
        for pr in self.pair_results:
            if position in pr.position_results:
                for lr in pr.position_results[position]:
                    if lr.layer not in layer_norms:
                        layer_norms[lr.layer] = []
                    layer_norms[lr.layer].append(float(lr.diff_norm))

        if not layer_norms:
            return [], [], []

        layers = sorted(layer_norms.keys())
        means = [float(np.mean(layer_norms[l])) for l in layers]
        stds = [float(np.std(layer_norms[l])) for l in layers]
        return layers, means, stds

    def get_all_analyzed_positions(self) -> set[int]:
        """Get all positions that have been analyzed across all pairs."""
        positions = set()
        for pr in self.pair_results:
            positions.update(pr.position_results.keys())
        return positions

    def save(self, output_dir: Path) -> None:
        """Save results to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.to_dict(), output_dir / "diffmeans_agg.json")

    @classmethod
    def load(cls, output_dir: Path) -> "DiffMeansAggregatedResults | None":
        """Load results from directory."""
        path = Path(output_dir) / "diffmeans_agg.json"
        if path.exists():
            return cls.from_json(path)
        return None

    def print_summary(self) -> None:
        """Print summary of results."""
        print(f"[diffmeans] Aggregated results: {self.n_pairs} pairs")
        if self.svd_results:
            ranks = [s.effective_rank for s in self.svd_results]
            print(f"  Effective rank range: {min(ranks):.2f} - {max(ranks):.2f}")
            ratios = [s.top1_ratio for s in self.svd_results]
            print(f"  Top-1 ratio range: {min(ratios):.3f} - {max(ratios):.3f}")
