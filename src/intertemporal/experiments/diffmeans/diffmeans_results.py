"""Result dataclasses for difference-in-means analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ....common.base_schema import BaseSchema
from ....common.file_io import save_json


@dataclass
class PositionPair(BaseSchema):
    """A pair of corresponding positions in clean and corrupted trajectories.

    Attributes:
        clean_pos: Position index in clean trajectory
        corrupted_pos: Position index in corrupted trajectory
    """
    clean_pos: int
    corrupted_pos: int


@dataclass
class ResolvedPositions(BaseSchema):
    """Resolved semantic positions to absolute position pairs.

    Attributes:
        positions: Dict mapping format_pos name -> list of position pairs
    """
    positions: dict[str, list[PositionPair]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.positions)

    def __iter__(self):
        return iter(self.positions.items())

    def __getitem__(self, format_pos: str) -> list[PositionPair]:
        return self.positions[format_pos]

    def __contains__(self, format_pos: str) -> bool:
        return format_pos in self.positions

    def keys(self):
        return self.positions.keys()

    def items(self):
        return self.positions.items()

    def get(self, format_pos: str, default: list[PositionPair] | None = None) -> list[PositionPair] | None:
        return self.positions.get(format_pos, default)


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
        position_results: Per-position layer results (format_pos name -> layer results)
    """
    pair_idx: int = 0
    position_results: dict[str, list[DiffMeansLayerResult]] = field(default_factory=dict)

    @property
    def positions_analyzed(self) -> int:
        """Number of positions analyzed."""
        return len(self.position_results)

    def get_layer_result(self, format_pos: str, layer: int) -> DiffMeansLayerResult | None:
        """Get result for a specific position and layer."""
        if format_pos not in self.position_results:
            return None
        for r in self.position_results[format_pos]:
            if r.layer == layer:
                return r
        return None

    def get_cosine_trajectory(self, format_pos: str) -> tuple[list[int], list[float]]:
        """Get cosine similarity trajectory across layers for a position."""
        if format_pos not in self.position_results:
            return [], []
        layers = []
        cosines = []
        for r in self.position_results[format_pos]:
            if r.cosine_to_next is not None:
                layers.append(r.layer)
                cosines.append(r.cosine_to_next)
        return layers, cosines

    def get_rotation_trajectory(self, format_pos: str) -> dict[str, tuple[list[int], list[float]]]:
        """Get rotation angle trajectories for a position."""
        if format_pos not in self.position_results:
            return {}
        layers = []
        attn_angles = []
        mlp_angles = []
        total_angles = []

        for r in self.position_results[format_pos]:
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

    def get_cosine_to_logit_trajectory(self, format_pos: str) -> tuple[list[int], list[float]]:
        """Get cosine similarity to logit direction trajectory for a position."""
        if format_pos not in self.position_results:
            return [], []
        layers = []
        cosines = []
        for r in self.position_results[format_pos]:
            if r.cosine_to_logit is not None:
                layers.append(r.layer)
                cosines.append(r.cosine_to_logit)
        return layers, cosines

    def get_cosine_to_initial_trajectory(self, format_pos: str) -> tuple[list[int], list[float]]:
        """Get cosine similarity to initial direction (cumulative drift) trajectory."""
        if format_pos not in self.position_results:
            return [], []
        layers = []
        cosines = []
        for r in self.position_results[format_pos]:
            if r.cosine_to_initial is not None:
                layers.append(r.layer)
                cosines.append(r.cosine_to_initial)
        return layers, cosines

    def get_component_norm_trajectory(self, format_pos: str) -> dict[str, tuple[list[int], list[float]]]:
        """Get attention and MLP output difference norm trajectories for a position."""
        if format_pos not in self.position_results:
            return {}
        layers = []
        attn_norms = []
        mlp_norms = []

        for r in self.position_results[format_pos]:
            if r.attn_out_diff_norm is not None and r.mlp_out_diff_norm is not None:
                layers.append(r.layer)
                attn_norms.append(r.attn_out_diff_norm)
                mlp_norms.append(r.mlp_out_diff_norm)

        return {
            "attn": (layers, attn_norms),
            "mlp": (layers, mlp_norms),
        }

    def get_diff_norm_trajectory(self, format_pos: str) -> tuple[list[int], list[float]]:
        """Get diff norm trajectory for a specific position."""
        if format_pos not in self.position_results:
            return [], []

        layers = []
        norms = []
        for r in self.position_results[format_pos]:
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

    def get_all_format_positions(self) -> set[str]:
        """Get all format_pos names analyzed across all pairs."""
        positions = set()
        for pr in self.pair_results:
            positions.update(pr.position_results.keys())
        return positions

    def _collect_layer_values(
        self, format_pos: str, attr: str
    ) -> dict[int, list[float]]:
        """Collect values for an attribute across all pairs for a position."""
        layer_values: dict[int, list[float]] = {}
        for pr in self.pair_results:
            if format_pos not in pr.position_results:
                continue
            for lr in pr.position_results[format_pos]:
                val = getattr(lr, attr, None)
                if val is not None:
                    if lr.layer not in layer_values:
                        layer_values[lr.layer] = []
                    layer_values[lr.layer].append(float(val))
        return layer_values

    def get_mean_cosine_trajectory(
        self, format_pos: str
    ) -> tuple[list[int], list[float], list[float]]:
        """Get mean and std cosine similarity trajectory for a position."""
        layer_cosines = self._collect_layer_values(format_pos, "cosine_to_next")
        if not layer_cosines:
            return [], [], []
        layers = sorted(layer_cosines.keys())
        means = [float(np.mean(layer_cosines[l])) for l in layers]
        stds = [float(np.std(layer_cosines[l])) for l in layers]
        return layers, means, stds

    def get_mean_rotation_trajectory(
        self, format_pos: str
    ) -> dict[str, tuple[list[int], list[float], list[float]]]:
        """Get mean and std rotation trajectories for a position."""
        layer_attn = self._collect_layer_values(format_pos, "attn_rotation_angle")
        layer_mlp = self._collect_layer_values(format_pos, "mlp_rotation_angle")
        layer_total = self._collect_layer_values(format_pos, "total_rotation_angle")
        if not layer_total:
            return {}
        layers = sorted(layer_total.keys())
        return {
            "attn": (layers, [float(np.mean(layer_attn.get(l, [0]))) for l in layers], [float(np.std(layer_attn.get(l, [0]))) for l in layers]),
            "mlp": (layers, [float(np.mean(layer_mlp.get(l, [0]))) for l in layers], [float(np.std(layer_mlp.get(l, [0]))) for l in layers]),
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
        self, format_pos: str
    ) -> tuple[list[int], list[float], list[float]]:
        """Get mean and std cosine to logit direction trajectory for a position."""
        layer_cosines = self._collect_layer_values(format_pos, "cosine_to_logit")
        if not layer_cosines:
            return [], [], []
        layers = sorted(layer_cosines.keys())
        means = [float(np.mean(layer_cosines[l])) for l in layers]
        stds = [float(np.std(layer_cosines[l])) for l in layers]
        return layers, means, stds

    def get_mean_cosine_to_initial_trajectory(
        self, format_pos: str
    ) -> tuple[list[int], list[float], list[float]]:
        """Get mean and std cosine to initial direction trajectory for a position."""
        layer_cosines = self._collect_layer_values(format_pos, "cosine_to_initial")
        if not layer_cosines:
            return [], [], []
        layers = sorted(layer_cosines.keys())
        means = [float(np.mean(layer_cosines[l])) for l in layers]
        stds = [float(np.std(layer_cosines[l])) for l in layers]
        return layers, means, stds

    def get_mean_component_norm_trajectory(
        self, format_pos: str
    ) -> dict[str, tuple[list[int], list[float], list[float]]]:
        """Get mean and std attention/MLP output difference norm trajectories."""
        layer_attn = self._collect_layer_values(format_pos, "attn_out_diff_norm")
        layer_mlp = self._collect_layer_values(format_pos, "mlp_out_diff_norm")
        if not layer_attn:
            return {}
        layers = sorted(layer_attn.keys())
        return {
            "attn": (
                layers,
                [float(np.mean(layer_attn[l])) for l in layers],
                [float(np.std(layer_attn[l])) for l in layers],
            ),
            "mlp": (
                layers,
                [float(np.mean(layer_mlp.get(l, [0]))) for l in layers],
                [float(np.std(layer_mlp.get(l, [0]))) for l in layers],
            ),
        }

    def get_mean_diff_norm_trajectory(
        self, format_pos: str
    ) -> tuple[list[int], list[float], list[float]]:
        """Get mean and std difference norm trajectory for a position."""
        layer_norms = self._collect_layer_values(format_pos, "diff_norm")
        if not layer_norms:
            return [], [], []
        layers = sorted(layer_norms.keys())
        means = [float(np.mean(layer_norms[l])) for l in layers]
        stds = [float(np.std(layer_norms[l])) for l in layers]
        return layers, means, stds

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
        print(f"  Format positions: {sorted(self.get_all_format_positions())}")
        if self.svd_results:
            ranks = [s.effective_rank for s in self.svd_results]
            print(f"  Effective rank range: {min(ranks):.2f} - {max(ranks):.2f}")
            ratios = [s.top1_ratio for s in self.svd_results]
            print(f"  Top-1 ratio range: {min(ratios):.3f} - {max(ratios):.3f}")
