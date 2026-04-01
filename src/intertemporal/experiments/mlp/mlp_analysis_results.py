"""Result dataclasses for MLP neuron analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ....common.base_schema import BaseSchema
from ....common.file_io import save_json
from ..fine.fine_results import LayerPositionResult


@dataclass
class NeuronInfo(BaseSchema):
    """Information about a single neuron."""

    neuron_idx: int
    activation_diff: float = 0.0  # clean - corrupted activation
    clean_activation: float = 0.0
    corrupted_activation: float = 0.0
    logit_contribution: float = 0.0  # activation_diff * W_out @ logit_direction
    w_out_logit_alignment: float = 0.0  # W_out[neuron] @ logit_direction (direction only)


@dataclass
class MLPNeuronLayerResult(BaseSchema):
    """Results for a single MLP layer."""

    layer: int
    n_neurons: int = 0

    # Top neurons by absolute activation difference
    top_neurons: list[NeuronInfo] = field(default_factory=list)

    # Aggregate statistics
    total_logit_contribution: float = 0.0  # Sum of all neuron contributions
    top_k_contribution_frac: float = 0.0  # Fraction from top-k neurons
    sparsity_ratio: float = 0.0  # top-k / total (higher = more sparse)

    # Neuron counts by direction
    n_positive_contributors: int = 0  # Push toward clean choice
    n_negative_contributors: int = 0  # Push toward corrupted choice

    def get_top_neuron_indices(self, n: int = 20) -> list[int]:
        """Get indices of top-n neurons by contribution magnitude."""
        return [n.neuron_idx for n in self.top_neurons[:n]]


@dataclass
class MLPPairResult(BaseSchema):
    """MLP analysis results for a single pair.

    Attributes:
        pair_idx: Pair index
        position_results: Per-position layer results (format_pos name -> layer results)
            Keys can be "time_horizon:0", "time_horizon", "response_choice", etc.
        neuron_activations: Per-neuron activation tracking for interpretability
        layer_position: Layer x position patching results (mlp_out component)
    """

    pair_idx: int = 0
    position_results: dict[str, list[MLPNeuronLayerResult]] = field(default_factory=dict)

    # Per-neuron max-activation tracking (for interpretability)
    # Maps (layer, neuron_idx) -> activation value for this pair
    neuron_activations: dict[str, float] = field(default_factory=dict)

    # Layer x position patching results
    layer_position: LayerPositionResult | None = None

    @property
    def positions_analyzed(self) -> int:
        """Number of positions analyzed."""
        return len(self.position_results)

    def get_layer_result(
        self, format_pos: str | None, layer: int
    ) -> MLPNeuronLayerResult | None:
        """Get results for a specific position and layer.

        Args:
            format_pos: Position name (e.g., "response_choice"). If None, uses first position.
            layer: Layer index

        Returns:
            MLPNeuronLayerResult or None if not found
        """
        if format_pos is None:
            if not self.position_results:
                return None
            format_pos = next(iter(self.position_results.keys()))

        if format_pos not in self.position_results:
            return None

        for r in self.position_results[format_pos]:
            if r.layer == layer:
                return r
        return None

    def get_all_top_neurons(
        self, format_pos: str | None = None, n_per_layer: int = 10
    ) -> list[tuple[int, int, float]]:
        """Get top neurons across all layers, optionally filtered by position.

        Args:
            format_pos: Position to get neurons for. If None, uses first position.
            n_per_layer: Number of top neurons per layer

        Returns:
            List of (layer, neuron_idx, contribution) tuples, sorted by |contribution|
        """
        if format_pos is None:
            if not self.position_results:
                return []
            format_pos = next(iter(self.position_results.keys()))

        if format_pos not in self.position_results:
            return []

        neurons = []
        for lr in self.position_results[format_pos]:
            for ni in lr.top_neurons[:n_per_layer]:
                neurons.append((lr.layer, ni.neuron_idx, ni.logit_contribution))
        return sorted(neurons, key=lambda x: abs(x[2]), reverse=True)

    def get_all_format_positions(self) -> list[str]:
        """Get all format_pos names analyzed."""
        return list(self.position_results.keys())


@dataclass
class MLPAggregatedResults(BaseSchema):
    """Aggregated MLP analysis across all pairs."""

    pair_results: list[MLPPairResult] = field(default_factory=list)

    # Configuration used
    layers_analyzed: list[int] = field(default_factory=list)
    n_top_neurons: int = 50

    @property
    def n_pairs(self) -> int:
        return len(self.pair_results)

    def add(self, result: MLPPairResult) -> None:
        """Add a pair result."""
        self.pair_results.append(result)

    def get_all_positions(self) -> set[str]:
        """Get all format_pos names analyzed across all pairs."""
        positions = set()
        for pr in self.pair_results:
            positions.update(pr.position_results.keys())
        return positions

    def get_mean_sparsity(
        self, layer: int, format_pos: str | None = None
    ) -> float:
        """Get mean sparsity ratio for a layer across pairs.

        Args:
            layer: Layer index
            format_pos: Position to analyze (None = first available)
        """
        values = []
        for pr in self.pair_results:
            lr = pr.get_layer_result(format_pos, layer)
            if lr:
                values.append(lr.sparsity_ratio)
        return float(np.mean(values)) if values else 0.0

    def get_consistent_neurons(
        self,
        layer: int,
        format_pos: str | None = None,
        min_pairs: int = 5,
        top_n: int = 20,
    ) -> list[tuple[int, int, float]]:
        """Find neurons that appear in top-n across multiple pairs.

        Args:
            layer: Layer to analyze
            format_pos: Position to analyze (None = first available)
            min_pairs: Minimum number of pairs a neuron must appear in
            top_n: Consider top-n neurons per pair

        Returns:
            List of (neuron_idx, n_pairs_appeared, mean_contribution)
        """
        neuron_counts: dict[int, list[float]] = {}

        for pr in self.pair_results:
            lr = pr.get_layer_result(format_pos, layer)
            if not lr:
                continue
            for ni in lr.top_neurons[:top_n]:
                if ni.neuron_idx not in neuron_counts:
                    neuron_counts[ni.neuron_idx] = []
                neuron_counts[ni.neuron_idx].append(ni.logit_contribution)

        results = []
        for neuron_idx, contributions in neuron_counts.items():
            if len(contributions) >= min_pairs:
                results.append(
                    (neuron_idx, len(contributions), float(np.mean(contributions)))
                )

        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_neuron_activation_across_pairs(
        self, layer: int, neuron_idx: int, format_pos: str | None = None
    ) -> list[tuple[int, float, float]]:
        """Get activation values for a neuron across all pairs.

        Args:
            layer: Layer index
            neuron_idx: Neuron index
            format_pos: Position to analyze (None = first available)

        Returns:
            List of (pair_idx, clean_activation, corrupted_activation)
        """
        results = []
        for pr in self.pair_results:
            lr = pr.get_layer_result(format_pos, layer)
            if not lr:
                continue
            for ni in lr.top_neurons:
                if ni.neuron_idx == neuron_idx:
                    results.append(
                        (pr.pair_idx, ni.clean_activation, ni.corrupted_activation)
                    )
                    break
        return results

    def save(self, output_dir: Path) -> None:
        """Save results to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.to_dict(), output_dir / "mlp_analysis_agg.json")

    @classmethod
    def load(cls, output_dir: Path) -> "MLPAggregatedResults | None":
        """Load from JSON."""
        path = Path(output_dir) / "mlp_analysis_agg.json"
        if path.exists():
            return cls.from_json(path)
        return None

    def filter_by_indices(self, indices: list[int]) -> "MLPAggregatedResults":
        """Create a filtered copy containing only the specified pair indices."""
        filtered = MLPAggregatedResults(
            layers_analyzed=self.layers_analyzed,
            n_top_neurons=self.n_top_neurons,
        )
        for idx in indices:
            if idx < len(self.pair_results):
                filtered.pair_results.append(self.pair_results[idx])
        return filtered

    def print_summary(self, format_pos: str | None = None) -> None:
        """Print summary of MLP analysis.

        Args:
            format_pos: Position to summarize (None = first available)
        """
        positions = sorted(self.get_all_positions())
        print(f"[mlp] MLP Analysis: {self.n_pairs} pairs, layers {self.layers_analyzed}")
        print(f"  Positions: {positions}")

        if format_pos is None and positions:
            format_pos = positions[0]

        if format_pos:
            print(f"  Summary for position: {format_pos}")

        for layer in self.layers_analyzed:
            # Compute mean logit contribution for this layer
            contribs = []
            for pr in self.pair_results:
                lr = pr.get_layer_result(format_pos, layer)
                if lr:
                    contribs.append(lr.total_logit_contribution)

            if contribs:
                mean_contrib = sum(contribs) / len(contribs)
                n_positive = sum(1 for c in contribs if c > 0)
                n_negative = sum(1 for c in contribs if c < 0)
                # Show signed contribution with direction indicator
                direction = "+" if mean_contrib > 0 else "-"
                print(f"  L{layer}: mean_contrib={direction}{abs(mean_contrib):.3f}, pos={n_positive}, neg={n_negative}")
            else:
                print(f"  L{layer}: no data")
