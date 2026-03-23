"""Result dataclasses for MLP neuron analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ....common.base_schema import BaseSchema
from ....common.file_io import save_json


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
    """MLP analysis results for a single pair."""

    pair_idx: int = 0
    position: int = 0  # Position analyzed (typically P_dest)

    layer_results: list[MLPNeuronLayerResult] = field(default_factory=list)

    # Per-neuron max-activation tracking (for interpretability)
    # Maps (layer, neuron_idx) -> activation value for this pair
    neuron_activations: dict[str, float] = field(default_factory=dict)

    def get_layer_result(self, layer: int) -> MLPNeuronLayerResult | None:
        """Get results for a specific layer."""
        for r in self.layer_results:
            if r.layer == layer:
                return r
        return None

    def get_all_top_neurons(self, n_per_layer: int = 10) -> list[tuple[int, int, float]]:
        """Get top neurons across all layers.

        Returns:
            List of (layer, neuron_idx, contribution) tuples, sorted by |contribution|
        """
        neurons = []
        for lr in self.layer_results:
            for ni in lr.top_neurons[:n_per_layer]:
                neurons.append((lr.layer, ni.neuron_idx, ni.logit_contribution))
        return sorted(neurons, key=lambda x: abs(x[2]), reverse=True)


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

    def get_mean_sparsity(self, layer: int) -> float:
        """Get mean sparsity ratio for a layer across pairs."""
        values = []
        for pr in self.pair_results:
            lr = pr.get_layer_result(layer)
            if lr:
                values.append(lr.sparsity_ratio)
        return float(np.mean(values)) if values else 0.0

    def get_consistent_neurons(
        self, layer: int, min_pairs: int = 5, top_n: int = 20
    ) -> list[tuple[int, int, float]]:
        """Find neurons that appear in top-n across multiple pairs.

        Args:
            layer: Layer to analyze
            min_pairs: Minimum number of pairs a neuron must appear in
            top_n: Consider top-n neurons per pair

        Returns:
            List of (neuron_idx, n_pairs_appeared, mean_contribution)
        """
        neuron_counts: dict[int, list[float]] = {}

        for pr in self.pair_results:
            lr = pr.get_layer_result(layer)
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
        self, layer: int, neuron_idx: int
    ) -> list[tuple[int, float, float]]:
        """Get activation values for a neuron across all pairs.

        Args:
            layer: Layer index
            neuron_idx: Neuron index

        Returns:
            List of (pair_idx, clean_activation, corrupted_activation)
        """
        results = []
        for pr in self.pair_results:
            lr = pr.get_layer_result(layer)
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

    def print_summary(self) -> None:
        """Print summary of MLP analysis."""
        print(f"[mlp] MLP Analysis: {self.n_pairs} pairs, layers {self.layers_analyzed}")

        for layer in self.layers_analyzed:
            # Compute mean logit contribution for this layer
            contribs = []
            for pr in self.pair_results:
                lr = pr.get_layer_result(layer)
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
