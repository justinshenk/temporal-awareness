"""Data structures for fine-grained patching results."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...common.base_schema import BaseSchema


@dataclass
class HeadResult(BaseSchema):
    """Result for a single attention head."""

    layer: int
    head: int
    score: float  # Attribution/importance score
    denoising_score: float = 0.0
    noising_score: float = 0.0

    def __lt__(self, other: HeadResult) -> bool:
        return abs(self.score) < abs(other.score)


@dataclass
class HeadPatchingResults(BaseSchema):
    """Results from head-level patching at one layer."""

    layer: int
    n_heads: int
    head_results: list[HeadResult] = field(default_factory=list)

    @property
    def top_heads(self) -> list[HeadResult]:
        """Heads sorted by absolute score (descending)."""
        return sorted(self.head_results, key=lambda h: abs(h.score), reverse=True)

    def get_top_n(self, n: int) -> list[HeadResult]:
        """Get top N heads by absolute score."""
        return self.top_heads[:n]

    @property
    def scores_array(self) -> np.ndarray:
        """All head scores as array."""
        return np.array([h.score for h in self.head_results])


@dataclass
class NeuronResult(BaseSchema):
    """Result for a single MLP neuron."""

    layer: int
    neuron_idx: int
    contribution: float  # Contribution to output difference
    activation_diff: float = 0.0  # Clean - corrupted activation difference
    logit_projection: float = 0.0  # Projection onto logit direction

    def __lt__(self, other: NeuronResult) -> bool:
        return abs(self.contribution) < abs(other.contribution)


@dataclass
class MLPNeuronResults(BaseSchema):
    """Results from MLP neuron analysis at one layer."""

    layer: int
    n_neurons: int
    neuron_results: list[NeuronResult] = field(default_factory=list)

    @property
    def top_neurons(self) -> list[NeuronResult]:
        """Neurons sorted by absolute contribution (descending)."""
        return sorted(
            self.neuron_results, key=lambda n: abs(n.contribution), reverse=True
        )

    def get_top_n(self, n: int) -> list[NeuronResult]:
        """Get top N neurons by absolute contribution."""
        return self.top_neurons[:n]

    @property
    def contribution_array(self) -> np.ndarray:
        """All neuron contributions as array."""
        return np.array([n.contribution for n in self.neuron_results])


@dataclass
class AttentionPatternResult(BaseSchema):
    """Attention pattern analysis for a single head."""

    layer: int
    head: int

    # Attention from destination to source positions
    src_to_dest_attention: np.ndarray | None = None  # [n_dest, n_src]

    # Mean attention from destination positions back to source positions
    mean_attention_to_source: float = 0.0

    # Value difference weighted by attention
    attention_weighted_value_diff: np.ndarray | None = None  # [d_head]

    # What information flows: clean - corrupted value, weighted by attention
    info_flow_norm: float = 0.0


@dataclass
class FinePatchingResults(BaseSchema):
    """Aggregated results from all fine-grained analyses."""

    # Head patching results by layer
    head_results: dict[int, HeadPatchingResults] = field(default_factory=dict)

    # MLP neuron results by layer
    mlp_results: dict[int, MLPNeuronResults] = field(default_factory=dict)

    # Attention pattern analysis
    attention_patterns: list[AttentionPatternResult] = field(default_factory=list)

    # Metadata
    sample_id: int | None = None
    n_layers: int = 0
    n_heads: int = 0
    d_head: int = 0
    d_mlp: int = 0

    def get_top_heads_all_layers(self, n_per_layer: int = 5) -> list[HeadResult]:
        """Get top heads across all analyzed layers."""
        all_heads = []
        for layer_result in self.head_results.values():
            all_heads.extend(layer_result.get_top_n(n_per_layer))
        return sorted(all_heads, key=lambda h: abs(h.score), reverse=True)

    def get_top_neurons_all_layers(self, n_per_layer: int = 10) -> list[NeuronResult]:
        """Get top neurons across all analyzed layers."""
        all_neurons = []
        for layer_result in self.mlp_results.values():
            all_neurons.extend(layer_result.get_top_n(n_per_layer))
        return sorted(all_neurons, key=lambda n: abs(n.contribution), reverse=True)

    def print_summary(self) -> None:
        """Print summary of results."""
        print(f"[fine] Sample {self.sample_id}")
        print(f"  Head patching: {len(self.head_results)} layers")
        print(f"  MLP analysis: {len(self.mlp_results)} layers")
        print(f"  Attention patterns: {len(self.attention_patterns)} heads")

        if self.head_results:
            top_heads = self.get_top_heads_all_layers(3)[:5]
            print("  Top heads:")
            for h in top_heads:
                print(f"    L{h.layer}.H{h.head}: {h.score:.4f}")

        if self.mlp_results:
            top_neurons = self.get_top_neurons_all_layers(5)[:5]
            print("  Top neurons:")
            for n in top_neurons:
                print(f"    L{n.layer}.N{n.neuron_idx}: {n.contribution:.4f}")
