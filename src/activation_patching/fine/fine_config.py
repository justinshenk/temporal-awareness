"""Configuration for fine-grained patching analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FineConfig:
    """Configuration for fine-grained activation patching.

    Attributes:
        head_layers: Layers for head-level decomposition
        mlp_layers: Layers for MLP neuron analysis
        n_top_heads: Number of top heads to report per layer
        n_top_neurons: Number of top neurons to report per layer
        source_positions: Source positions (short-term option info, P86-P88)
        destination_positions: Destination positions (long-term option info)
    """

    # Head-level patching layers (priority order)
    head_layers: list[int] = field(default_factory=lambda: [24, 21, 19, 29, 30])

    # MLP neuron analysis layers
    mlp_layers: list[int] = field(default_factory=lambda: [31, 24, 28])

    # Reporting limits
    n_top_heads: int = 5
    n_top_neurons: int = 20

    # Key positions in prompt structure
    source_positions: list[int] = field(default_factory=lambda: [86, 87, 88])
    destination_positions: list[int] = field(default_factory=lambda: [143, 144, 145])

    @property
    def all_layers(self) -> list[int]:
        """All unique layers to analyze."""
        return sorted(set(self.head_layers + self.mlp_layers))

    @property
    def key_positions(self) -> list[int]:
        """All key positions."""
        return sorted(set(self.source_positions + self.destination_positions))


# Default configuration based on circuit hypothesis
DEFAULT_FINE_CONFIG = FineConfig()
