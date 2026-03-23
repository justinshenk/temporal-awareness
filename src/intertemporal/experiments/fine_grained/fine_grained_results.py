"""Result dataclasses for fine-grained patching analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ....common.base_schema import BaseSchema


@dataclass
class HeadPatchingResult(BaseSchema):
    """Result for a single attention head in both patching modes.

    Attributes:
        layer: Layer index
        head: Head index
        denoising_recovery: Recovery score in denoising mode
        noising_disruption: Disruption score in noising mode
    """

    layer: int
    head: int
    denoising_recovery: float = 0.0
    noising_disruption: float = 0.0

    @property
    def label(self) -> str:
        """Human-readable label for this head."""
        return f"L{self.layer}.H{self.head}"

    @property
    def combined_score(self) -> float:
        """Combined importance score (average of both modes)."""
        return (self.denoising_recovery + self.noising_disruption) / 2


@dataclass
class HeadSweepResults(BaseSchema):
    """Results from head-level patching sweep across all layers.

    Attributes:
        n_layers: Number of layers analyzed
        n_heads: Number of heads per layer
        results: List of HeadPatchingResult for each (layer, head)
        denoising_matrix: [n_layers, n_heads] array of denoising recovery scores
        noising_matrix: [n_layers, n_heads] array of noising disruption scores
    """

    n_layers: int = 0
    n_heads: int = 0
    results: list[HeadPatchingResult] = field(default_factory=list)
    denoising_matrix: np.ndarray | None = None
    noising_matrix: np.ndarray | None = None
    layers_analyzed: list[int] = field(default_factory=list)

    def get_top_heads(self, n: int = 20, by: str = "combined") -> list[HeadPatchingResult]:
        """Get top N heads by specified metric.

        Args:
            n: Number of heads to return
            by: "combined", "denoising", or "noising"

        Returns:
            List of top N HeadPatchingResult sorted by metric
        """
        if by == "denoising":
            key = lambda h: abs(h.denoising_recovery)
        elif by == "noising":
            key = lambda h: abs(h.noising_disruption)
        else:
            key = lambda h: abs(h.combined_score)
        return sorted(self.results, key=key, reverse=True)[:n]

    def build_matrices(self) -> None:
        """Build denoising_matrix and noising_matrix from results."""
        if not self.results or not self.layers_analyzed:
            return

        n_layers = len(self.layers_analyzed)
        layer_to_idx = {l: i for i, l in enumerate(self.layers_analyzed)}

        self.denoising_matrix = np.zeros((n_layers, self.n_heads))
        self.noising_matrix = np.zeros((n_layers, self.n_heads))

        for r in self.results:
            if r.layer in layer_to_idx:
                idx = layer_to_idx[r.layer]
                self.denoising_matrix[idx, r.head] = r.denoising_recovery
                self.noising_matrix[idx, r.head] = r.noising_disruption

    def to_dict(self, **kwargs) -> dict:
        """Custom serialization to handle numpy arrays."""
        d = {
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "results": [r.to_dict() for r in self.results],
            "layers_analyzed": self.layers_analyzed,
        }
        if self.denoising_matrix is not None:
            d["denoising_matrix"] = self.denoising_matrix.tolist()
        if self.noising_matrix is not None:
            d["noising_matrix"] = self.noising_matrix.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HeadSweepResults":
        """Custom deserialization to handle numpy arrays."""
        result = cls(
            n_layers=d.get("n_layers", 0),
            n_heads=d.get("n_heads", 0),
            results=[HeadPatchingResult.from_dict(r) for r in d.get("results", [])],
            layers_analyzed=d.get("layers_analyzed", []),
        )
        if "denoising_matrix" in d and d["denoising_matrix"] is not None:
            result.denoising_matrix = np.array(d["denoising_matrix"])
        if "noising_matrix" in d and d["noising_matrix"] is not None:
            result.noising_matrix = np.array(d["noising_matrix"])
        return result


@dataclass
class PositionPatchingResult(BaseSchema):
    """Result for position-level patching of a head.

    For each top head, stores patching effect at each position.

    Attributes:
        layer: Layer index
        head: Head index
        positions: List of positions analyzed
        denoising_by_position: List of denoising recovery at each position
        noising_by_position: List of noising disruption at each position
    """

    layer: int
    head: int
    positions: list[int] = field(default_factory=list)
    denoising_by_position: list[float] = field(default_factory=list)
    noising_by_position: list[float] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"L{self.layer}.H{self.head}"


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
class NeuronPatchingResult(BaseSchema):
    """Result for neuron-level ablation at a target layer.

    Attributes:
        layer: Layer index
        neuron_idx: Neuron index
        effect: Drop in correct logit probability when ablating this neuron
        activation_mean: Mean activation of this neuron
    """

    layer: int
    neuron_idx: int
    effect: float = 0.0
    activation_mean: float = 0.0


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
            result.denoising_grid = np.array(d["denoising_grid"])
        if "noising_grid" in d and d["noising_grid"] is not None:
            result.noising_grid = np.array(d["noising_grid"])
        return result


@dataclass
class FineGrainedResults(BaseSchema):
    """Aggregated results from all fine-grained analyses.

    Attributes:
        sample_id: Sample identifier
        head_sweep: Results from head-level patching sweep
        position_results: Position patching for top heads
        path_to_mlp: Head-to-MLP path patching results
        path_to_head: Head-to-head path patching results
        multi_site: Multi-site interaction results
        neuron_results: Neuron-level ablation results
        layer_position: Layer x position fine heatmap data
    """

    sample_id: int | None = None
    head_sweep: HeadSweepResults | None = None
    position_results: list[PositionPatchingResult] = field(default_factory=list)
    path_to_mlp: list[PathPatchingResult] = field(default_factory=list)
    path_to_head: list[PathPatchingResult] = field(default_factory=list)
    multi_site: list[MultiSiteResult] = field(default_factory=list)
    neuron_results: list[NeuronPatchingResult] = field(default_factory=list)
    layer_position: dict[str, LayerPositionResult] = field(default_factory=dict)

    # Metadata
    n_layers: int = 0
    n_heads: int = 0
    neuron_target_layer: int = 31

    def get_top_neurons(self, n: int = 50) -> list[NeuronPatchingResult]:
        """Get top N neurons by ablation effect."""
        return sorted(
            self.neuron_results,
            key=lambda x: abs(x.effect),
            reverse=True
        )[:n]

    def get_cumulative_neuron_effect(self, n_neurons: int) -> float:
        """Get cumulative effect of top N neurons.

        Returns fraction of total MLP layer's patching effect recovered.
        """
        if not self.neuron_results:
            return 0.0
        total = sum(abs(r.effect) for r in self.neuron_results)
        if total == 0:
            return 0.0
        top_n = self.get_top_neurons(n_neurons)
        top_sum = sum(abs(r.effect) for r in top_n)
        return top_sum / total

    def print_summary(self) -> None:
        """Print summary of results."""
        print(f"[fine_grained] Sample {self.sample_id}")
        if self.head_sweep:
            print(f"  Head sweep: {len(self.head_sweep.results)} heads analyzed")
            top = self.head_sweep.get_top_heads(3)
            for h in top:
                print(f"    {h.label}: dn={h.denoising_recovery:.3f}, ns={h.noising_disruption:.3f}")
        if self.position_results:
            print(f"  Position patching: {len(self.position_results)} heads")
        if self.path_to_mlp:
            print(f"  Path to MLP: {len(self.path_to_mlp)} paths")
        if self.path_to_head:
            print(f"  Path to head: {len(self.path_to_head)} paths")
        if self.multi_site:
            print(f"  Multi-site: {len(self.multi_site)} interactions")
        if self.neuron_results:
            print(f"  Neuron ablation: {len(self.neuron_results)} neurons at L{self.neuron_target_layer}")
        if self.layer_position:
            for comp, lp in self.layer_position.items():
                print(f"  Layer-position ({comp}): {len(lp.layers)} layers x {len(lp.positions)} positions")

    def to_dict(self, **kwargs) -> dict:
        """Custom serialization to handle nested dataclasses with numpy arrays."""
        d = {
            "sample_id": self.sample_id,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "neuron_target_layer": self.neuron_target_layer,
            "position_results": [r.to_dict() for r in self.position_results],
            "path_to_mlp": [r.to_dict() for r in self.path_to_mlp],
            "path_to_head": [r.to_dict() for r in self.path_to_head],
            "multi_site": [r.to_dict() for r in self.multi_site],
            "neuron_results": [r.to_dict() for r in self.neuron_results],
        }
        if self.head_sweep is not None:
            d["head_sweep"] = self.head_sweep.to_dict()
        d["layer_position"] = {k: v.to_dict() for k, v in self.layer_position.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FineGrainedResults":
        """Custom deserialization to handle nested dataclasses with numpy arrays."""
        result = cls(
            sample_id=d.get("sample_id"),
            n_layers=d.get("n_layers", 0),
            n_heads=d.get("n_heads", 0),
            neuron_target_layer=d.get("neuron_target_layer", 31),
            position_results=[PositionPatchingResult.from_dict(r) for r in d.get("position_results", [])],
            path_to_mlp=[PathPatchingResult.from_dict(r) for r in d.get("path_to_mlp", [])],
            path_to_head=[PathPatchingResult.from_dict(r) for r in d.get("path_to_head", [])],
            multi_site=[MultiSiteResult.from_dict(r) for r in d.get("multi_site", [])],
            neuron_results=[NeuronPatchingResult.from_dict(r) for r in d.get("neuron_results", [])],
        )
        if "head_sweep" in d and d["head_sweep"] is not None:
            result.head_sweep = HeadSweepResults.from_dict(d["head_sweep"])
        if "layer_position" in d:
            result.layer_position = {k: LayerPositionResult.from_dict(v) for k, v in d["layer_position"].items()}
        return result
