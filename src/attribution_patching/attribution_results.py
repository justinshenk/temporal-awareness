"""Results dataclasses for attribution patching experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

import numpy as np

from ..common.base_schema import BaseSchema

if TYPE_CHECKING:
    from ..activation_patching import ActivationPatchingTarget


@dataclass
class AttributionScore(BaseSchema):
    """Attribution score for a single layer/position.

    Attributes:
        layer: Layer index
        position: Position index
        score: Attribution score (clean - corrupted) * gradient
        component: Component analyzed
    """

    layer: int
    position: int
    score: float
    component: str = "resid_post"

    def __lt__(self, other: "AttributionScore") -> bool:
        """Sort by absolute score descending."""
        return abs(self.score) > abs(other.score)


@dataclass
class LayerAttributionResult(BaseSchema):
    """Attribution scores for all positions at one layer.

    Attributes:
        layer: Layer index
        scores: Attribution scores [n_positions]
        component: Component analyzed
    """

    layer: int
    scores: np.ndarray
    component: str = "resid_post"

    @property
    def n_positions(self) -> int:
        return len(self.scores)

    @property
    def max_score(self) -> float:
        """Maximum absolute score."""
        if len(self.scores) == 0:
            return 0.0
        return float(np.max(np.abs(self.scores)))

    @property
    def max_position(self) -> int:
        """Position with maximum absolute score."""
        if len(self.scores) == 0:
            return 0
        return int(np.argmax(np.abs(self.scores)))

    @property
    def mean_abs_score(self) -> float:
        """Mean absolute score across positions."""
        if len(self.scores) == 0:
            return 0.0
        return float(np.mean(np.abs(self.scores)))

    def get_top_positions(self, n: int = 5) -> list[tuple[int, float]]:
        """Get top N positions by absolute score.

        Returns:
            List of (position, score) tuples
        """
        indices = np.argsort(np.abs(self.scores))[::-1][:n]
        return [(int(i), float(self.scores[i])) for i in indices]


@dataclass
class AttributionPatchingResult(BaseSchema):
    """Full attribution patching result for one method.

    Attributes:
        scores: Attribution scores [n_layers, n_positions]
        layers: Layer indices corresponding to rows
        component: Component analyzed
        method: Attribution method used
    """

    scores: np.ndarray
    layers: list[int]
    component: str = "resid_post"
    method: Literal["standard", "eap", "eap_ig"] = "standard"

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def n_positions(self) -> int:
        return self.scores.shape[1] if self.scores.ndim >= 2 else 0

    @property
    def max_score(self) -> float:
        """Maximum absolute score."""
        if self.scores.size == 0:
            return 0.0
        return float(np.max(np.abs(self.scores)))

    @property
    def mean_abs_score(self) -> float:
        """Mean absolute score."""
        if self.scores.size == 0:
            return 0.0
        return float(np.mean(np.abs(self.scores)))

    def get_top_scores(self, n: int = 10) -> list[AttributionScore]:
        """Get top N scores by absolute value.

        Returns:
            List of AttributionScore objects sorted by |score|
        """
        flat_indices = np.argsort(np.abs(self.scores).ravel())[::-1][:n]
        results = []
        for idx in flat_indices:
            layer_idx = int(idx // self.scores.shape[1])
            pos = int(idx % self.scores.shape[1])
            results.append(
                AttributionScore(
                    layer=self.layers[layer_idx],
                    position=pos,
                    score=float(self.scores[layer_idx, pos]),
                    component=self.component,
                )
            )
        return results

    def get_layer_result(self, layer: int) -> LayerAttributionResult | None:
        """Get attribution result for a specific layer.

        Args:
            layer: Layer index

        Returns:
            LayerAttributionResult or None if layer not found
        """
        if layer not in self.layers:
            return None
        layer_idx = self.layers.index(layer)
        return LayerAttributionResult(
            layer=layer,
            scores=self.scores[layer_idx],
            component=self.component,
        )

    def get_scores_by_layer(self) -> dict[int, np.ndarray]:
        """Get scores grouped by layer.

        Returns:
            Dict mapping layer index to scores array
        """
        return {layer: self.scores[i] for i, layer in enumerate(self.layers)}

    def get_top_targets_for_activation_patching(
        self, n: int = 5
    ) -> list["ActivationPatchingTarget"]:
        """Convert top attributions to activation patching targets.

        Args:
            n: Number of targets to return

        Returns:
            List of ActivationPatchingTarget for top scoring positions
        """
        from ..activation_patching import ActivationPatchingTarget

        top_scores = self.get_top_scores(n)
        targets = []
        for score in top_scores:
            # Create target with both position and layer specified
            targets.append(
                ActivationPatchingTarget(
                    position_mode="explicit",
                    token_positions=[score.position],
                    layers=[score.layer],
                    component=score.component,
                )
            )
        return targets

    def print_summary(self) -> None:
        print(f"  {self.method} ({self.component}):")
        print(f"    Shape: {self.n_layers} layers x {self.n_positions} positions")
        print(f"    Max: {self.max_score:.4f}, Mean(|x|): {self.mean_abs_score:.4f}")
        top = self.get_top_scores(3)
        if top:
            print("    Top 3:")
            for s in top:
                print(f"      L{s.layer} @ pos {s.position}: {s.score:.4f}")


@dataclass
class AggregatedAttributionResult(BaseSchema):
    """Aggregated attribution results across methods and/or pairs.

    Attributes:
        results: Dict mapping method/component name to result
        n_pairs: Number of pairs aggregated (1 if single pair)
    """

    results: dict[str, AttributionPatchingResult] = field(default_factory=dict)
    n_pairs: int = 1

    @property
    def methods(self) -> list[str]:
        """List of method names in results."""
        return list(self.results.keys())

    def get_result(self, method: str) -> AttributionPatchingResult | None:
        """Get result for a specific method.

        Args:
            method: Method name (e.g., "standard_resid_post", "eap_attn")

        Returns:
            AttributionPatchingResult or None
        """
        return self.results.get(method)

    def get_top_scores(self, n: int = 10) -> list[AttributionScore]:
        """Get top N scores across all methods.

        Returns:
            List of AttributionScore objects
        """
        all_scores = []
        for result in self.results.values():
            all_scores.extend(result.get_top_scores(n))
        return sorted(all_scores)[:n]

    def get_consensus_target(
        self, n: int = 10, min_methods: int = 1
    ) -> "ActivationPatchingTarget | None":
        """Get single target combining top consensus positions.

        Finds (layer, position) pairs where methods agree and returns
        a single ActivationPatchingTarget that patches all of them together.

        NOTE: For most effective activation patching, prefer get_layer_target()
        which patches ALL positions at high-attribution layers. This is because
        causal effects are distributed across many positions, and individual
        high-attribution positions often don't capture enough signal.

        Args:
            n: Number of top (layer, position) pairs to include
            min_methods: Minimum methods that must agree

        Returns:
            Single ActivationPatchingTarget or None if no consensus
        """
        from ..activation_patching import ActivationPatchingTarget
        from collections import Counter

        # Count how many methods rank each (layer, position) in top 2n
        position_counts: Counter[tuple[int, int]] = Counter()
        for result in self.results.values():
            for score in result.get_top_scores(n * 2):
                position_counts[(score.layer, score.position)] += 1

        # Filter to positions with enough agreement
        consensus = [
            (layer, pos) for (layer, pos), count in position_counts.most_common()
            if count >= min_methods
        ][:n]

        if not consensus:
            return None

        # Combine all consensus layers and positions
        layers = sorted(set(layer for layer, _ in consensus))
        positions = sorted(set(pos for _, pos in consensus))

        return ActivationPatchingTarget(
            position_mode="explicit",
            token_positions=positions,
            layers=layers,
        )

    def get_union_target(
        self, n: int = 10, min_methods: int = 1
    ) -> "ActivationPatchingTarget | None":
        """Get target with UNION of top positions across all methods.

        Unlike get_consensus_target which requires positions to appear in
        multiple methods, this takes all unique positions from the top N
        scores of each method. Provides broader coverage.

        Args:
            n: Number of top scores to take from each method
            min_methods: Minimum methods a position must appear in (1=union, 2+=intersection)

        Returns:
            ActivationPatchingTarget with all unique positions and layers
        """
        from ..activation_patching import ActivationPatchingTarget
        from collections import Counter

        # Count occurrences of each (layer, position) across methods
        position_counts: Counter[tuple[int, int]] = Counter()
        for result in self.results.values():
            for score in result.get_top_scores(n):
                position_counts[(score.layer, score.position)] += 1

        # Filter by min_methods threshold
        selected = [
            (layer, pos) for (layer, pos), count in position_counts.items()
            if count >= min_methods
        ]

        if not selected:
            return None

        # Extract unique layers and positions
        layers = sorted(set(layer for layer, _ in selected))
        positions = sorted(set(pos for _, pos in selected))

        return ActivationPatchingTarget(
            position_mode="explicit",
            token_positions=positions,
            layers=layers,
        )

    def get_layer_target(
        self, n_layers: int = 10, min_methods: int = 1
    ) -> "ActivationPatchingTarget | None":
        """Get target that patches ALL positions at top attributed layers.

        Attribution identifies WHERE differences are encoded, but causal effects
        are distributed across positions. Patching all positions at important
        layers achieves much higher recovery than patching specific positions.

        Args:
            n_layers: Number of top layers to include
            min_methods: Minimum methods that must rank a layer highly

        Returns:
            ActivationPatchingTarget with position_mode="all" and top layers
        """
        from ..activation_patching import ActivationPatchingTarget
        from collections import Counter

        # Count how many times each layer appears in top N scores across methods
        layer_counts: Counter[int] = Counter()
        for result in self.results.values():
            for score in result.get_top_scores(n_layers * 3):
                layer_counts[score.layer] += 1

        # Get layers with enough agreement
        top_layers = [
            layer for layer, count in layer_counts.most_common()
            if count >= min_methods
        ][:n_layers]

        if not top_layers:
            return None

        return ActivationPatchingTarget(
            position_mode="all",
            layers=sorted(top_layers),
        )

    def get_recommended_target(
        self,
        n: int = 10,
        mode: str = "layer",
    ) -> "ActivationPatchingTarget | None":
        """Get recommended target for activation patching.

        The default "layer" mode patches ALL positions at top N attributed layers.
        This is recommended because:
        - Attribution identifies WHERE differences are encoded
        - But causal effects are distributed across many positions
        - Patching all positions at important layers captures more of the effect

        Position-based modes ("union", "consensus") typically give lower recovery
        because individual positions don't capture enough causal signal.

        Args:
            n: Number of top items (layers for "layer" mode, positions per method for others)
            mode: Target mode:
                - "layer": ALL positions at top N layers (recommended, ~0.5-0.7 recovery)
                - "union": Top N positions from each method (~0.05-0.25 recovery)
                - "consensus": Positions in multiple methods (~0.01-0.05 recovery)

        Returns:
            ActivationPatchingTarget configured for the specified mode
        """
        if mode == "layer":
            return self.get_layer_target(n_layers=n)
        elif mode == "union":
            return self.get_union_target(n=n, min_methods=1)
        elif mode == "consensus":
            return self.get_consensus_target(n=n, min_methods=2)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'layer', 'union', or 'consensus'")

    @classmethod
    def aggregate(
        cls, results: list["AggregatedAttributionResult"]
    ) -> "AggregatedAttributionResult":
        """Aggregate multiple results (e.g., from multiple pairs).

        Args:
            results: List of results to aggregate

        Returns:
            Aggregated result with averaged scores
        """
        if not results:
            return cls()

        if len(results) == 1:
            return results[0]

        # Find all method keys
        all_keys = set()
        for r in results:
            all_keys.update(r.results.keys())

        aggregated_results = {}
        for key in all_keys:
            # Collect all arrays for this key
            arrays = []
            layers = None
            component = "resid_post"
            method: Literal["standard", "eap", "eap_ig"] = "standard"

            for r in results:
                if key in r.results:
                    result = r.results[key]
                    arrays.append(result.scores)
                    layers = result.layers
                    component = result.component
                    method = result.method

            if not arrays or layers is None:
                continue

            # Pad to same shape and average
            max_len = max(a.shape[1] for a in arrays)
            padded = []
            for a in arrays:
                if a.shape[1] < max_len:
                    p = np.zeros((a.shape[0], max_len))
                    p[:, :a.shape[1]] = a
                    padded.append(p)
                else:
                    padded.append(a)

            aggregated_results[key] = AttributionPatchingResult(
                scores=np.mean(padded, axis=0),
                layers=layers,
                component=component,
                method=method,
            )

        return cls(results=aggregated_results, n_pairs=len(results))

    def print_summary(self) -> None:
        print(f"Attribution results ({self.n_pairs} pairs):")
        for name, result in self.results.items():
            result.print_summary()

        # Overall top scores
        top = self.get_top_scores(5)
        if top:
            print("\nTop 5 overall:")
            for s in top:
                print(f"  L{s.layer} @ pos {s.position}: {s.score:.4f} ({s.component})")
