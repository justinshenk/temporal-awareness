"""Results dataclasses for attribution patching experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..common import BaseSchema
from ..inference import InterventionTarget


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
        """Get scores grouped by layer."""
        return {layer: self.scores[i] for i, layer in enumerate(self.layers)}

    def get_top_targets(self, n: int = 5) -> list[InterventionTarget]:
        """Get InterventionTargets for top scoring positions."""
        return [
            InterventionTarget.at(
                positions=[s.position], layers=[s.layer], component=s.component
            )
            for s in self.get_top_scores(n)
        ]

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
class AttributionSummary(BaseSchema):
    """Aggregated attribution results across methods and/or pairs.

    Attributes:
        results: Dict mapping method/component name to result
        n_pairs: Number of pairs aggregated (1 if single pair)
        mode: Attribution mode ("denoising" or "noising")
    """

    results: dict[str, AttributionPatchingResult] = field(default_factory=dict)
    n_pairs: int = 1
    mode: Literal["denoising", "noising"] | None = None

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

    def get_position_target(
        self, n: int = 10, min_methods: int = 1
    ) -> InterventionTarget | None:
        """Get target from top attributed positions.

        NOTE: For most effective activation patching, prefer get_layer_target()
        which patches ALL positions at high-attribution layers.

        Args:
            n: Number of top scores per method
            min_methods: Minimum methods a position must appear in

        Returns:
            InterventionTarget or None
        """
        counts: Counter[tuple[int, int]] = Counter()
        for result in self.results.values():
            for score in result.get_top_scores(n):
                counts[(score.layer, score.position)] += 1

        selected = [(l, p) for (l, p), c in counts.most_common() if c >= min_methods][:n]
        if not selected:
            return None

        layers = sorted(set(l for l, _ in selected))
        positions = sorted(set(p for _, p in selected))
        return InterventionTarget.at(positions=positions, layers=layers)

    def get_layer_target(
        self, n_layers: int = 10, min_methods: int = 1
    ) -> "InterventionTarget | None":
        """Get target that patches ALL positions at top attributed layers.

        Attribution identifies WHERE differences are encoded, but causal effects
        are distributed across positions. Patching all positions at important
        layers achieves much higher recovery than patching specific positions.

        Args:
            n_layers: Number of top layers to include
            min_methods: Minimum methods that must rank a layer highly

        Returns:
            InterventionTarget with position_mode="all" and top layers
        """
        # Count how many times each layer appears in top N scores across methods
        layer_counts: Counter[int] = Counter()
        for result in self.results.values():
            for score in result.get_top_scores(n_layers * 3):
                layer_counts[score.layer] += 1

        # Get layers with enough agreement
        top_layers = [
            layer for layer, count in layer_counts.most_common() if count >= min_methods
        ][:n_layers]

        if not top_layers:
            return None

        return InterventionTarget.at_layers(sorted(top_layers))

    def get_target(
        self,
        n: int = 10,
        mode: str = "layer",
    ) -> InterventionTarget | None:
        """Get intervention target for activation patching.

        Args:
            n: Number of top layers or positions
            mode: "layer" (recommended) or "position"

        Returns:
            InterventionTarget
        """
        if mode == "layer":
            return self.get_layer_target(n_layers=n)
        elif mode == "position":
            return self.get_position_target(n=n)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'layer' or 'position'")

    @classmethod
    def aggregate(
        cls, results: list["AttributionSummary"]
    ) -> "AttributionSummary":
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
                    p[:, : a.shape[1]] = a
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


# =============================================================================
# Pair-level Results (mirroring activation patching structure)
# =============================================================================


@dataclass
class AttrPatchTargetResult(BaseSchema):
    """Attribution results for one target (both modes).

    Mirrors ActPatchTargetResult in activation_patching.
    """

    denoising: AttributionSummary | None = None
    noising: AttributionSummary | None = None

    @property
    def mean_max_score(self) -> float:
        """Mean of max scores across modes."""
        scores = []
        if self.denoising:
            for r in self.denoising.results.values():
                scores.append(r.max_score)
        if self.noising:
            for r in self.noising.results.values():
                scores.append(r.max_score)
        return sum(scores) / len(scores) if scores else 0.0

    def get_top_scores(self, n: int = 10) -> list[AttributionScore]:
        """Get top scores across both modes."""
        all_scores = []
        if self.denoising:
            all_scores.extend(self.denoising.get_top_scores(n))
        if self.noising:
            all_scores.extend(self.noising.get_top_scores(n))
        return sorted(all_scores)[:n]

    def get_target(self, n: int = 10, mode: str = "layer") -> InterventionTarget | None:
        """Get intervention target from attribution results."""
        result = self.denoising or self.noising
        return result.get_target(n=n, mode=mode) if result else None


@dataclass
class AttrPatchPairResult(BaseSchema):
    """Attribution results for one contrastive pair.

    Mirrors ActPatchPairResult in activation_patching.
    """

    sample_id: int = 0
    result: AttrPatchTargetResult = field(default_factory=AttrPatchTargetResult)

    def get_top_scores(self, n: int = 10) -> list[AttributionScore]:
        return self.result.get_top_scores(n)

    def get_target(self, n: int = 10, mode: str = "layer") -> InterventionTarget | None:
        return self.result.get_target(n=n, mode=mode)

    def print_summary(self) -> None:
        print(f"Sample {self.sample_id}:")
        if self.result.denoising:
            print("  Denoising:")
            self.result.denoising.print_summary()
        if self.result.noising:
            print("  Noising:")
            self.result.noising.print_summary()


@dataclass
class AttrPatchAggregatedResults(BaseSchema):
    """Aggregator for attribution patching results across pairs.

    Mirrors CoarseActPatchAggregatedResults pattern.
    """

    denoising: list[AttributionSummary] = field(default_factory=list)
    noising: list[AttributionSummary] = field(default_factory=list)
    _denoising_agg: AttributionSummary | None = field(default=None, init=False)
    _noising_agg: AttributionSummary | None = field(default=None, init=False)

    def add(self, pair_result: AttrPatchPairResult) -> None:
        """Add a pair result to the aggregation."""
        if pair_result.result.denoising:
            self.denoising.append(pair_result.result.denoising)
            self._denoising_agg = None  # Invalidate cache
        if pair_result.result.noising:
            self.noising.append(pair_result.result.noising)
            self._noising_agg = None

    @property
    def denoising_agg(self) -> AttributionSummary | None:
        """Aggregated denoising results."""
        if self._denoising_agg is None and self.denoising:
            self._denoising_agg = AttributionSummary.aggregate(self.denoising)
        return self._denoising_agg

    @property
    def noising_agg(self) -> AttributionSummary | None:
        """Aggregated noising results."""
        if self._noising_agg is None and self.noising:
            self._noising_agg = AttributionSummary.aggregate(self.noising)
        return self._noising_agg

    def get_target(self, n: int = 10, mode: str = "layer") -> InterventionTarget | None:
        """Get intervention target from aggregated results."""
        agg = self.denoising_agg or self.noising_agg
        return agg.get_target(n=n, mode=mode) if agg else None

    def print_summary(self) -> None:
        print(f"Attribution Patching ({len(self.denoising)} denoising, {len(self.noising)} noising):")
        if self.denoising_agg:
            print("  Denoising aggregated:")
            self.denoising_agg.print_summary()
        if self.noising_agg:
            print("  Noising aggregated:")
            self.noising_agg.print_summary()
