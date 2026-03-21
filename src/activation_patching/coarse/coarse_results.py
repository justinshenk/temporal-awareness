"""Result dataclasses for coarse activation patching."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...common.base_schema import BaseSchema
from ...inference.interventions.intervention_target import InterventionTarget
from ..act_patch_results import ActPatchTargetResult


@dataclass
class SweepStepResults(BaseSchema):
    """Results for a single step size in a layer or position sweep.

    Maps start index (layer or position) to patching result.
    """

    by_start: dict[int, ActPatchTargetResult] = field(default_factory=dict)

    def __getitem__(self, key: int) -> ActPatchTargetResult:
        return self.by_start[key]

    def __setitem__(self, key: int, value: ActPatchTargetResult) -> None:
        self.by_start[key] = value

    def get(self, key: int) -> ActPatchTargetResult | None:
        return self.by_start.get(key)

    def keys(self):
        return self.by_start.keys()

    def values(self):
        return self.by_start.values()

    def items(self):
        return self.by_start.items()

    def __len__(self) -> int:
        return len(self.by_start)

    def __bool__(self) -> bool:
        return bool(self.by_start)

    def pop_heavy(self) -> None:
        """Remove heavy data from all results."""
        for result in self.by_start.values():
            result.pop_heavy()


@dataclass
class CoarseActPatchResults(BaseSchema):
    """Results from coarse activation patching on single pair.

    Results are organized by step size:
    - layer_results[step_size].by_start[layer_start] = ActPatchTargetResult
    - position_results[step_size].by_start[pos_start] = ActPatchTargetResult
    """

    sample_id: int = 0
    sanity_result: ActPatchTargetResult | None = None
    layer_results: dict[int, SweepStepResults] = field(default_factory=dict)
    position_results: dict[int, SweepStepResults] = field(default_factory=dict)

    @property
    def layer_step_sizes(self) -> list[int]:
        """Available layer step sizes."""
        return sorted(self.layer_results.keys())

    @property
    def position_step_sizes(self) -> list[int]:
        """Available position step sizes."""
        return sorted(self.position_results.keys())

    @property
    def component(self) -> str:
        """Component used for patching (from sanity_result target)."""
        if self.sanity_result and self.sanity_result.target:
            return self.sanity_result.target.component or "resid_post"
        return "resid_post"

    def get_layer_results_for_step(self, step_size: int) -> SweepStepResults:
        """Get layer results for a specific step size."""
        return self.layer_results.get(step_size, SweepStepResults())

    def get_position_results_for_step(self, step_size: int) -> SweepStepResults:
        """Get position results for a specific step size."""
        return self.position_results.get(step_size, SweepStepResults())

    def get_result_for_layer(
        self, layer: int, step_size: int | None = None
    ) -> ActPatchTargetResult | None:
        """Get result for a layer. If step_size not specified, uses first available."""
        if step_size is None:
            step_size = self.layer_step_sizes[0] if self.layer_step_sizes else None
        if step_size is None:
            return None
        return self.get_layer_results_for_step(step_size).get(layer)

    def get_result_for_pos(
        self, n_positions: int, step_size: int | None = None
    ) -> ActPatchTargetResult | None:
        """Get result for a position. If step_size not specified, uses first available."""
        if step_size is None:
            step_size = (
                self.position_step_sizes[0] if self.position_step_sizes else None
            )
        if step_size is None:
            return None
        return self.get_position_results_for_step(step_size).get(n_positions)

    def best_layers(self, n_top: int = 3, step_size: int | None = None) -> list[int]:
        """Top n layers by score for a given step size."""
        if step_size is None:
            step_size = self.layer_step_sizes[0] if self.layer_step_sizes else None
        if step_size is None:
            return []
        results = self.get_layer_results_for_step(step_size)
        sorted_layers = sorted(
            results.items(),
            key=lambda x: x[1].score(),
            reverse=True,
        )
        return [layer for layer, _ in sorted_layers[:n_top]]

    def best_n_positions(
        self, threshold: float = 0.8, step_size: int | None = None
    ) -> int:
        """Min positions for recovery > threshold."""
        if step_size is None:
            step_size = (
                self.position_step_sizes[0] if self.position_step_sizes else None
            )
        if step_size is None:
            return 0
        results = self.get_position_results_for_step(step_size)
        for n in sorted(results.keys()):
            if results[n].score() > threshold:
                return n
        return max(results.keys()) if results else 0

    def get_union_target(
        self,
        n_top_layers: int = 3,
        position_threshold: float = 0.8,
        component: str = "resid_post",
        layer_step_size: int | None = None,
        position_step_size: int | None = None,
    ) -> InterventionTarget:
        """Get target combining best layers and positions."""
        layers = self.best_layers(n_top=n_top_layers, step_size=layer_step_size)
        n_pos = self.best_n_positions(
            threshold=position_threshold, step_size=position_step_size
        )
        positions = list(range(n_pos)) if n_pos else None
        return InterventionTarget.at(
            positions=positions,
            layers=layers if layers else None,
            component=component,
        )

    def pop_heavy(self) -> None:
        """Remove heavy data from all results."""
        if self.sanity_result:
            self.sanity_result.pop_heavy()
        for step_results in self.layer_results.values():
            step_results.pop_heavy()
        for step_results in self.position_results.values():
            step_results.pop_heavy()


@dataclass
class CoarseActPatchAggregatedResults(BaseSchema):
    """Aggregated coarse patching results across multiple pairs."""

    by_sample: dict[int, CoarseActPatchResults] = field(default_factory=dict)

    def add(self, result: CoarseActPatchResults) -> None:
        """Add a result to the aggregation."""
        self.by_sample[result.sample_id] = result

    @property
    def n_samples(self) -> int:
        return len(self.by_sample)

    @property
    def layer_step_sizes(self) -> list[int]:
        """All available layer step sizes across samples."""
        sizes = set()
        for r in self.by_sample.values():
            sizes.update(r.layer_step_sizes)
        return sorted(sizes)

    @property
    def position_step_sizes(self) -> list[int]:
        """All available position step sizes across samples."""
        sizes = set()
        for r in self.by_sample.values():
            sizes.update(r.position_step_sizes)
        return sorted(sizes)

    @property
    def component(self) -> str:
        """Component used for patching (from first sample)."""
        if self.by_sample:
            return next(iter(self.by_sample.values())).component
        return "resid_post"

    def mean_sanity_score(self) -> float:
        """Mean sanity check score across samples."""
        scores = [
            r.sanity_result.score() for r in self.by_sample.values() if r.sanity_result
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def get_mean_layer_scores(self, step_size: int | None = None) -> dict[int, float]:
        """Mean recovery per layer across all samples for a given step size."""
        if step_size is None:
            step_size = self.layer_step_sizes[0] if self.layer_step_sizes else None
        if step_size is None:
            return {}
        by_layer: dict[int, list[float]] = {}
        for result in self.by_sample.values():
            for layer, target_result in result.get_layer_results_for_step(
                step_size
            ).items():
                by_layer.setdefault(layer, []).append(target_result.score())
        return {l: sum(s) / len(s) for l, s in by_layer.items()}

    def get_mean_position_scores(
        self, step_size: int | None = None
    ) -> dict[int, float]:
        """Mean recovery per position across all samples for a given step size."""
        if step_size is None:
            step_size = (
                self.position_step_sizes[0] if self.position_step_sizes else None
            )
        if step_size is None:
            return {}
        by_pos: dict[int, list[float]] = {}
        for result in self.by_sample.values():
            for pos, target_result in result.get_position_results_for_step(
                step_size
            ).items():
                by_pos.setdefault(pos, []).append(target_result.score())
        return {p: sum(s) / len(s) for p, s in by_pos.items()}

    def best_layers(self, n_top: int = 3, step_size: int | None = None) -> list[int]:
        """Top n layers by mean score."""
        layer_scores = self.get_mean_layer_scores(step_size=step_size)
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        return [layer for layer, _ in sorted_layers[:n_top]]

    def get_union_target(
        self,
        n_top_layers: int = 3,
        position_threshold: float = 0.8,
        component: str = "resid_post",
        layer_step_size: int | None = None,
        position_step_size: int | None = None,
    ) -> InterventionTarget:
        """Get target combining best layers and positions across all samples."""
        layers = self.best_layers(n_top=n_top_layers, step_size=layer_step_size)

        # Find min position where mean score > threshold
        pos_scores = self.get_mean_position_scores(step_size=position_step_size)
        n_pos = 0
        for pos in sorted(pos_scores.keys()):
            if pos_scores[pos] > position_threshold:
                n_pos = pos
                break
        if n_pos == 0 and pos_scores:
            n_pos = max(pos_scores.keys())

        positions = list(range(n_pos)) if n_pos else None
        return InterventionTarget.at(
            positions=positions,
            layers=layers if layers else None,
            component=component,
        )

    def print_summary(self) -> None:
        """Print summary of aggregated results."""
        print(f"Coarse patching: {self.n_samples} samples")
        print(f"  Mean sanity score: {self.mean_sanity_score():.3f}")
        for step_size in self.layer_step_sizes:
            best = self.best_layers(n_top=3, step_size=step_size)
            if best:
                layer_scores = self.get_mean_layer_scores(step_size=step_size)
                scores_str = [f"{layer_scores[l]:.3f}" for l in best]
                print(
                    f"  [step={step_size}] Best layers: {best} (scores: {scores_str})"
                )

    def pop_heavy(self) -> None:
        """Remove heavy data from all results."""
        for result in self.by_sample.values():
            result.pop_heavy()
