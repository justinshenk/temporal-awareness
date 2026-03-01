"""Coarse activation patching: layer and position sweeps on single pair."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..common.base_schema import BaseSchema
from ..common.device_utils import clear_gpu_memory
from ..inference.interventions.intervention_target import InterventionTarget
from . import patch_target, ActPatchTargetResult
from ..binary_choice import BinaryChoiceRunner
from ..common.contrastive_pair import ContrastivePair


@dataclass
class CoarseActPatchResults(BaseSchema):
    """Results from coarse activation patching on single pair.

    Results are organized by step size:
    - layer_results[step_size][layer_start] = ActPatchTargetResult
    - position_results[step_size][pos_start] = ActPatchTargetResult
    """

    sample_id: int = 0
    sanity_result: ActPatchTargetResult | None = None
    # Outer key: step_size, inner key: layer/position start
    layer_results: dict[int, dict[int, ActPatchTargetResult]] = field(
        default_factory=dict
    )
    position_results: dict[int, dict[int, ActPatchTargetResult]] = field(
        default_factory=dict
    )

    @property
    def layer_step_sizes(self) -> list[int]:
        """Available layer step sizes."""
        return sorted(self.layer_results.keys())

    @property
    def position_step_sizes(self) -> list[int]:
        """Available position step sizes."""
        return sorted(self.position_results.keys())

    def get_layer_results_for_step(
        self, step_size: int
    ) -> dict[int, ActPatchTargetResult]:
        """Get layer results for a specific step size."""
        return self.layer_results.get(step_size, {})

    def get_position_results_for_step(
        self, step_size: int
    ) -> dict[int, ActPatchTargetResult]:
        """Get position results for a specific step size."""
        return self.position_results.get(step_size, {})

    def get_result_for_layer(
        self, layer: int, step_size: int | None = None
    ) -> ActPatchTargetResult | None:
        """Get result for a layer. If step_size not specified, uses first available."""
        if step_size is None:
            step_size = self.layer_step_sizes[0] if self.layer_step_sizes else None
        if step_size is None:
            return None
        return self.layer_results.get(step_size, {}).get(layer)

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
        return self.position_results.get(step_size, {}).get(n_positions)

    def best_layers(self, n_top: int = 3, step_size: int | None = None) -> list[int]:
        """Top n layers by score for a given step size."""
        if step_size is None:
            step_size = self.layer_step_sizes[0] if self.layer_step_sizes else None
        if step_size is None:
            return []
        results = self.layer_results.get(step_size, {})
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
        results = self.position_results.get(step_size, {})
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


def run_coarse_act_patching(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    component: str = "resid_post",
    min_layer_depth: float = 0.01,
    max_layer_depth: float = 0.99,
    layer_step_sizes: list[int] | None = None,
    pos_step_sizes: list[int] | None = None,
) -> CoarseActPatchResults:
    """Run sanity check, layer sweep, and position sweep on single pair.

    Args:
        runner: BinaryChoiceRunner for inference
        pair: ContrastivePair to patch
        component: Component to patch (default: resid_post)
        min_layer_depth: Start layer as fraction of total layers
        max_layer_depth: End layer as fraction of total layers
        layer_step_sizes: List of step sizes for layer sweeps (default: [8])
        pos_step_sizes: List of step sizes for position sweeps (default: [10])

    Returns:
        CoarseActPatchResults with results organized by step size
    """
    if layer_step_sizes is None:
        layer_step_sizes = [1, 3, 9, 16]
        # layer_step_sizes = [3]
    if pos_step_sizes is None:
        pos_step_sizes = [1, 3, 9, 16]
        # pos_step_sizes = [20]

    # Sanity check: patch all positions
    print("[coarse] Starting sanity check (all layers, all positions)...")
    sanity_target = InterventionTarget.all(component)
    sanity_result = patch_target(runner, pair, sanity_target)
    print(f"[coarse] Sanity check done: recovery={sanity_result.score():.3f}")

    # Layer sweeps for each step size
    n_layers = len(pair.available_layers)
    start_layer = int(n_layers * min_layer_depth)
    end_layer = int(n_layers * max_layer_depth)
    layers_of_interest = pair.available_layers[start_layer:end_layer]

    layer_results: dict[int, dict[int, ActPatchTargetResult]] = {}
    for layer_step in layer_step_sizes:
        layer_results[layer_step] = {}
        print(
            f"[coarse] Layer sweep (step={layer_step}) from {layers_of_interest[0]} to {layers_of_interest[-1]}..."
        )
        for i in range(0, len(layers_of_interest), layer_step):
            layer_range = layers_of_interest[i : i + layer_step]
            target = InterventionTarget.at_layers(layer_range, component=component)
            layer_results[layer_step][layer_range[0]] = patch_target(
                runner, pair, target
            )
            print(
                f"[coarse] Layers:{layer_range} recovery={layer_results[layer_step][layer_range[0]].score():.3f}, {i // layer_step + 1}/{-(-len(layers_of_interest) // layer_step)}"
            )
        # Clear memory after each step size sweep
        clear_gpu_memory()

    # Position sweeps for each step size
    start_pos = pair.position_mapping.first_interesting_pos
    end_pos = min(pair.choice_divergent_positions)

    position_results: dict[int, dict[int, ActPatchTargetResult]] = {}
    for pos_step in pos_step_sizes:
        position_results[pos_step] = {}
        print(
            f"[coarse] Position sweep (step={pos_step}) from {start_pos} to {end_pos}..."
        )
        iter_count = 0
        for pos in range(start_pos, end_pos, pos_step):
            pos_range = list(range(pos, min(pos + pos_step, end_pos)))
            target = InterventionTarget.at_positions(pos_range, component=component)
            position_results[pos_step][pos] = patch_target(runner, pair, target)
            print(
                f"[coarse] pos={pos} recovery={position_results[pos_step][pos].score():.3f}"
            )
            # Clear memory periodically during position sweep
            iter_count += 1
            if iter_count % 10 == 0:
                clear_gpu_memory()
        # Clear memory after each step size sweep
        clear_gpu_memory()

    clear_gpu_memory()
    print("[coarse] Done.")
    return CoarseActPatchResults(
        sanity_result=sanity_result,
        layer_results=layer_results,
        position_results=position_results,
    )
