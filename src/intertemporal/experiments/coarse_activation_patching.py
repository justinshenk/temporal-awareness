"""Coarse activation patching: layer and position sweeps on single pair."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...common.base_schema import BaseSchema
from ...common.device_utils import clear_gpu_memory
from ...inference.interventions.intervention_target import InterventionTarget
from ...activation_patching import patch_target, ActPatchTargetResult


from ...binary_choice import BinaryChoiceRunner
from ...common.contrastive_pair import ContrastivePair


@dataclass
class CoarseActPatchResults(BaseSchema):
    """Results from coarse activation patching on single pair."""

    sample_id: int = 0
    sanity_result: ActPatchTargetResult | None = None
    layer_results: dict[int, ActPatchTargetResult] = field(default_factory=dict)
    position_results: dict[int, ActPatchTargetResult] = field(default_factory=dict)

    def get_result_for_layer(self, layer: int) -> ActPatchTargetResult | None:
        return self.layer_results.get(layer)

    def get_result_for_pos(self, n_positions: int) -> ActPatchTargetResult | None:
        return self.position_results.get(n_positions)

    def best_layers(self, n_top: int = 3) -> list[int]:
        """Top n layers by score."""
        sorted_layers = sorted(
            self.layer_results.items(),
            key=lambda x: x[1].score(),
            reverse=True,
        )
        return [layer for layer, _ in sorted_layers[:n_top]]

    def best_n_positions(self, threshold: float = 0.8) -> int:
        """Min positions for recovery > threshold."""
        for n in sorted(self.position_results.keys()):
            if self.position_results[n].score() > threshold:
                return n
        return max(self.position_results.keys()) if self.position_results else 0

    def get_union_target(
        self,
        n_top_layers: int = 3,
        position_threshold: float = 0.8,
        component: str = "resid_post",
    ) -> InterventionTarget:
        """Get target combining best layers and positions."""
        layers = self.best_layers(n_top=n_top_layers)
        n_pos = self.best_n_positions(threshold=position_threshold)
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

    def mean_sanity_score(self) -> float:
        """Mean sanity check score across samples."""
        scores = [
            r.sanity_result.score() for r in self.by_sample.values() if r.sanity_result
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def get_mean_layer_scores(self) -> dict[int, float]:
        """Mean recovery per layer across all samples."""
        by_layer: dict[int, list[float]] = {}
        for result in self.by_sample.values():
            for layer, target_result in result.layer_results.items():
                by_layer.setdefault(layer, []).append(target_result.score())
        return {l: sum(s) / len(s) for l, s in by_layer.items()}

    def get_mean_position_scores(self) -> dict[int, float]:
        """Mean recovery per position across all samples."""
        by_pos: dict[int, list[float]] = {}
        for result in self.by_sample.values():
            for pos, target_result in result.position_results.items():
                by_pos.setdefault(pos, []).append(target_result.score())
        return {p: sum(s) / len(s) for p, s in by_pos.items()}

    def best_layers(self, n_top: int = 3) -> list[int]:
        """Top n layers by mean score."""
        layer_scores = self.get_mean_layer_scores()
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        return [layer for layer, _ in sorted_layers[:n_top]]

    def get_union_target(
        self,
        n_top_layers: int = 3,
        position_threshold: float = 0.8,
        component: str = "resid_post",
    ) -> InterventionTarget:
        """Get target combining best layers and positions across all samples."""
        layers = self.best_layers(n_top=n_top_layers)

        # Find min position where mean score > threshold
        pos_scores = self.get_mean_position_scores()
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
        best = self.best_layers(n_top=3)
        if best:
            layer_scores = self.get_mean_layer_scores()
            scores_str = [f"{layer_scores[l]:.3f}" for l in best]
            print(f"  Best layers: {best} (scores: {scores_str})")


def run_coarse_act_patching(
    runner: BinaryChoiceRunner,
    pair: ContrastivePair,
    component: str = "resid_post",
    min_layer_depth: float = 0.01,
    max_layer_depth: float = 0.99,
    layer_step: int = 8,
    pos_step: int = 10,
) -> CoarseActPatchResults:
    """Run sanity check, layer sweep, and position sweep on single pair."""

    # Sanity
    print("[coarse] Starting sanity check (all layers, all positions)...")
    sanity_target = InterventionTarget.all(component=component)
    sanity_target = InterventionTarget.at_positions(
        range(len(pair.long_traj.token_ids)), component=component
    )
    sanity_result = patch_target(runner, pair, sanity_target)
    print(f"[coarse] Sanity check done: recovery={sanity_result.score():.3f}")

    # Layer
    n_layers = len(pair.available_layers)
    start_layer = int(n_layers * min_layer_depth)
    end_layer = int(n_layers * max_layer_depth)
    layers_of_interest = pair.available_layers[start_layer:end_layer]
    layer_results = {}
    print(
        f"[coarse] Starting layer sweep from {layers_of_interest[0]} to {layers_of_interest[-1]} with {layer_step} step-size..."
    )
    for i in range(0, len(layers_of_interest), layer_step):
        layer_range = layers_of_interest[i : i + layer_step]
        target = InterventionTarget.at_layers(layer_range, component=component)
        layer_results[layer_range[0]] = patch_target(runner, pair, target)
        print(
            f"[coarse] Layers:{layer_range} recovery={layer_results[layer_range[0]].score():.3f}, {i // layer_step + 1}/{-(-len(layers_of_interest) // layer_step)}"
        )

    # Pos
    position_results = {}
    start_pos = pair.position_mapping.first_interesting_pos
    end_pos = min(pair.choice_divergent_positions)
    print(
        f"[coarse] Starting position sweep from {start_pos} to {end_pos} with {pos_step} step-size..."
    )
    for pos in range(start_pos, end_pos, pos_step):
        pos_range = list(range(pos, min(pos + pos_step, end_pos)))
        target = InterventionTarget.at_positions(pos_range, component=component)
        position_results[pos] = patch_target(runner, pair, target)
        print(f"[coarse  pos={pos} recovery={position_results[pos].score():.3f}")

    clear_gpu_memory()
    print("[coarse] Done.")
    return CoarseActPatchResults(
        sanity_result=sanity_result,
        layer_results=layer_results,
        position_results=position_results,
    )
