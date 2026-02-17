"""Attribution patching experiment for identifying important layers and positions.

This module provides the high-level experiment interface for running attribution
patching on intertemporal preference choices, measuring which layers and positions
are most important for the model's time-horizon-dependent decisions.

Attribution patching uses gradients to approximate causal effects, providing
fast importance scores that can guide more expensive activation patching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ...common import BaseSchema, get_device
from ...inference.backends import get_recommended_backend_internals
from ...attribution_patching import (
    AttributionTarget,
    AggregatedAttributionResult,
    attribute_for_choice,
)
from ..common.contrastive_preferences import get_contrastive_preferences
from ...common.contrastive_pair import ContrastivePair

from ...binary_choice import BinaryChoiceRunner
from ..preference import PreferenceDataset


@dataclass
class IntertemporalAttributionConfig(BaseSchema):
    """Configuration for intertemporal attribution patching experiment.

    Attributes:
        n_pairs: Number of contrastive pairs to process
        target: Attribution target specification (positions, layers, methods)
        normalize_by_diff: If True, normalize scores by metric diff
    """

    n_pairs: int = 1
    target: AttributionTarget | dict = field(default_factory=AttributionTarget)
    normalize_by_diff: bool = True

    def __post_init__(self):
        if isinstance(self.target, dict):
            self.target = AttributionTarget(**self.target)

    def print_summary(self) -> None:
        print("Attribution patching config:")
        print(f"  methods={self.target.methods}")
        print(f"  layers={self.target.layers}")
        print(f"  components={self.target.components}")
        print(f"  n_pairs={self.n_pairs}")
        if "eap_ig" in self.target.methods:
            print(f"  ig_steps={self.target.ig_steps}")


def run_attribution_patching(
    pref_dataset: PreferenceDataset,
    cfg: IntertemporalAttributionConfig | None = None,
) -> AggregatedAttributionResult:
    """Run full attribution patching experiment on preference dataset.

    This function:
    1. Finds contrastive pairs (samples with different time horizons and choices)
    2. For each pair, captures activations and builds ContrastivePair
    3. Computes attribution scores across specified layers and positions
    4. Aggregates results across all pairs

    Args:
        pref_dataset: PreferenceDataset with samples to analyze
        cfg: Configuration for the experiment

    Returns:
        AggregatedAttributionResult with scores for each method
    """
    if cfg is None:
        cfg = IntertemporalAttributionConfig()

    all_pairs = get_contrastive_preferences(pref_dataset)
    if not all_pairs:
        print("No contrastive pairs found!")
        return AggregatedAttributionResult()

    pairs_to_process = all_pairs[: cfg.n_pairs]
    print(f"Found {len(all_pairs)} pairs, processing {len(pairs_to_process)}")

    cfg.print_summary()
    backend = get_recommended_backend_internals()
    runner = BinaryChoiceRunner(
        pref_dataset.model, device=get_device(), backend=backend
    )

    # Get anchor texts from prompt format config for position alignment
    prompt_format = pref_dataset.prompt_format_config
    anchor_texts = prompt_format.get_anchor_texts()

    pair_results: list[AggregatedAttributionResult] = []
    for i, contrastive_pref in enumerate(pairs_to_process):
        print(f"\nPair {i + 1}/{len(pairs_to_process)}")

        contrastive_pair = contrastive_pref.get_contrastive_pair(
            runner, anchor_texts=anchor_texts
        )
        if contrastive_pair is None:
            print("  Skipping: invalid sample data")
            continue
        contrastive_pair.print_summary()

        result = attribute_for_choice(
            runner=runner,
            contrastive_pair=contrastive_pair,
            target=cfg.target,
        )

        pair_results.append(result)
        result.print_summary()

    # Aggregate across pairs
    aggregated = AggregatedAttributionResult.aggregate(pair_results)

    print("\nExperiment complete:")
    aggregated.print_summary()
    return aggregated


def run_attribution_for_activation_targets(
    pref_dataset: PreferenceDataset,
    n_pairs: int = 5,
    n_targets: int = 10,
) -> list:
    """Run attribution to find best targets for activation patching.

    This is a convenience function that runs attribution patching and
    returns the top scoring positions as activation patching targets.

    Args:
        pref_dataset: PreferenceDataset with samples
        n_pairs: Number of pairs for attribution
        n_targets: Number of top targets to return

    Returns:
        List of ActivationPatchingTarget for top scoring positions
    """
    cfg = IntertemporalAttributionConfig(
        n_pairs=n_pairs,
        target=AttributionTarget.standard_only(),  # Fast mode
    )

    result = run_attribution_patching(pref_dataset, cfg)
    return result.get_consensus_targets(n=n_targets, min_methods=1)


def compare_attribution_methods(
    pref_dataset: PreferenceDataset,
    n_pairs: int = 3,
    ig_steps: int = 10,
) -> dict[str, float]:
    """Compare attribution methods by correlation with activation patching.

    Runs all attribution methods and measures their agreement.

    Args:
        pref_dataset: PreferenceDataset with samples
        n_pairs: Number of pairs to use
        ig_steps: Integration steps for EAP-IG

    Returns:
        Dict with method statistics
    """
    cfg = IntertemporalAttributionConfig(
        n_pairs=n_pairs,
        target=AttributionTarget.with_ig(steps=ig_steps),
    )

    result = run_attribution_patching(pref_dataset, cfg)

    stats = {}
    for name, res in result.results.items():
        stats[f"{name}_max"] = res.max_score
        stats[f"{name}_mean"] = res.mean_abs_score

    return stats
