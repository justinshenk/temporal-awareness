#!/usr/bin/env python3
"""Simple test script for activation patching implementation.

This script verifies that:
1. ContrastivePreferences can be found from a preference dataset
2. ContrastivePair can be built with cached activations
3. Patching experiments run successfully
4. Denoising improves recovery toward clean behavior

Usage:
    uv run python3 scripts/test_activation_patching.py --test
    uv run python3 scripts/test_activation_patching.py --dataset <path>
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.binary_choice import BinaryChoiceRunner
from src.intertemporal.preference import PreferenceDataset
from src.intertemporal.common.contrastive_preferences import (
    ContrastivePreferences,
    get_contrastive_preferences,
)
from src.activation_patching import patch_activation_for_choice
from src.inference.interventions import InterventionTarget
from src.intertemporal.experiments.activation_patching import (
    run_activation_patching,
    IntertemporalActivationPatchingConfig,
)


def test_contrastive_preferences(dataset: PreferenceDataset) -> list[ContrastivePreferences]:
    """Test finding contrastive preferences from a dataset."""
    print("\n=== Testing ContrastivePreferences ===")

    pairs = get_contrastive_preferences(dataset)
    print(f"Found {len(pairs)} contrastive pairs")

    if not pairs:
        print("WARNING: No contrastive pairs found!")
        print("  This may be expected if the dataset doesn't have samples")
        print("  with different time horizons leading to different choices.")
        return []

    # Show first pair details
    pair = pairs[0]
    print(f"\nFirst pair:")
    print(f"  Short-term choice sample:")
    print(f"    time_horizon: {pair.short_term.time_horizon}")
    print(f"    choice: {pair.short_term.choice_term}")
    print(f"  Long-term choice sample:")
    print(f"    time_horizon: {pair.long_term.time_horizon}")
    print(f"    choice: {pair.long_term.choice_term}")

    return pairs


def test_contrastive_pair(runner: BinaryChoiceRunner, pair: ContrastivePreferences):
    """Test building a ContrastivePair with cached activations."""
    print("\n=== Testing ContrastivePair ===")

    contrastive_pair = pair.get_contrastive_pair(runner)

    print(f"Short trajectory length: {contrastive_pair.short_length}")
    print(f"Long trajectory length: {contrastive_pair.long_length}")
    print(f"Available layers: {contrastive_pair.available_layers}")
    print(f"Available components: {contrastive_pair.available_components}")
    print(f"Position mapping entries: {len(contrastive_pair.position_mapping)}")

    return contrastive_pair


def test_single_patch(runner: BinaryChoiceRunner, pair: ContrastivePreferences):
    """Test a single patching experiment."""
    print("\n=== Testing Single Patch ===")

    contrastive_pair = pair.get_contrastive_pair(runner)

    # Get labels
    labels = (
        pair.short_term.short_term_label or "a)",
        pair.short_term.long_term_label or "b)",
    )

    # Patch all positions at layer 0
    layers = contrastive_pair.available_layers[:1]  # Just first layer
    if not layers:
        print("No layers available for patching")
        return

    targets = [InterventionTarget.all()]

    result = patch_activation_for_choice(
        runner=runner,
        contrastive_pair=contrastive_pair,
        targets=targets,
        layers=layers,
        mode="denoising",
        choice_prefix="I choose: ",
        labels=labels,
    )

    print(f"Number of interventions: {result.n_results}")
    print(f"Mean recovery: {result.mean_recovery:.4f}")
    print(f"Flip rate: {result.get_flip_rate():.2%}")

    if result.results:
        r = result.results[0]
        print(f"\nFirst result:")
        print(f"  Layer: {r.layer}")
        print(f"  Original choice: {r.original_chosen_label}")
        print(f"  Intervened choice: {r.intervened_chosen_label}")
        print(f"  Recovery: {r.recovery:.4f}")
        print(f"  Flipped: {r.choice_flipped}")


def test_full_experiment(runner: BinaryChoiceRunner, dataset: PreferenceDataset):
    """Test the full activation patching experiment."""
    print("\n=== Testing Full Experiment ===")

    cfg = IntertemporalActivationPatchingConfig(
        n_pairs=2,
        positions="all",
        component="resid_post",
        mode="denoising",
    )

    result = run_activation_patching(runner, dataset, cfg)

    print(f"\nExperiment Summary:")
    print(f"  Pairs processed: {result.n_pairs_processed}")
    print(f"  Mean recovery: {result.mean_recovery:.4f}")
    print(f"  Flip rate: {result.flip_rate:.2%}")

    best_layer, best_recovery = result.get_best_layer()
    if best_layer is not None:
        print(f"  Best layer: {best_layer} (recovery: {best_recovery:.4f})")


def find_default_dataset() -> str | None:
    """Find a default preference dataset to use for testing."""
    pref_dir = Path(__file__).parent.parent / "out" / "preference_datasets"
    if not pref_dir.exists():
        return None

    # Look for a small model dataset
    for pattern in ["*0.5B*", "*1.5B*", "*0.6B*"]:
        files = list(pref_dir.glob(pattern))
        if files:
            return str(files[0])

    # Fall back to any dataset
    files = list(pref_dir.glob("*.json"))
    if files:
        return str(files[0])

    return None


def main():
    parser = argparse.ArgumentParser(description="Test activation patching implementation")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test with auto-detected dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to preference dataset JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: uses model from dataset)",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip the full experiment test (faster)",
    )
    args = parser.parse_args()

    # Load or find test dataset
    dataset_path = args.dataset
    if dataset_path is None and args.test:
        dataset_path = find_default_dataset()
        if dataset_path:
            print(f"Auto-detected dataset: {dataset_path}")

    if dataset_path is None:
        print("No dataset provided or found.")
        print("Usage:")
        print("  --test              Auto-detect a dataset")
        print("  --dataset <path>    Use specific dataset")
        return 1

    print(f"Loading dataset from {dataset_path}")
    dataset = PreferenceDataset.from_json(dataset_path)

    print(f"Dataset: {dataset.prompt_dataset_id}")
    print(f"Model: {dataset.model}")
    print(f"Samples: {len(dataset.preferences)}")

    # Test contrastive preferences
    pairs = test_contrastive_preferences(dataset)
    if not pairs:
        print("\nNo pairs found. Cannot continue testing.")
        return 1

    # Determine model to use
    model_name = args.model or dataset.model
    print(f"\nLoading model: {model_name}")
    runner = BinaryChoiceRunner(model_name)

    # Test ContrastivePair building
    test_contrastive_pair(runner, pairs[0])

    # Test single patch
    test_single_patch(runner, pairs[0])

    # Test full experiment
    if not args.skip_full:
        test_full_experiment(runner, dataset)

    print("\n=== All tests completed! ===")
    return 0


if __name__ == "__main__":
    main()
