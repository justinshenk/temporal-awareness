#!/usr/bin/env python
"""
Load preference datasets and find contrastive preference pairs.

Usage:
    python scripts/intertemporal/pair_preference_contrastive.py
    python scripts/intertemporal/pair_preference_contrastive.py --input path/to/preferences.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap path before imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.intertemporal.common.project_paths import get_pref_dataset_dir
from src.intertemporal.data.default_configs import FULL_EXPERIMENT_CONFIG
from src.intertemporal.preference import (
    PreferenceDataset,
    analyze_preferences,
    print_analysis,
)
from src.intertemporal.prompt import PromptDatasetConfig
from src.intertemporal.common.contrastive_utils import get_contrastive_preferences
from src.intertemporal.common.contrastive_analysis import print_contrastive_pairs


def get_default_pref_dataset_path() -> Path:
    """Derive the default preference dataset path from FULL_EXPERIMENT_CONFIG."""
    config = PromptDatasetConfig.from_dict(FULL_EXPERIMENT_CONFIG["dataset_config"])
    model = FULL_EXPERIMENT_CONFIG["model"]
    model_name = model.split("/")[-1]
    prompt_dataset_id = config.get_id()
    filename = f"{prompt_dataset_id}_{model_name}_{config.name}.json"
    return get_pref_dataset_dir() / filename


def get_args():
    """Parse command line arguments."""
    default_path = get_default_pref_dataset_path()
    parser = argparse.ArgumentParser(
        description="Load preference datasets and find contrastive preference pairs"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_path,
        help=f"Path to preference dataset JSON file (default: {default_path})",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="choice",
        choices=["content", "horizon", "choice"],
        help="Grouping mode for contrastive pairs (default: horizon)",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        default=False,
        help="Only keep the single best pair per group (default: True)",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Keep all pairs, not just best (overrides --best-only)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum choice probability threshold (default: 0.6)",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Keep only one pair per unique content×horizon combination",
    )
    parser.add_argument(
        "--max-per-sample",
        type=int,
        default=None,
        help="Max pairs each sample can participate in (reduces 975 -> ~50-100)",
    )
    parser.add_argument(
        "--max-per-horizon",
        type=int,
        default=None,
        help="Max pairs per horizon combination (stratified by horizon diversity)",
    )
    parser.add_argument(
        "--max-per-ratio",
        type=int,
        default=None,
        help="Max pairs per reward ratio (stratified by preference strength)",
    )
    parser.add_argument(
        "--max-per-confidence",
        type=int,
        default=None,
        help="Max pairs per confidence bucket (stratified across 0.5-1.0 range)",
    )
    parser.add_argument(
        "--smart-reduce",
        type=str,
        choices=["balanced", "diverse", "minimal"],
        default=None,
        help="Convenience presets: balanced (~60-80), diverse (~40-60), minimal (~20-30)",
    )
    parser.add_argument(
        "--prefer-different-horizon",
        action="store_true",
        help="Prioritize pairs where short/long have different horizons",
    )
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=None,
        help="Target number of pairs (auto-calculates max_per_sample to achieve)",
    )
    parser.add_argument(
        "--selection-strategy",
        type=str,
        choices=["greedy", "round_robin"],
        default="greedy",
        help="Selection strategy: greedy (highest confidence first) or round_robin (cycle horizon groups)",
    )
    return parser.parse_args()


def print_summary(pref_dataset: PreferenceDataset) -> None:
    """Print preference analysis."""
    analysis = analyze_preferences(pref_dataset)
    print_analysis(analysis)

    print(f"\n{'=' * 150}")
    print(f"\n{'=' * 150}")
    print(f"{'\n' * 20}")
    print(f"\n{'=' * 150}")
    print(f"\n{'=' * 150}")


def main() -> int:
    args = get_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    print(f"\n{'=' * 60}")
    print(f"Loading preference dataset: {args.input}")
    print(f"{'=' * 60}")

    pref_dataset = PreferenceDataset.from_json(str(args.input))

    print(f"\nDataset: {pref_dataset.prompt_dataset_id} | Model: {pref_dataset.model}")
    print(f"Samples: {len(pref_dataset.preferences)}")

    print_summary(pref_dataset)

    best_only = args.best_only and not args.all_pairs

    print(f"\n{'=' * 60}")
    print("Finding contrastive pairs")
    print(f"  group_by: {args.group_by}")
    print(f"  best_only: {best_only}")
    print(f"  min_confidence: {args.min_confidence}")
    print(f"  deduplicate: {args.deduplicate}")
    print(f"  max_per_sample: {args.max_per_sample}")
    print(f"  max_per_horizon: {args.max_per_horizon}")
    print(f"  max_per_ratio: {args.max_per_ratio}")
    print(f"  max_per_confidence: {args.max_per_confidence}")
    print(f"  smart_reduce: {args.smart_reduce}")
    print(f"  target_pairs: {args.target_pairs}")
    print(f"  selection_strategy: {args.selection_strategy}")
    print(f"{'=' * 60}")

    # Calculate baseline for comparison
    n_short = sum(1 for p in pref_dataset.preferences if p.choice_term == "short_term")
    n_long = sum(1 for p in pref_dataset.preferences if p.choice_term == "long_term")
    max_pairs = n_short * n_long

    contrastive_pairs = get_contrastive_preferences(
        pref_dataset,
        group_by=args.group_by,
        best_only=best_only,
        min_confidence=args.min_confidence,
        deduplicate=args.deduplicate,
        max_per_sample=args.max_per_sample,
        max_per_horizon_pair=args.max_per_horizon,
        max_per_reward_ratio=args.max_per_ratio,
        max_per_confidence_bucket=args.max_per_confidence,
        smart_reduce=args.smart_reduce,
        prefer_different_horizon=args.prefer_different_horizon,
        target_pairs=args.target_pairs,
        selection_strategy=args.selection_strategy,
    )

    # Show reduction stats
    reduction_pct = 100 * (1 - len(contrastive_pairs) / max_pairs) if max_pairs > 0 else 0
    print(f"\nReduction: {max_pairs} -> {len(contrastive_pairs)} pairs ({reduction_pct:.1f}% reduced)")

    print_contrastive_pairs(contrastive_pairs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
