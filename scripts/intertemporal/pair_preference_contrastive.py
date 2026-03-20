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
    return parser.parse_args()


def print_summary(pref_dataset: PreferenceDataset) -> None:
    """Print preference analysis."""
    analysis = analyze_preferences(pref_dataset)
    print_analysis(analysis)


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
    return 0

    best_only = args.best_only and not args.all_pairs

    print(f"\n{'=' * 60}")
    print("Finding contrastive pairs")
    print(f"  group_by: {args.group_by}")
    print(f"  best_only: {best_only}")
    print(f"  min_confidence: {args.min_confidence}")
    print(f"  deduplicate: {args.deduplicate}")
    print(f"{'=' * 60}")

    contrastive_pairs = get_contrastive_preferences(
        pref_dataset,
        group_by=args.group_by,
        best_only=best_only,
        min_confidence=args.min_confidence,
        deduplicate=args.deduplicate,
    )

    print_contrastive_pairs(contrastive_pairs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
