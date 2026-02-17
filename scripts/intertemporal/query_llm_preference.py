#!/usr/bin/env python
"""
Query language models with intertemporal preference dataset.

Usage:
    python scripts/intertemporal/query_llm_preference.py --config default
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap path before imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.file_io import parse_file_path, load_json
from src.intertemporal.common.project_paths import (
    get_pref_dataset_dir,
    get_query_configs_dir,
)
from src.intertemporal.data.default_configs import (
    DEFAULT_MODEL,
    TEST_PROMPT_DATASET_CONFIG,
)
from src.intertemporal.preference import (
    PreferenceQuerier,
    PreferenceQueryConfig,
    PreferenceDataset,
)
from src.intertemporal.prompt import (
    PromptDatasetConfig,
    PromptDatasetGenerator,
    PromptDataset,
)


# Default query config for querying models
DEFAULT_QUERY_CONFIG = {
    "models": [DEFAULT_MODEL],
    "internals": None,
    "subsample": 1.0,
    "batch_size": 4,
    "skip_generation": True,  # Fast: infer choice from probs only
}


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query language models with intertemporal preference dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Query config file path (or config name from configs/query/). "
        "If not provided, uses built-in DEFAULT_QUERY_CONFIG.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Dataset IDs to query. Required if not using a config file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=get_pref_dataset_dir(),
        help=f"Output dir (default: {get_pref_dataset_dir()})",
    )
    return parser.parse_args()


def generate_test_dataset() -> str:
    """Generate a test dataset and return its ID."""
    prompt_dataset_cfg = PromptDatasetConfig.from_dict(TEST_PROMPT_DATASET_CONFIG)
    generator = PromptDatasetGenerator(prompt_dataset_cfg)
    dataset = generator.generate()
    dataset.save_as_json()

    print("Using built-in TEST_PROMPT_DATASET_CONFIG")
    return dataset.dataset_id


def load_config(args) -> PreferenceQueryConfig:
    """Load query config from args."""
    if args.config is None:
        config_dict = dict(DEFAULT_QUERY_CONFIG)
    else:
        filepath = parse_file_path(
            args.config,
            default_dir_path=str(get_query_configs_dir()),
            default_ext=".json",
        )
        if not filepath.exists():
            raise FileNotFoundError(f"Query config not found: {filepath}")
        config_dict = load_json(filepath)

    if args.datasets:
        config_dict["datasets"] = args.datasets
    if not config_dict.get("datasets") or len(config_dict["datasets"]) == 0:
        dataset_id = generate_test_dataset()
        config_dict["datasets"] = [dataset_id]
        print("Using generate_test_dataset")
    if not config_dict.get("models") or len(config_dict["models"]) == 0:
        config_dict["models"] = DEFAULT_QUERY_CONFIG["models"]
        print("Using DEFAULT_QUERY_CONFIG['models']")

    return (
        PreferenceQueryConfig.from_dict(config_dict),
        config_dict["datasets"],
        config_dict["models"],
    )


def print_summary(pref_dataset: PreferenceDataset) -> None:
    print("\n\n")
    print(pref_dataset.to_string())
    print("\n\n")

    # Print summary
    short_count = sum(1 for p in pref_dataset.preferences if p.choice == "short_term")
    long_count = sum(1 for p in pref_dataset.preferences if p.choice == "long_term")
    print(f"  Total: {len(pref_dataset.preferences)}")
    print(f"  Short-term: {short_count}, Long-term: {long_count}")


def main() -> int:
    args = get_args()
    config, prompt_datasets, model_names = load_config(args)
    output_dir = args.output

    runner = PreferenceQuerier(config)

    for dataset_id in prompt_datasets:
        for model_name in model_names:
            print(f"\n{'=' * 60}")
            print(f"Dataset: {dataset_id} | Model: {model_name}")
            print(f"{'=' * 60}")

            prompt_dataset = PromptDataset.load_from_id(dataset_id)
            pref_dataset = runner.query_dataset(prompt_dataset, model_name)

            output_path = output_dir / pref_dataset.get_filename()
            pref_dataset.save_as_json(output_path)
            print_summary(pref_dataset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
