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

from src.intertemporal.common.project_paths import (
    get_pref_dataset_dir,
)
from src.intertemporal.data.default_configs import MINIMAL_EXPERIMENT_CONFIG
from src.intertemporal.preference import (
    PreferenceQuerier,
    PreferenceQueryConfig,
    analyze_preferences,
    print_analysis,
)
from src.intertemporal.prompt import (
    PromptDatasetConfig,
    PromptDatasetGenerator,
    PromptDataset,
)


# Default query config for querying models
DEFAULT_QUERY_CONFIG = {
    "models": [MINIMAL_EXPERIMENT_CONFIG["model"]],
    "internals": None,
    "subsample": 1.0,
    "batch_size": 4,
    "skip_generation": False,  # Fast if True: infer choice from probs only
}


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query language models with intertemporal preference dataset"
    )
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        type=str,
        nargs="*",
        default=None,
        help="Dataset identifier(s): ID, path, filename, or directory name.",
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
    prompt_dataset_cfg = PromptDatasetConfig.from_dict(
        MINIMAL_EXPERIMENT_CONFIG["dataset_config"]
    )
    generator = PromptDatasetGenerator(prompt_dataset_cfg)
    dataset = generator.generate()
    dataset.save_as_json()

    print("Using MINIMAL_EXPERIMENT_CONFIG:")
    for key, value in MINIMAL_EXPERIMENT_CONFIG.items():
        print(f"  {key}: {value}")
    return dataset.dataset_id


def load_config(args) -> PreferenceQueryConfig:
    """Load query config from args."""
    config_dict = dict(DEFAULT_QUERY_CONFIG)

    if args.datasets:
        config_dict["datasets"] = args.datasets
    if not config_dict.get("datasets") or len(config_dict["datasets"]) == 0:
        dataset_id = generate_test_dataset()
        config_dict["datasets"] = [dataset_id]
        print("Using generate_test_dataset")
    if not config_dict.get("models") or len(config_dict["models"]) == 0:
        config_dict["models"] = MINIMAL_EXPERIMENT_CONFIG["model"]
        print(
            f"Using MINIMAL_EXPERIMENT_CONFIG model: {MINIMAL_EXPERIMENT_CONFIG['model']}"
        )

    return (
        PreferenceQueryConfig.from_dict(config_dict),
        config_dict["datasets"],
        config_dict["models"],
    )


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
            pref_dataset = runner.query_dataset(
                prompt_dataset, model_name, verbose=True
            )

            output_path = output_dir / pref_dataset.get_filename()
            pref_dataset.save_as_json(output_path)

            analysis = analyze_preferences(pref_dataset)
            print_analysis(analysis)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
