#!/usr/bin/env python
"""
Generate intertemporal preference prompt dataset.

Usage:
    python scripts/intertemporal/generate_prompt_dataset.py --config housing --output [PATH_TO_FOLDER]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap path before imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.intertemporal.prompt import PromptDatasetGenerator, PromptDatasetConfig
from src.common.file_io import parse_file_path
from src.intertemporal.common.project_paths import get_prompt_dataset_dir, get_prompt_dataset_configs_dir
from src.intertemporal.data.default_configs import TEST_PROMPT_DATASET_CONFIG


def generate_and_save_dataset(cfg: PromptDatasetConfig, output_dirpath: Path) -> str:
    """
    Generate prompt dataset and save it.

    Args:
        cfg: Prompt dataset configuration
        output_dirpath: Directory to save the dataset

    Returns:
        Dataset ID
    """
    output_filepath = Path(output_dirpath) / cfg.get_filename()

    generator = PromptDatasetGenerator(cfg)
    dataset = generator.generate()
    dataset.save_as_json(output_filepath)

    print(f"Dataset saved to {output_filepath}")
    print(f"  - {len(dataset.samples)} prompts")
    print(f"  - Dataset ID: {dataset.dataset_id}")

    return dataset.dataset_id


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate intertemporal preference prompt dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="*",
        default=None,
        help="Dataset config file path (or config name from configs/prompt_datasets/). "
        "If not provided, uses built-in TEST_PROMPT_DATASET_CONFIG.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        dest="output_prompt_dataset_dir",
        default=get_prompt_dataset_dir(),
        help=f"Output dir (defaults: {get_prompt_dataset_dir()}",
    )
    return parser.parse_args()


def parse_args(args):
    """Create input from command line arguments."""
    runs = []
    if not args.config:
        # Use built-in default config
        config = PromptDatasetConfig.from_dict(TEST_PROMPT_DATASET_CONFIG)
        runs.append(config)
        print(f"Using built-in TEST_PROMPT_DATASET_CONFIG: {config.name}")
    else:
        for filename in args.config:
            # Get full json file path
            filepath = parse_file_path(
                filename,
                default_dir_path=str(get_prompt_dataset_configs_dir()),
                default_ext=".json",
            )
            if not filepath.exists():
                raise FileNotFoundError(f"Dataset config not found: {filepath}")
            # Load dataset config
            config = PromptDatasetConfig.from_json(filepath)
            runs.append(config)
            print(f"Loaded config: {config.name}")

    return runs, args.output_prompt_dataset_dir


def main() -> int:
    args = get_args()
    runs, output_dirpath = parse_args(args)
    for cfg in runs:
        generate_and_save_dataset(cfg, output_dirpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
