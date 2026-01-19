#!/usr/bin/env python
"""
Generate intertemporal preference dataset.

Usage:
    python scripts/generate_dataset.py --config housing --output [PATH_TO_FOLDER]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap path before imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets import DatasetGenerator, DatasetConfig
from src.common.io import (
    parse_file_path,
    save_json,
    get_timestamp,
    ensure_dir,
)


PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = Path(__file__).parent


def save_dataset_as_json(samples, cfg: DatasetConfig, output_filepath: Path):
    from dataclasses import asdict

    ensure_dir(output_filepath.parent)
    data = {
        "dataset_id": cfg.get_id(),
        "timestamp": get_timestamp(),
        "config": cfg.to_dict(),
        "samples": [asdict(s) for s in samples],
    }
    save_json(data, output_filepath)


def generate_and_save_dataset(cfg: DatasetConfig, output_filepath: Path):
    """
    Generate dataset from dataset_config_path and save it in output_filepath.

    Args:
        cfg: Dataset configuration
        output_filepath: Path to save the dataset
    """
    dataset_id = cfg.get_id()

    generator = DatasetGenerator(cfg)
    samples = generator.generate()
    save_dataset_as_json(samples, cfg, output_filepath)

    print(f"Dataset saved to {output_filepath}")
    print(f"  - {len(samples)} prompts")
    print(f"  - Dataset ID: {dataset_id}")


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate intertemporal preference dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="*",
        default=["default"],
        help="Dataset config file path (or config file name from PROJECT_ROOT/scripts/data/configs/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dir (default: PROJECT_ROOT/out/datasets/)",
    )
    return parser.parse_args()


def parse_args(args):
    """Create input from command line arguments."""

    output_dirpath = args.output
    if output_dirpath is None:
        output_dirpath = PROJECT_ROOT / "out" / "datasets"

    runs = []
    for filename in args.config:
        # Get full json file path
        filepath = parse_file_path(
            filename, default_dir_path=str(SCRIPTS_DIR / "configs"), default_ext=".json"
        )
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset config not found: {filepath}")

        # Load dataset config
        config = DatasetGenerator.load_dataset_config(filepath)
        print(f"Loaded config: {config.name}")

        # Specify output filepath per dataset config
        output_filename = f"{config.name}_{config.get_id()}.json"
        output_filepath = output_dirpath / output_filename

        runs.append((config, Path(output_filepath)))

    return runs


def main() -> int:
    args = get_args()
    runs = parse_args(args)
    for cfg, output_filepath in runs:
        generate_and_save_dataset(cfg, output_filepath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
