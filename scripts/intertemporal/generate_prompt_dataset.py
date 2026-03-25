#!/usr/bin/env python
"""
Generate intertemporal preference prompt dataset.

Usage:
    python scripts/intertemporal/generate_prompt_dataset.py --config housing --output [PATH_TO_FOLDER]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Bootstrap path before imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.intertemporal.prompt import PromptDatasetGenerator, PromptDatasetConfig
from src.common.file_io import parse_file_path
from src.intertemporal.common.project_paths import (
    get_prompt_dataset_dir,
    get_prompt_dataset_configs_dir,
)
from src.intertemporal.data.default_configs import FULL_EXPERIMENT_CONFIG


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
    print("\nDataset contents:")
    for i, sample in enumerate(dataset.samples):
        print(f"\n--- Sample {i + 1} ---")
        print(sample)

    # Print time and reward ranges table
    print_dataset_ranges(generator, dataset)

    return dataset.dataset_id


def print_dataset_ranges(generator: PromptDatasetGenerator, dataset) -> None:
    """Print time and reward ranges table and samples per intersection."""
    cfg = generator.dataset_config

    # Generate the actual step values
    short_term_grid = generator.generate_option_grid("short_term")
    long_term_grid = generator.generate_option_grid("long_term")

    # Extract unique reward and time values
    st_rewards = sorted(set(r for r, _ in short_term_grid))
    st_times = sorted(set(t for _, t in short_term_grid), key=lambda t: t.to_months())
    lt_rewards = sorted(set(r for r, _ in long_term_grid))
    lt_times = sorted(set(t for _, t in long_term_grid), key=lambda t: t.to_months())
    horizons = cfg.time_horizons

    # Count samples per intersection FIRST
    print("\n" + "=" * 60)
    print("SAMPLES PER INTERSECTION")
    print("=" * 60)

    # Store (st_time_obj, lt_time_obj, horizon_obj) -> count for proper sorting
    time_horizon_counts: dict[tuple, int] = {}
    for sample in dataset.samples:
        pair = sample.prompt.preference_pair
        st = pair.short_term
        lt = pair.long_term
        horizon = sample.prompt.time_horizon
        key = (st.time, lt.time, horizon)
        time_horizon_counts[key] = time_horizon_counts.get(key, 0) + 1

    # Sort by actual time values (using to_months for comparison)
    def sort_key(item):
        st_time, lt_time, horizon = item[0]
        st_months = st_time.to_months()
        lt_months = lt_time.to_months()
        horizon_months = horizon.to_months() if horizon else -1
        return (st_months, lt_months, horizon_months)

    print(f"\n{'ST Time':<12} {'LT Time':<12} {'Horizon':<12} {'# Samples':<10}")
    print("-" * 46)
    for (st_time, lt_time, horizon), count in sorted(time_horizon_counts.items(), key=sort_key):
        horizon_str = str(horizon) if horizon else "None"
        print(f"{str(st_time):<12} {str(lt_time):<12} {horizon_str:<12} {count:<10}")

    print(f"\nTotal intersections: {len(time_horizon_counts)}")
    print(f"Total samples: {len(dataset.samples)}")

    # Time and reward ranges SECOND
    print("\n" + "=" * 60)
    print("TIME AND REWARD RANGES")
    print("=" * 60)

    # Get scale types from config
    st_cfg = cfg.options["short_term"]
    lt_cfg = cfg.options["long_term"]
    st_reward_scale = st_cfg.reward_steps[1].value
    st_time_scale = st_cfg.time_steps[1].value
    lt_reward_scale = lt_cfg.reward_steps[1].value
    lt_time_scale = lt_cfg.time_steps[1].value

    # Short-term section
    print("\n┌─ Short-term")
    print(f"│  Rewards ({st_reward_scale}):")
    for r in st_rewards:
        print(f"│    • ${r:,.0f}")
    print(f"│  Times ({st_time_scale}):")
    for t in st_times:
        print(f"│    • {t}")

    # Long-term section
    print("│")
    print("├─ Long-term")
    print(f"│  Rewards ({lt_reward_scale}):")
    for r in lt_rewards:
        print(f"│    • ${r:,.0f}")
    print(f"│  Times ({lt_time_scale}):")
    for t in lt_times:
        print(f"│    • {t}")

    # Time horizons section
    print("│")
    print("└─ Time Horizons:")
    for h in horizons:
        print(f"     • {h if h else 'None'}")


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
        "If not provided, uses FULL_EXPERIMENT_CONFIG.",
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
        config = PromptDatasetConfig.from_dict(FULL_EXPERIMENT_CONFIG["dataset_config"])
        runs.append(config)
        print("Using FULL_EXPERIMENT_CONFIG:")
        for key, value in FULL_EXPERIMENT_CONFIG.items():
            print(f"  {key}: {value}")
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
