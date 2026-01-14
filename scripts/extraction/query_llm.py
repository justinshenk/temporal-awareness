#!/usr/bin/env python
"""
Query language models with intertemporal preference dataset.

Usage:
    python scripts/extraction/query_llm.py --config default
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import ensure_dir, parse_file_path, save_json
from src.models import QueryRunner, QueryOutput


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query language models with intertemporal preference dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="*",
        default=["default"],
        help="Query config file path (or config name from configs/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dir (default: PROJECT_ROOT/out/preference_data/)",
    )
    return parser.parse_args()


def parse_args(args):
    """Create runs from command line arguments."""
    output_dir = args.output or (PROJECT_ROOT / "out" / "preference_data")
    datasets_dir = PROJECT_ROOT / "out" / "datasets"

    runs = []
    for filename in args.config:
        filepath = parse_file_path(
            filename,
            default_dir_path=str(SCRIPTS_DIR / "configs"),
            default_ext=".json",
        )
        if not filepath.exists():
            raise FileNotFoundError(f"Query config not found: {filepath}")

        config = QueryRunner.load_config(filepath)
        print(f"Loaded config: {filename}")
        runs.append((config, output_dir, datasets_dir))

    return runs


def save_output(output: QueryOutput, output_dir: Path) -> None:
    """Save query output to JSON and internals to .pt files."""
    ensure_dir(output_dir)
    internals_dir = output_dir / "internals"

    model_name = output.model.split("/")[-1]

    # Prepare JSON-serializable output
    json_output = {
        "dataset_id": output.dataset_id,
        "model": output.model,
        "preferences": [],
    }

    for pref in output.preferences:
        pref_dict = {
            "sample_id": pref.sample_id,
            "time_horizon": pref.time_horizon,
            "short_term_label": pref.short_term_label,
            "long_term_label": pref.long_term_label,
            "choice": pref.choice,
            "choice_prob": pref.choice_prob,
            "alt_prob": pref.alt_prob,
            "response": pref.response,
            "internals": None,
        }

        # Save internals to .pt file if present
        if pref.internals is not None:
            ensure_dir(internals_dir)
            filename = f"{output.dataset_id}_{model_name}_sample_{pref.sample_id}.pt"
            file_path = internals_dir / filename
            torch.save(pref.internals.activations, file_path)

            pref_dict["internals"] = {
                "file_path": str(file_path),
                "activations": pref.internals.activation_names,
            }

        json_output["preferences"].append(pref_dict)

    # Save JSON output
    output_path = output_dir / f"{output.dataset_id}_{model_name}.json"
    save_json(json_output, output_path)
    print(f"Saved: {output_path}")


def main() -> int:
    args = get_args()
    runs = parse_args(args)

    for config, output_dir, datasets_dir in runs:
        runner = QueryRunner(config, datasets_dir)

        for dataset_id in config.datasets:
            for model_name in config.models:
                print(f"\n{'=' * 60}")
                print(f"Dataset: {dataset_id} | Model: {model_name}")
                print(f"{'=' * 60}")

                output = runner.query_dataset(dataset_id, model_name)
                save_output(output, output_dir)
                output.print_summary()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
