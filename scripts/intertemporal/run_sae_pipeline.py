#!/usr/bin/env python
"""
Temporal SAE Training Pipeline

Trains Sparse Autoencoders on sentence-level activations from LLM responses
to temporal preference questions. Runs iteratively: each iteration generates
new samples, extracts activations, continues training the same SAEs on
accumulated data, and evaluates.

Usage:
    python run_sae_pipeline.py                       # Show status + resume pipeline
    python run_sae_pipeline.py --resume              # Resume from checkpoint
    python run_sae_pipeline.py --test-iter           # Quick smoke test (6 samples, 1 iteration)
    python run_sae_pipeline.py --new                 # Start new pipeline from experiment_cfg.json
    python run_sae_pipeline.py --config other.json   # Use alternate config
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.sae.state import (
    PipelineConfig,
    PipelineState,
    find_state,
    find_latest_state,
    show_status,
)
from src.sae.pipeline import run_pipeline, run_test_iteration

# Default config location
DEFAULT_CONFIG = PROJECT_ROOT / "src" / "sae" / "experiment_cfg.json"


def get_args():
    parser = argparse.ArgumentParser(description="Temporal SAE Training Pipeline")
    parser.add_argument(
        "--resume",
        type=str,
        dest="pipeline_id",
        default="",
        help="Resume from checkpoint by pipeline_id",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        dest="start_new_pipeline",
        help="Start a new pipeline instead of resuming",
    )
    parser.add_argument(
        "--test-iter",
        action="store_true",
        help="Copy run_data/ to test_iter/, run one small iteration there, print results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to experiment config JSON",
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Load config from JSON
    with open(args.config) as f:
        cfg = json.load(f)
    config = PipelineConfig.from_dict(cfg)

    # Load prev pipeline
    if args.pipeline_id:
        latest_state = find_state(args.pipeline_id)
        if not latest_state:
            print(f"\n\n\nCannot load pipeline_id: {args.pipeline_id}\n\n\n")
            return 0
    else:
        latest_state = find_latest_state()
        print(f"\n\n\nlatest_state is: {latest_state}\n\n\n")

    if args.start_new_pipeline or latest_state is None:
        current_state = PipelineState.create_new(config)
        print(
            f"\n\n\nCreated new pipeline with pipeline_id: {current_state.pipeline_id} \n\n\n"
        )
    else:
        current_state = latest_state
        current_state.update_config(config)
    show_status(current_state)

    if args.test_iter:
        run_test_iteration(current_state)
    else:
        run_pipeline(current_state)


if __name__ == "__main__":
    main()
