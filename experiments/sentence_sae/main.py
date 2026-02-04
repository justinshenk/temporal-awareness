#!/usr/bin/env python
"""
Temporal SAE Training Pipeline

Trains Sparse Autoencoders on sentence-level activations from LLM responses
to temporal preference questions. Runs iteratively: each iteration generates
new samples, extracts activations, continues training the same SAEs on
accumulated data, and evaluates.

Usage:
    python main.py                       # Show status + resume pipeline
    python main.py --resume              # Resume from checkpoint
    python main.py --test-iter           # Quick smoke test (6 samples, 1 iteration)
    python main.py --new                 # Start new pipeline from experiment_cfg.json
    python main.py --config other.json   # Use alternate config
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.state import PipelineConfig, PipelineState, find_latest_state, show_status
from src.pipeline import run_pipeline, run_test_iteration


def get_args():
    parser = argparse.ArgumentParser(description="Temporal SAE Training Pipeline")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from last checkpoint (default)",
    )
    parser.add_argument(
        "--new",
        action="store_false",
        dest="resume",
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
        default=str(Path(__file__).parent / "experiment_cfg.json"),
        help="Path to experiment config JSON",
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Load config from JSON
    with open(args.config) as f:
        cfg = json.load(f)

    latest_state = find_latest_state()
    show_status(latest_state)

    config = PipelineConfig.from_dict(cfg)

    if not args.resume or latest_state is None:
        current_state = PipelineState.create_new(config)
        print(f"Created new pipeline: {current_state.pipeline_id}")
    else:
        current_state = latest_state
        current_state.update_config(config)

    if args.test_iter:
        run_test_iteration(current_state)
    else:
        run_pipeline(current_state)


if __name__ == "__main__":
    main()
