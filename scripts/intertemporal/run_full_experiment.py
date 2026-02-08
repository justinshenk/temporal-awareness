#!/usr/bin/env python
"""
Run the full intertemporal preference experiment.

Usage:
    # Quick test with minimal config
    uv run python scripts/intertemporal/run_full_experiment.py --test

    # Full pipeline
    uv run python scripts/intertemporal/run_full_experiment.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.profiler import P

from src.experiments.intertemporal_experiment import (
    ExperimentConfig,
    run_experiment,
)

from src.data.default_configs import (
    DEFAULT_EXPERIMENT_CONFIG,
    TEST_EXPERIMENT_CONFIG,
)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run full intertemporal preference experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test", action="store_true", help="Use minimal config for testing"
    )
    parser.add_argument(
        "--skip",
        type=str,
        nargs="*",
        default=["activation_patching", "attribution_patching", "steering_eval"],
        help="Steps to skip: activation_patching, attribution_patching, contrastive_steering, probe_training, steering_eval",
    )

    args = parser.parse_args()

    config_dict = DEFAULT_EXPERIMENT_CONFIG
    if args.test:
        config_dict = TEST_EXPERIMENT_CONFIG
    config_dict["skip"] = args.skip

    return ExperimentConfig.from_dict(config_dict)


def main() -> int:
    exp_cfg = parse_args()
    run_experiment(exp_cfg)
    P.report()  # profiling


if __name__ == "__main__":
    raise SystemExit(main())
