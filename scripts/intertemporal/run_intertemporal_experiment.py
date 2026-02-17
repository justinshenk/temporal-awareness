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

from src.intertemporal.experiments.intertemporal_experiment import (
    ExperimentConfig,
    run_experiment,
)

from src.intertemporal.data.default_configs import (
    FULL_EXPERIMENT_CONFIG,
    MINIMAL_EXPERIMENT_CONFIG,
)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run full intertemporal preference experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--full", action="store_true", help="Runs with many samples")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., Qwen/Qwen3-4B-Instruct-2507)",
    )

    args = parser.parse_args()

    if args.full:
        config_dict = FULL_EXPERIMENT_CONFIG.copy()
    else:
        config_dict = MINIMAL_EXPERIMENT_CONFIG.copy()

    if args.model:
        config_dict["model"] = args.model

    return ExperimentConfig.from_dict(config_dict)


def main() -> int:
    exp_cfg = parse_args()
    run_experiment(exp_cfg)
    P.report()  # profiling


if __name__ == "__main__":
    raise SystemExit(main())
