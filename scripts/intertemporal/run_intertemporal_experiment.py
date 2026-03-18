#!/usr/bin/env python
"""
Run the full intertemporal preference experiment.

Usage:
    # Quick test with minimal config
    uv run python scripts/intertemporal/run_intertemporal_experiment.py

    # Full pipeline with many samples
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --full

    # Use cached data
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --cache

    # Use cached data from a specific folder
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --cache my_experiment

    # Save to a custom folder name
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --rename my_experiment

    # Override coarse patching settings
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --coarse '{"components": ["mlp_out"]}'

    # Enable attribution patching with specific methods
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --attrib '{"enabled": true, "methods": ["eap_ig"]}'

    # Regenerate visualizations for all cached experiments
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --viz '{"regenerate_all": true}'

    # Regenerate visualizations for one specific experiment
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --viz '{"regenerate_one": "label_grid"}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.profiler import P
from src.common.logging import set_log_file, close_log_file, log, log_header, log_kv
from src.intertemporal.common import get_experiment_dir
from src.intertemporal.experiments.intertemporal_experiment import (
    ExperimentConfig,
    run_experiment,
)
from src.intertemporal.experiments.intertemporal_viz import (
    generate_viz,
    regenerate_all_visualizations,
)

from src.intertemporal.data.default_configs import (
    FULL_EXPERIMENT_CONFIG,
    MINIMAL_EXPERIMENT_CONFIG,
)
from src.intertemporal.viz.coarse.component_comparison.constants import COMPONENTS


def detect_cached_components(exp_dir: Path) -> list[str]:
    """Detect which components have cached coarse patching results."""
    cached = []
    pair_0 = exp_dir / "pair_0"
    if not pair_0.exists():
        return cached
    for comp in COMPONENTS:
        if (pair_0 / f"sweep_{comp}" / "coarse_results.json").exists():
            cached.append(comp)
    return cached


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--cache",
        nargs="?",
        const=True,
        default=False,
        metavar="FOLDER",
        help="Try loading cached data. Optionally specify folder name to load from.",
    )
    parser.add_argument(
        "--rename",
        type=str,
        default=None,
        metavar="NAME",
        help="Custom folder name for output (only works without --cache)",
    )
    parser.add_argument(
        "--coarse",
        type=str,
        default=None,
        metavar="JSON",
        help='Override coarse patching settings as JSON, e.g. \'{"components": ["mlp_out"]}\'',
    )
    parser.add_argument(
        "--attrib",
        type=str,
        default=None,
        metavar="JSON",
        help='Override attribution patching settings as JSON, e.g. \'{"enabled": true, "methods": ["eap_ig"]}\'',
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["pyvene", "transformerlens", "huggingface", "nnsight"],
        help="Override backend for model internals (default: auto-detect)",
    )
    parser.add_argument(
        "--viz",
        type=str,
        default=None,
        metavar="JSON",
        help='Visualization settings as JSON, e.g. \'{"enabled": true, "regenerate_all": true}\'',
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Handle --viz with regenerate options first (doesn't need other config)
    if args.viz:
        viz_config = json.loads(args.viz)
        if viz_config.get("regenerate_all", False):
            log_header("REGENERATING ALL VISUALIZATIONS", gap=1)
            regenerate_all_visualizations(get_experiment_dir())
            return 0
        if "regenerate_one" in viz_config:
            exp_name = viz_config["regenerate_one"]
            exp_dir = get_experiment_dir() / exp_name
            if not exp_dir.exists():
                print(f"Error: Experiment directory not found: {exp_dir}")
                return 1
            log_header(f"REGENERATING VISUALIZATIONS: {exp_name}", gap=1)
            generate_viz(exp_dir)
            return 0

    if args.full:
        config_dict = FULL_EXPERIMENT_CONFIG.copy()
    else:
        config_dict = MINIMAL_EXPERIMENT_CONFIG.copy()

    if args.model:
        config_dict["model"] = args.model

    if args.coarse:
        coarse_overrides = json.loads(args.coarse)
        if "coarse_patch" not in config_dict:
            config_dict["coarse_patch"] = {}
        config_dict["coarse_patch"].update(coarse_overrides)

    if args.attrib:
        attrib_overrides = json.loads(args.attrib)
        if "att_patch" not in config_dict:
            config_dict["att_patch"] = {}
        config_dict["att_patch"].update(attrib_overrides)

    if args.viz:
        viz_overrides = json.loads(args.viz)
        if "viz" not in config_dict:
            config_dict["viz"] = {}
        config_dict["viz"].update(viz_overrides)

    # Determine output directory
    output_dir = None
    try_loading_data = bool(args.cache)

    if args.cache and isinstance(args.cache, str):
        # --cache FOLDER: load from specific folder
        output_dir = get_experiment_dir() / args.cache

        # Auto-detect cached components
        cached = detect_cached_components(output_dir)
        if cached:
            if "coarse_patch" not in config_dict:
                config_dict["coarse_patch"] = {}
            config_dict["coarse_patch"]["components"] = cached
            print(f"Auto-detected cached components: {cached}")
    elif args.rename:
        if args.cache:
            print("Warning: --rename is ignored when --cache is used")
        else:
            # --rename NAME: use custom folder name
            output_dir = get_experiment_dir() / args.rename

    # Create experiment config after all config modifications
    exp_cfg = ExperimentConfig.from_dict(config_dict)

    # If no custom output_dir, use default based on config ID
    if output_dir is None:
        output_dir = get_experiment_dir() / exp_cfg.get_id()

    # Set up logging to file
    output_dir.mkdir(parents=True, exist_ok=True)
    set_log_file(output_dir / "log.txt")

    log_header(f"EXPERIMENT: {exp_cfg.get_id()}", gap=1)
    log_kv("Output", str(output_dir))
    log()

    try:
        run_experiment(
            exp_cfg,
            try_loading_data=try_loading_data,
            output_dir=output_dir,
            backend=args.backend,
        )
        P.report()
        P.save(output_dir / "profile.json")
    finally:
        close_log_file()


if __name__ == "__main__":
    raise SystemExit(main())
