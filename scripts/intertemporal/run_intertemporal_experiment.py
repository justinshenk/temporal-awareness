#!/usr/bin/env python
"""
Run the full intertemporal preference experiment.

Usage:
    # Quick test with minimal config
    uv run python scripts/intertemporal/run_intertemporal_experiment.py

    # Full pipeline with many samples
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --full

    # Use cached data (loads from working_config.json if it exists)
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --cache

    # Use cached data from a specific folder
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --cache my_experiment

    # Update working_config.json with CLI overrides
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --cache my_experiment --update-config --coarse '{"enabled": false}'

    # Generate camera-ready figures (PNG + SVG)
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --camera-ready

    # Save to a custom folder name
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --output-dir-name my_experiment

    # Override coarse patching settings
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --coarse '{"components": ["mlp_out"]}'

    # Enable attribution patching with specific methods
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --attrib '{"enabled": true, "methods": ["eap_ig"]}'

    # Run ONLY diffmeans (disable all other steps except those specified)
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --disable --diffmeans '{"enabled": true}'

    # Run ONLY coarse patching
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --disable --coarse '{"enabled": true}'

    # Use a custom dataset config from JSON file
    uv run python scripts/intertemporal/run_intertemporal_experiment.py --dataset path/to/dataset_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.file_io import backup_dir, move_dir
from src.common.logging import close_log_file, log, log_header, log_kv, set_log_file
from src.common.profiler import P
from src.intertemporal.common import get_experiment_dir
from src.intertemporal.data.default_configs import (
    FULL_EXPERIMENT_CONFIG,
    MINIMAL_EXPERIMENT_CONFIG,
    MULTILABEL_EXPERIMENT_CONFIG,
)
from src.intertemporal.experiments.experiment_config import (
    ATTRIB_CFG,
    ATTN_CFG,
    COARSE_CFG,
    DIFFMEANS_CFG,
    FINE_CFG,
    MLP_CFG,
    PAIR_REQ_CFG,
    VIZ_CFG,
)
from src.intertemporal.experiments.intertemporal_experiment import (
    ExperimentConfig,
    run_experiment_per_pair,
)
# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full intertemporal preference experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Dataset selection ---
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run with full dataset (many samples)",
    )
    parser.add_argument(
        "--multilabel",
        action="store_true",
        help="Use multilabel dataset (do_formatting_variation_grid=True)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to JSON file containing dataset config (PromptDatasetConfig)",
    )

    # --- Output directory ---
    parser.add_argument(
        "--out",
        "--output-dir-name",
        dest="output_dir_name",
        type=str,
        default=None,
        metavar="NAME",
        help="Custom folder name for output (mutually exclusive with --cache)",
    )
    parser.add_argument(
        "--cache",
        nargs="?",
        const=True,
        default=False,
        metavar="FOLDER",
        help="Load cached data. Optionally specify folder name.",
    )

    # --- Model configuration ---
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., Qwen/Qwen3-4B-Instruct-2507)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["pyvene", "transformerlens", "huggingface", "nnsight"],
        help="Backend for model internals (default: auto-detect)",
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=None,
        help="Number of contrastive pairs to use",
    )

    # --- Pipeline control ---
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Disable all steps, then selectively enable with other flags",
    )
    parser.add_argument(
        "--only-viz-agg",
        action="store_true",
        help="Skip per-pair viz, only generate aggregated viz for all steps",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update working_config.json with CLI overrides (only with --cache)",
    )
    parser.add_argument(
        "--camera-ready",
        action="store_true",
        help="Save SVG files alongside PNGs for publication-quality figures",
    )

    # --- Phase overrides (JSON, ordered to match run_experiment) ---
    _add_json_override(parser, "--attrib", "att_patch", "attribution patching")
    _add_json_override(parser, "--coarse", "coarse_patch", "coarse patching")
    _add_json_override(parser, "--diffmeans", "diffmeans", "difference-of-means")
    _add_json_override(parser, "--mlp", "mlp", "MLP analysis")
    _add_json_override(parser, "--attn", "attn", "attention analysis")
    _add_json_override(parser, "--fine", "fine", "fine patching")
    _add_json_override(parser, "--viz", "viz", "visualization")
    _add_json_override(parser, "--pair_req", "pair_req", "pair requirements")

    return parser.parse_args()


def _add_json_override(
    parser: argparse.ArgumentParser,
    flag: str,
    config_key: str,
    description: str,
) -> None:
    """Add a JSON override argument.

    - Flag not used: default=None (no override)
    - Flag used without argument: const='{}' (enable with defaults)
    - Flag used with argument: uses provided JSON
    """
    parser.add_argument(
        flag,
        type=str,
        nargs='?',
        const='{}',
        default=None,
        metavar="JSON",
        help=f"Override {description} settings as JSON",
    )


# =============================================================================
# Config Building
# =============================================================================


def build_base_config(args: argparse.Namespace) -> dict:
    """Build base config from dataset selection."""
    if args.multilabel:
        config = MULTILABEL_EXPERIMENT_CONFIG.copy()
    elif args.full:
        config = FULL_EXPERIMENT_CONFIG.copy()
    else:
        config = MINIMAL_EXPERIMENT_CONFIG.copy()

    # Override dataset_config from JSON file if provided
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset config file not found: {dataset_path}")
        with open(dataset_path) as f:
            config["dataset_config"] = json.load(f)
        print(f"[dataset] Loaded dataset config from {dataset_path}")

    if args.model:
        config["model"] = args.model
    if args.n_pairs:
        config["n_pairs"] = args.n_pairs

    return config


def apply_disable_flag(config: dict) -> None:
    """Disable all pipeline steps (for selective re-enabling)."""
    config["attrib_cfg"] = {**ATTRIB_CFG, "enabled": False, "no_viz": True}
    config["coarse_cfg"] = {**COARSE_CFG, "enabled": False, "no_viz": True}
    config["diffmeans_cfg"] = {**DIFFMEANS_CFG, "enabled": False, "no_viz": True}
    config["mlp_cfg"] = {**MLP_CFG, "enabled": False, "no_viz": True}
    config["attn_cfg"] = {**ATTN_CFG, "enabled": False, "no_viz": True}
    config["fine_cfg"] = {**FINE_CFG, "enabled": False, "no_viz": True}


def apply_only_viz_agg_flag(config: dict) -> None:
    """Set only_viz_agg=True for all steps (skip per-pair viz, only aggregated)."""
    config.setdefault("attrib_cfg", ATTRIB_CFG.copy())["only_viz_agg"] = True
    config.setdefault("coarse_cfg", COARSE_CFG.copy())["only_viz_agg"] = True
    config.setdefault("diffmeans_cfg", DIFFMEANS_CFG.copy())["only_viz_agg"] = True
    config.setdefault("mlp_cfg", MLP_CFG.copy())["only_viz_agg"] = True
    config.setdefault("attn_cfg", ATTN_CFG.copy())["only_viz_agg"] = True
    config.setdefault("fine_cfg", FINE_CFG.copy())["only_viz_agg"] = True


def apply_camera_ready_flag(config: dict) -> None:
    """Enable SVG and PDF output alongside PNGs for publication-quality figures."""
    viz_cfg = config.setdefault("viz_cfg", VIZ_CFG.copy())
    viz_cfg["save_svg"] = True
    viz_cfg["save_pdf"] = True


def apply_json_overrides(config: dict, args: argparse.Namespace) -> None:
    """Apply all JSON override arguments to config.

    Args are None when not provided, '{}' when flag used without argument,
    or a JSON string when flag used with argument.
    """
    # Simple overrides (merge only, no auto-enable)
    simple_overrides = [
        (args.viz, "viz_cfg", VIZ_CFG),
        (args.pair_req, "pair_req_cfg", PAIR_REQ_CFG),
    ]
    for arg_value, key, default in simple_overrides:
        if arg_value is not None:
            config.setdefault(key, default.copy()).update(json.loads(arg_value))

    # Auto-enable overrides (passing flag implies enabled=True, ordered to match run_experiment)
    # Enable if arg is provided (not None)
    auto_enable_overrides = [
        (args.attrib, "attrib_cfg", ATTRIB_CFG),
        (args.coarse, "coarse_cfg", COARSE_CFG),
        (args.diffmeans, "diffmeans_cfg", DIFFMEANS_CFG),
        (args.mlp, "mlp_cfg", MLP_CFG),
        (args.attn, "attn_cfg", ATTN_CFG),
        (args.fine, "fine_cfg", FINE_CFG),
    ]
    for arg_value, key, default in auto_enable_overrides:
        if arg_value is not None:
            section = config.setdefault(key, default.copy())
            section["enabled"] = True
            section["no_viz"] = False
            section.update(json.loads(arg_value))


# =============================================================================
# Output Directory Resolution
# =============================================================================


def resolve_output_dir(
    args: argparse.Namespace,
    exp_cfg: ExperimentConfig,
) -> tuple[Path, bool]:
    """
    Resolve output directory and whether to load cached data.

    Returns:
        (output_dir, try_loading_data)
    """
    try_loading_data = bool(args.cache)

    # --cache FOLDER: load from specific folder
    if args.cache and isinstance(args.cache, str):
        output_dir = get_experiment_dir() / args.cache
        if output_dir.exists():
            backup_dir(output_dir)
        return output_dir, try_loading_data

    # --output-dir-name NAME: use custom folder name
    if args.output_dir_name:
        output_dir = get_experiment_dir() / args.output_dir_name
        if output_dir.exists():
            move_dir(output_dir)
        return output_dir, try_loading_data

    # Default: use config ID as folder name
    return get_experiment_dir() / exp_cfg.get_id(), try_loading_data


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    args = parse_args()

    # Validate mutually exclusive options
    if args.cache and args.output_dir_name:
        print("Error: --cache and --output-dir-name are mutually exclusive")
        return 1

    if args.update_config and not args.cache:
        print("Error: --update-config requires --cache")
        return 1

    # When using --cache with a folder name, try to load working_config.json first
    working_cfg = None
    cached_cfg = None
    if args.cache and isinstance(args.cache, str):
        cache_dir = get_experiment_dir() / args.cache

        # Try loading working_config.json (for reusing step configs)
        if not args.update_config:
            working_cfg = ExperimentConfig.load_working(cache_dir)
            if working_cfg:
                print(f"[cache] Loaded working config from {cache_dir}/working_config.json")
                print(f"[cache] Model: {working_cfg.model}")
                print(f"[cache] Dataset: {working_cfg.dataset_name}")

        # Fall back to original config for identity fields
        if not working_cfg:
            cached_cfg = ExperimentConfig.load(cache_dir)
            if cached_cfg:
                print(f"[cache] Loaded original config from {cache_dir}")
                print(f"[cache] Model: {cached_cfg.model}")
                print(f"[cache] Dataset: {cached_cfg.dataset_name}")

    # If working_config.json was loaded (and not --update-config), use it directly
    if working_cfg:
        # Apply camera-ready flag even when using cached config
        if args.camera_ready:
            working_cfg.viz_cfg["save_svg"] = True
            working_cfg.viz_cfg["save_pdf"] = True
        exp_cfg = working_cfg
    else:
        # Build experiment config from CLI args
        # When using cache: start with cached identity fields only (model, dataset, counts)
        # All step configs (coarse, attrib, etc.) and other configs (viz, pair_req) use defaults
        if cached_cfg:
            config = {
                "model": cached_cfg.model,
                "dataset_config": cached_cfg.dataset_config,
                "n_pairs": cached_cfg.n_pairs,
                "max_samples": cached_cfg.max_samples,
            }
            # CLI args override cached values
            if args.model:
                config["model"] = args.model
            if args.n_pairs:
                config["n_pairs"] = args.n_pairs
        else:
            config = build_base_config(args)

        if args.disable:
            apply_disable_flag(config)

        if args.only_viz_agg:
            apply_only_viz_agg_flag(config)

        if args.camera_ready:
            apply_camera_ready_flag(config)

        apply_json_overrides(config, args)

        # Create experiment config object
        exp_cfg = ExperimentConfig.from_dict(config)

    # Resolve output directory
    output_dir, try_loading_data = resolve_output_dir(args, exp_cfg)

    # Set up logging
    output_dir.mkdir(parents=True, exist_ok=True)
    set_log_file(output_dir / "log.txt")

    # Save config (creates original_config.json + working_config.json on first run,
    # updates working_config.json if --update-config is used)
    exp_cfg.save(output_dir, update_working=args.update_config)

    log_header(f"EXPERIMENT: {exp_cfg.get_id()}", gap=1)
    log_kv("Output", str(output_dir))
    log()

    # Run experiment
    try:
        run_experiment_per_pair(
            exp_cfg,
            try_loading_data=try_loading_data,
            output_dir=output_dir,
            backend=args.backend,
        )
        P.report()
        P.save(output_dir / "profile.json")
    finally:
        close_log_file()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
