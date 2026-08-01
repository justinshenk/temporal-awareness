#!/usr/bin/env python3
"""Generate geometry samples and extract raw activations.

This is Script 1 of the geometry pipeline. It handles:
- Generating samples from the dataset configuration
- Extracting activations for all layer/component/position targets
- Saving per-sample activation files to data/samples/

Output structure:
    out/geo/{dataset_name}_{timestamp}/
        data/
            metadata.json
            prompt_dataset.json
            samples/
                sample_0/
                    position_mapping.json
                    prompt_sample.json
                    preference_sample.json
                    choice.json
                    L{layer}_{component}_{abs_pos}.npy
                sample_1/
                    ...

Usage:
    # Generate with default dataset (GEOMETRY_CFG from FULL_EXPERIMENT_CONFIG)
    uv run python scripts/intertemporal/generate_geometry_samples.py
    # Output: out/geo/geometry_20240101_120000/

    # Generate with a config file from configs/prompt_datasets/
    uv run python scripts/intertemporal/generate_geometry_samples.py --config nano
    # Output: out/geo/nano_20240101_120000/

    # Generate with a custom JSON config file path
    uv run python scripts/intertemporal/generate_geometry_samples.py --config path/to/config.json

    # Use cached model data if available
    uv run python scripts/intertemporal/generate_geometry_samples.py --cache

    # Extract only the change-of-turn window, two components, half-precision
    uv run python scripts/intertemporal/generate_geometry_samples.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --turn-only --components resid_post,attn_out --dtype float16

    # Resolve and print the run scope without loading the model
    uv run python scripts/intertemporal/generate_geometry_samples.py --turn-only --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.file_io import parse_file_path
from src.intertemporal.common.model_layers import fetch_n_layers
from src.intertemporal.common.project_paths import get_prompt_dataset_configs_dir
from src.intertemporal.data.default_configs import DEFAULT_MODEL, FULL_EXPERIMENT_CONFIG
from src.intertemporal.geometry import GeometryConfig, RunScope, TargetSpec
from src.intertemporal.geometry.geometry_config import (
    DEFAULT_STORAGE_DTYPE,
    STORAGE_DTYPES,
)
from src.intertemporal.geometry.geometry_pipeline import generate_geo_samples
from src.intertemporal.geometry.geometry_scope import (
    parse_int_list,
    parse_str_list,
    resolve_scope,
)
from src.intertemporal.geometry.geometry_utils import COMPONENTS
from src.intertemporal.prompt import PromptDatasetConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


def build_targets(
    layers: list[int],
    components: list[str],
    positions: list[str],
) -> list[TargetSpec]:
    """Build target specifications for all layer/component/position combinations."""
    return RunScope(
        layers=layers, components=components, positions=positions
    ).targets()


# Default configuration. Targets are not listed here: they depend on the model's
# depth and on the run scope, so they are resolved per run in build_scope().
DEFAULT_CONFIG = {
    "base_dir": "out/geo",
    "model": DEFAULT_MODEL,
    "seed": 42,
}


# =============================================================================
# CLI
# =============================================================================


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments (argv defaults to sys.argv)."""
    parser = argparse.ArgumentParser(
        description="Generate geometry samples and extract activations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Dataset config file path (or config name from configs/prompt_datasets/). "
        "If not provided, uses default.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Use cached data if available",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=DEFAULT_CONFIG["base_dir"],
        help=f"Base output directory (default: {DEFAULT_CONFIG['base_dir']})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CONFIG["model"],
        help=f"Model identifier (default: {DEFAULT_CONFIG['model']})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
        help=f"Random seed (default: {DEFAULT_CONFIG['seed']})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (default: all)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from existing output directory (e.g., out/geo/cityhousing_geometry_20260401_182424)",
    )

    scope_group = parser.add_argument_group("run scope")
    layer_choice = scope_group.add_mutually_exclusive_group()
    layer_choice.add_argument(
        "--layers",
        nargs="+",
        metavar="L",
        default=None,
        help="Explicit layer indices, comma or space separated (e.g. 0,12,31). "
        "Default: the standard layers projected onto the model's depth.",
    )
    layer_choice.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Use N layers evenly spaced across the model's depth "
        "(alternative to --layers)",
    )
    scope_group.add_argument(
        "--components",
        nargs="+",
        metavar="C",
        default=None,
        help="Components to extract, comma or space separated "
        f"(default: {','.join(COMPONENTS)})",
    )
    position_choice = scope_group.add_mutually_exclusive_group()
    position_choice.add_argument(
        "--positions",
        nargs="+",
        metavar="P",
        default=None,
        help="Explicit semantic positions, comma or space separated "
        "(default: all positions)",
    )
    position_choice.add_argument(
        "--turn-only",
        action="store_true",
        help="Restrict positions to the change-of-turn window "
        "(chat_suffix, chat_suffix_tail)",
    )
    scope_group.add_argument(
        "--dtype",
        type=str,
        choices=sorted(STORAGE_DTYPES),
        default=DEFAULT_STORAGE_DTYPE,
        help=f"Storage dtype for saved activations (default: {DEFAULT_STORAGE_DTYPE})",
    )
    scope_group.add_argument(
        "--n-model-layers",
        type=int,
        default=None,
        help="Model depth to project layers onto. Default: read num_hidden_layers "
        "from the model's hub config.json (no weights are downloaded).",
    )
    scope_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the run scope, then exit without loading the model",
    )

    return parser.parse_args(argv)


def build_scope(args: argparse.Namespace) -> RunScope:
    """Resolve the run scope from CLI arguments.

    Layer indices are checked against the model's real depth here, before any
    weights are loaded, so an impossible layer fails immediately instead of
    being dropped during extraction.

    List options accept either "0,12,31" or "0 12 31", because callers that
    build the argument list in a shell get one word per value.
    """
    n_model_layers = (
        args.n_model_layers
        if args.n_model_layers is not None
        else fetch_n_layers(args.model)
    )

    return resolve_scope(
        n_model_layers=n_model_layers,
        layers=parse_int_list(",".join(args.layers)) if args.layers else None,
        n_layers=args.n_layers,
        components=(
            parse_str_list(",".join(args.components)) if args.components else None
        ),
        positions=parse_str_list(",".join(args.positions)) if args.positions else None,
        turn_only=args.turn_only,
        dtype=args.dtype,
    )


def parse_config(args: argparse.Namespace) -> PromptDatasetConfig:
    """Parse dataset config from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        PromptDatasetConfig for sample generation
    """
    if not args.config:
        # Use built-in default config (GEOMETRY_CFG)
        config = PromptDatasetConfig.from_dict(FULL_EXPERIMENT_CONFIG["dataset_config"])
        print("Using FULL_EXPERIMENT_CONFIG (GEOMETRY_CFG):")
        print(f"  name: {config.name}")
        return config

    # Get full json file path
    filepath = parse_file_path(
        args.config,
        default_dir_path=str(get_prompt_dataset_configs_dir()),
        default_ext=".json",
    )
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset config not found: {filepath}")

    # Load dataset config
    config = PromptDatasetConfig.from_json(filepath)
    print(f"Loaded config: {config.name} from {filepath}")
    return config


def create_summary_json(
    output_dir: Path,
    n_samples: int,
    scope: RunScope,
    sparse_positions: list[str],
    dataset_name: str,
) -> None:
    """Create summary.json with metadata about generated data.

    The scope is written out verbatim so downstream analysis reads the target
    set this run actually extracted, rather than re-importing module constants.

    Args:
        output_dir: Output directory
        n_samples: Number of samples
        scope: Resolved layers, components, positions and storage dtype
        sparse_positions: Positions that only exist in some samples (not all)
        dataset_name: Name of the dataset config used
    """
    summary = {
        "n_samples": n_samples,
        "layers": scope.layers,
        "components": scope.components,
        "positions": scope.positions,
        "dtype": scope.dtype,
        "n_layers": len(scope.layers),
        "n_components": len(scope.components),
        "n_positions": len(scope.positions),
        "n_targets": scope.n_targets,
        "dataset_config": {
            "name": dataset_name,
        },
        "datasets": {
            "prompt_dataset": "data/prompt_dataset.json",
            "metadata": "data/metadata.json",
        },
        "data_paths": {
            "samples": "data/samples/",
            "activations": "data/samples/sample_{idx}/L{layer}_{component}_{abs_pos}.npy",
            "position_mapping": "data/samples/sample_{idx}/position_mapping.json",
            "choice": "data/samples/sample_{idx}/choice.json",
            "prompt_sample": "data/samples/sample_{idx}/prompt_sample.json",
            "preference_sample": "data/samples/sample_{idx}/preference_sample.json",
        },
        "analysis_paths": {
            "embeddings_pca": "analysis/embeddings/pca/L{layer}_{component}_{position}.npy",
            "trajectories_layer": "analysis/trajectories/layers_{component}_{position}.npz",
            "trajectories_position": "analysis/trajectories/positions_L{layer}_{component}.npz",
        },
        "notes": {
            "sparse_positions": sparse_positions,
            "sparse_position_explanation": "These positions exist only in some samples, not all.",
        },
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Created summary.json: {summary_path}")


def main() -> int:
    """Run sample generation and activation extraction."""
    args = get_args()

    # Resolve the run scope first: it validates layers against the model's real
    # depth, which is cheap to get wrong and expensive to discover mid-run.
    scope = build_scope(args)

    if args.dry_run:
        print(json.dumps({**scope.to_dict(), "n_targets": scope.n_targets}, indent=2))
        return 0

    # Parse dataset config (like generate_prompt_dataset.py)
    dataset_config = parse_config(args)
    dataset_name = dataset_config.name

    # Use resume directory or generate new timestamped output directory
    if args.resume:
        output_dir = Path(args.resume)
        if not output_dir.exists():
            logger.error(f"Resume directory does not exist: {output_dir}")
            return 1
        logger.info(f"RESUMING from existing directory: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.base_dir) / f"{dataset_name}_{timestamp}"

    # Build geometry config with dataset_cfg dict for collect_samples
    config = GeometryConfig(
        scope=scope,
        output_dir=output_dir,
        model=args.model,
        seed=args.seed,
        max_samples=args.max_samples,
        dataset_cfg=dataset_config.to_dict(),
    )

    # Log config summary
    logger.info("=" * 60)
    logger.info("GENERATE GEOMETRY SAMPLES")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Model: {config.model}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Layers: {scope.layers}")
    logger.info(f"Components: {scope.components}")
    logger.info(f"Positions: {len(scope.positions)} {scope.positions}")
    logger.info(f"Storage dtype: {scope.dtype}")
    logger.info(f"Total targets: {len(config.targets)}")

    # Run sample generation
    data = generate_geo_samples(config, use_cache=args.cache)

    # Determine which positions actually have data by parsing target keys
    # Target key format: L{layer}_{component}_{position}
    target_keys = data.get_target_keys()
    positions_with_data = set()
    for key in target_keys:
        # Parse: L0_resid_pre_response_choice -> response_choice
        parts = key.split("_")
        # Skip layer (L0) and component (resid_pre, attn_out, mlp_out, resid_post)
        for comp in scope.components:
            comp_parts = comp.split("_")
            comp_len = len(comp_parts)
            if "_".join(parts[1 : 1 + comp_len]) == comp:
                position = "_".join(parts[1 + comp_len :])
                positions_with_data.add(position)
                break

    # Sparse positions: positions that were requested but only exist in some samples
    sparse_positions = [p for p in scope.positions if p not in positions_with_data]

    # Create summary.json
    create_summary_json(
        output_dir=output_dir,
        n_samples=len(data.samples),
        scope=scope,
        sparse_positions=sparse_positions,
        dataset_name=dataset_name,
    )

    logger.info("=" * 60)
    logger.info("SAMPLE GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Samples: {len(data.samples)}")
    logger.info(f"Targets available: {len(target_keys)}")
    logger.info(f"Positions with data: {len(positions_with_data)}")
    logger.info(f"Output directory: {config.output_dir / 'data'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
