#!/usr/bin/env python3
"""Generate geometry samples and extract raw activations.

This is Script 1 of the geometry pipeline. It handles:
- Generating samples from the dataset configuration
- Extracting activations for all layer/component/position targets
- Saving per-sample activation files to data/samples/

Output structure:
    out/geometry/
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
    # Generate new samples
    uv run python scripts/intertemporal/generate_geometry_samples.py

    # Use existing samples if available
    uv run python scripts/intertemporal/generate_geometry_samples.py --cache

    # Custom output directory
    uv run python scripts/intertemporal/generate_geometry_samples.py --output-dir out/geo_test
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.intertemporal.common.semantic_positions import (
    PROMPT_POSITIONS,
    RESPONSE_POSITIONS,
)
from src.intertemporal.data.default_configs import DEFAULT_MODEL
from src.intertemporal.geometry import GeometryConfig, TargetSpec
from src.intertemporal.geometry.geometry_pipeline import generate_geo_samples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Layers selected based on circuit analysis
LAYERS = [
    0,  # baseline embedding
    1,  # early embedding
    3,  # early embedding
    12,  # mid-network (before circuit)
    18,  # just before circuit onset
    19,  # circuit onset
    21,  # integration point
    24,  # top attention layer
    28,  # secondary MLP processor
    31,  # most reliable MLP
    34,  # final processing
    35,  # penultimate layer
]

# Activation components to extract
COMPONENTS = ["resid_pre", "attn_out", "mlp_out", "resid_post"]

# All positions for extraction
ALL_POSITIONS = PROMPT_POSITIONS + RESPONSE_POSITIONS


def build_targets(
    layers: list[int],
    components: list[str],
    positions: list[str],
) -> list[TargetSpec]:
    """Build target specifications for all layer/component/position combinations."""
    return [
        TargetSpec(layer=layer, component=component, position=position)
        for layer in layers
        for component in components
        for position in positions
    ]


# Default configuration
DEFAULT_CONFIG = {
    "targets": build_targets(LAYERS, COMPONENTS, ALL_POSITIONS),
    "output_dir": "out/geometry",
    "model": DEFAULT_MODEL,
    "seed": 42,
    "n_pca_components": 10,
}


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate geometry samples and extract activations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--cache",
        action="store_true",
        help="Use cached data if available",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_CONFIG["output_dir"],
        help=f"Output directory (default: {DEFAULT_CONFIG['output_dir']})",
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

    return parser.parse_args()


def backup_existing_output(output_dir: Path) -> None:
    """Move existing output folder to {output_dir}_{timestamp} if it exists."""
    if not output_dir.exists():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = output_dir.parent / f"{output_dir.name}_{timestamp}"
    logger.info(f"Moving existing output to: {backup_dir}")
    shutil.move(str(output_dir), str(backup_dir))


def create_summary_json(
    output_dir: Path,
    n_samples: int,
    layers: list[int],
    components: list[str],
    all_positions: list[str],
    sparse_positions: list[str],
) -> None:
    """Create summary.json with metadata about generated data.

    This file documents what layers, components, and positions were extracted,
    as well as paths to all data files.

    Args:
        output_dir: Output directory
        n_samples: Number of samples
        layers: List of layers extracted
        components: List of components extracted
        all_positions: All positions that were requested for extraction
        sparse_positions: Positions that only exist in some samples (not all)
    """
    summary = {
        "n_samples": n_samples,
        "layers": layers,
        "components": components,
        "positions": all_positions,
        "n_layers": len(layers),
        "n_components": len(components),
        "n_positions": len(all_positions),
        "n_targets": len(layers) * len(components) * len(all_positions),
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
            "sparse_position_explanation": "These positions exist only in some samples, not all. They have data but not universally across samples.",
        },
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Created summary.json: {summary_path}")


def main() -> int:
    """Run sample generation and activation extraction."""
    args = parse_args()

    output_dir = Path(args.output_dir)

    # Backup existing output folder (skip if using cache)
    if not args.cache:
        backup_existing_output(output_dir)

    # Build config
    config = GeometryConfig(
        targets=DEFAULT_CONFIG["targets"],
        output_dir=output_dir,
        model=args.model,
        seed=args.seed,
        max_samples=args.max_samples,
        n_pca_components=DEFAULT_CONFIG["n_pca_components"],
    )

    # Log config summary
    logger.info("=" * 60)
    logger.info("GENERATE GEOMETRY SAMPLES")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Layers: {LAYERS}")
    logger.info(f"Components: {COMPONENTS}")
    logger.info(
        f"Positions: {len(ALL_POSITIONS)} ({len(PROMPT_POSITIONS)} prompt + {len(RESPONSE_POSITIONS)} response)"
    )
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
        # Component is 2 parts for resid_pre/resid_post, 2 parts for attn_out/mlp_out
        for comp in COMPONENTS:
            comp_parts = comp.split("_")
            comp_len = len(comp_parts)
            if "_".join(parts[1:1+comp_len]) == comp:
                position = "_".join(parts[1+comp_len:])
                positions_with_data.add(position)
                break

    # Sparse positions: positions that were requested but only exist in some samples
    sparse_positions = [p for p in ALL_POSITIONS if p not in positions_with_data]

    # Create summary.json with ALL requested positions
    create_summary_json(
        output_dir=output_dir,
        n_samples=len(data.samples),
        layers=LAYERS,
        components=COMPONENTS,
        all_positions=ALL_POSITIONS,
        sparse_positions=sparse_positions,
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
