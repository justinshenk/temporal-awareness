#!/usr/bin/env python3
"""Geometric visualization of temporal awareness in model activations.

Analyzes how time horizon information is encoded in transformer activations using
linear probes, PCA correlation analysis, and dimensionality reduction.

Key findings:
- Time horizon is linearly decodable at response positions (R² ~ 0.98)
- Source positions show no time signal (R² ~ -0.01)
- The signal emerges strongly in layers 19-24

Usage:
    uv run python scripts/intertemporal/analyze_geometry.py
    uv run python scripts/intertemporal/analyze_geometry.py --cache
    uv run python scripts/intertemporal/analyze_geometry.py --only-viz
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.intertemporal.data.default_configs import DEFAULT_MODEL
from src.intertemporal.geometry import GeometryConfig, TargetSpec, run_geometry_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Analysis Configuration
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

# Prompt positions (where time horizon info is encoded)
PROMPT_POSITIONS = [
    "time_horizon",
    "post_time_horizon",
]

# Response positions (where model output is generated)
RESPONSE_POSITIONS = [
    "chat_suffix",
    "response_choice_prefix",
    "response_choice",
    "response_reasoning_prefix",
    "response_reasoning",
]


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


# All positions for extraction
ALL_POSITIONS = PROMPT_POSITIONS + RESPONSE_POSITIONS

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
        description="Geometric visualization of temporal awareness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--cache",
        action="store_true",
        help="Use cached data if available",
    )
    parser.add_argument(
        "--only-viz",
        action="store_true",
        help="Only run visualization (skip extraction, requires cache)",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip sample generation and extraction (requires cache)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip all visualization (extraction and analysis only)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: skip per-target plots (09_targets folder)",
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


# =============================================================================
# Summary Output
# =============================================================================


def classify_position(key: str) -> str:
    """Classify a target key as PROMPT or RESPONSE position."""
    response_markers = ["response", "choice", "reasoning"]
    prompt_markers = ["time_horizon", "short_term", "long_term"]

    if any(marker in key for marker in response_markers):
        return "RESPONSE"
    if any(marker in key for marker in prompt_markers):
        return "PROMPT"
    return "OTHER"


def print_summary(results: dict, output_dir: str) -> None:
    """Print summary of pipeline results."""
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)

    summary = results.get("summary", {})
    linear_probe = summary.get("linear_probe", {})

    if linear_probe:
        logger.info("\nTop Linear Probe R² (ability to decode time horizon):")
        sorted_items = sorted(
            linear_probe.items(),
            key=lambda x: x[1]["r2"],
            reverse=True,
        )[:15]

        for key, data in sorted_items:
            pos_type = classify_position(key)
            logger.info(
                f"  [{pos_type}] {key}: R²={data['r2']:.3f} (corr={data['corr']:.3f})"
            )

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Key outputs:")
    output_dirs = [
        "01_dashboard",
        "02_linear_probe",
        "03_decision_boundary",
        "04_trajectories",
        "05_direction_alignment",
        "06_scree",
        "07_component_decomp",
        "08_component_decomp_3d",
        "09_targets",
    ]
    for d in output_dirs:
        logger.info(f"  - {output_dir}/plots/{d}/")
    logger.info(f"  - {output_dir}/summary.json")


# =============================================================================
# Main
# =============================================================================


def backup_existing_output(output_dir: Path) -> None:
    """Move existing output folder to {output_dir}_{timestamp} if it exists."""
    if not output_dir.exists():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = output_dir.parent / f"{output_dir.name}_{timestamp}"
    logger.info(f"Moving existing output to: {backup_dir}")
    shutil.move(str(output_dir), str(backup_dir))


def main() -> None:
    """Run the geometric visualization pipeline."""
    args = parse_args()

    output_dir = Path(args.output_dir)

    # Backup existing output folder (skip if using cache)
    if not args.cache and not args.only_viz and not args.skip_extraction:
        backup_existing_output(output_dir)

    # Build config from defaults + CLI args
    config = GeometryConfig(
        targets=DEFAULT_CONFIG["targets"],
        output_dir=Path(args.output_dir),
        model=args.model,
        seed=args.seed,
        max_samples=args.max_samples,
        n_pca_components=DEFAULT_CONFIG["n_pca_components"],
    )

    # Log config summary (not all 396 targets individually)
    logger.info("Configuration:")
    logger.info(f"  Model: {config.model}")
    logger.info(f"  Output: {config.output_dir}")
    logger.info(f"  Layers: {LAYERS}")
    logger.info(f"  Components: {COMPONENTS}")
    logger.info(
        f"  Positions: {len(ALL_POSITIONS)} ({len(PROMPT_POSITIONS)} prompt + {len(RESPONSE_POSITIONS)} response)"
    )
    logger.info(f"  Total targets: {len(config.targets)}")

    # Handle --only-viz (implies --cache and --skip-extraction)
    use_cache = args.cache or args.only_viz
    skip_extraction = args.skip_extraction or args.only_viz

    # Run pipeline
    results = run_geometry_pipeline(
        config,
        use_cache=use_cache,
        skip_extraction=skip_extraction,
        skip_viz=args.no_viz,
        skip_per_target_plots=args.quick,
    )

    print_summary(results, str(config.output_dir))


if __name__ == "__main__":
    main()
