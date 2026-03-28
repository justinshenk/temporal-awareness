#!/usr/bin/env python3
"""Geometric visualization of temporal awareness in model activations.

This is a convenience wrapper that runs all 3 geometry pipeline scripts:
1. generate_geometry_samples.py - Generate samples and extract activations
2. compute_geometry_analysis.py - Compute analysis and embeddings
3. visualize_geometry_analysis.py - Generate static plots

For more control, run the individual scripts directly.

Key findings:
- Time horizon is linearly decodable at response positions (R² ~ 0.98)
- Source positions show no time signal (R² ~ -0.01)
- The signal emerges strongly in layers 19-24

Usage:
    # Run full pipeline
    uv run python scripts/intertemporal/analyze_geometry.py

    # Use cached data (skip sample generation)
    uv run python scripts/intertemporal/analyze_geometry.py --cache

    # Skip visualization
    uv run python scripts/intertemporal/analyze_geometry.py --no-viz

    # Quick mode (skip per-target plots)
    uv run python scripts/intertemporal/analyze_geometry.py --quick
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
        help="Use cached data if available (skip sample generation)",
    )
    parser.add_argument(
        "--only-viz",
        action="store_true",
        help="Only run visualization (skip extraction and analysis, requires cache)",
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
        default="out/geometry",
        help="Output directory (default: out/geometry)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (default: from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (default: all)",
    )
    parser.add_argument(
        "--full-embeddings",
        action="store_true",
        help="Compute all embeddings (PCA + UMAP + t-SNE). Default: PCA only.",
    )

    return parser.parse_args()


def run_script(script_name: str, args: list[str]) -> int:
    """Run a script with the given arguments."""
    script_path = PROJECT_ROOT / "scripts" / "intertemporal" / script_name
    cmd = ["uv", "run", "python", str(script_path)] + args
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main() -> int:
    """Run the geometry visualization pipeline."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("GEOMETRY VISUALIZATION PIPELINE")
    logger.info("=" * 60)

    # Handle --only-viz (implies --cache and --skip-extraction)
    skip_extraction = args.skip_extraction or args.only_viz
    use_cache = args.cache or args.only_viz

    # Script 1: Generate samples (unless skipping)
    if not skip_extraction:
        logger.info("\n" + "=" * 60)
        logger.info("Step 1/3: Generate Samples")
        logger.info("=" * 60)

        script1_args = ["--output-dir", args.output_dir]
        if use_cache:
            script1_args.append("--cache")
        if args.model:
            script1_args.extend(["--model", args.model])
        if args.max_samples:
            script1_args.extend(["--max-samples", str(args.max_samples)])
        script1_args.extend(["--seed", str(args.seed)])

        ret = run_script("generate_geometry_samples.py", script1_args)
        if ret != 0:
            logger.error("Sample generation failed")
            return ret
    else:
        logger.info("\n" + "=" * 60)
        logger.info("Step 1/3: Generate Samples (SKIPPED)")
        logger.info("=" * 60)

    # Script 2: Compute analysis (unless only-viz)
    if not args.only_viz:
        logger.info("\n" + "=" * 60)
        logger.info("Step 2/3: Compute Analysis")
        logger.info("=" * 60)

        script2_args = ["--data-dir", args.output_dir]
        if args.full_embeddings:
            script2_args.append("--full")

        ret = run_script("compute_geometry_analysis.py", script2_args)
        if ret != 0:
            logger.error("Analysis computation failed")
            return ret
    else:
        logger.info("\n" + "=" * 60)
        logger.info("Step 2/3: Compute Analysis (SKIPPED)")
        logger.info("=" * 60)

    # Script 3: Visualize (unless no-viz)
    if not args.no_viz:
        logger.info("\n" + "=" * 60)
        logger.info("Step 3/3: Generate Visualizations")
        logger.info("=" * 60)

        script3_args = ["--data-dir", args.output_dir]
        if args.quick:
            script3_args.append("--quick")

        ret = run_script("visualize_geometry_analysis.py", script3_args)
        if ret != 0:
            logger.error("Visualization generation failed")
            return ret
    else:
        logger.info("\n" + "=" * 60)
        logger.info("Step 3/3: Generate Visualizations (SKIPPED)")
        logger.info("=" * 60)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("Key outputs:")
    logger.info(f"  - {args.output_dir}/data/         # Raw activations")
    logger.info(f"  - {args.output_dir}/analysis/     # Analysis results + embeddings")
    logger.info(f"  - {args.output_dir}/viz/          # Static plots")
    logger.info(f"  - {args.output_dir}/summary.json  # Summary statistics")

    return 0


if __name__ == "__main__":
    sys.exit(main())
