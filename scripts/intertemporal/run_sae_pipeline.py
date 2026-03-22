#!/usr/bin/env python
"""
Temporal SAE Training Pipeline

Trains Sparse Autoencoders on position-specific activations from LLM responses
to temporal preference questions. Extracts activations at key token positions
(source, dest, secondary_source) across multiple components (resid_pre, resid_post,
mlp_out, attn_out).

Usage:
    python run_sae_pipeline.py                       # Show status + resume pipeline
    python run_sae_pipeline.py --resume              # Resume from checkpoint
    python run_sae_pipeline.py --test-iter           # Quick smoke test
    python run_sae_pipeline.py --new                 # Start new pipeline
    python run_sae_pipeline.py --config other.json   # Use alternate config

Layer/Component Selection:
    python run_sae_pipeline.py --layers 21 31 24     # Specific layers
    python run_sae_pipeline.py --components resid_post mlp_out  # Specific components
    python run_sae_pipeline.py --positions dest source  # Specific positions
    python run_sae_pipeline.py --priority high       # Use high-priority targets only

Priority Levels (from circuit analysis):
    high:   L21 resid_post, L31 mlp_out, L24 resid_post
    medium: L19 resid_pre, L34 resid_post
    lower:  L19 mlp_out (counterproductive), L25 attn_out (counterproductive)

Position Names:
    dest:             Where the model makes its choice (P145-type)
    source:           Primary source - time horizon info (P86-type)
    secondary_source: Adjacent to primary source (P87-type)
"""

import argparse
import faulthandler
import json
import sys
import traceback
from pathlib import Path

# Enable faulthandler to get tracebacks on segfaults/crashes
faulthandler.enable()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.intertemporal.sae.sae_positions import (
    COMPONENTS,
    POSITION_NAMES,
    SAEConfig,
)

# Default config location
DEFAULT_CONFIG = PROJECT_ROOT / "src" / "intertemporal" / "sae" / "experiment_cfg.json"

# Priority layer configurations
PRIORITY_LAYERS = {
    "high": [21, 31, 24],
    "medium": [19, 34],
    "lower": [25],
    "all": [21, 31, 24, 19, 34, 25],
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Temporal SAE Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
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

    # Layer/Component/Position selection
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Specific layers to analyze (e.g., --layers 21 31 24)",
    )
    parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        choices=COMPONENTS,
        help=f"Components to extract (choices: {COMPONENTS})",
    )
    parser.add_argument(
        "--positions",
        type=str,
        nargs="+",
        choices=POSITION_NAMES,
        help=f"Positions to extract (choices: {POSITION_NAMES})",
    )
    parser.add_argument(
        "--priority",
        type=str,
        choices=["high", "medium", "lower", "all"],
        help="Use layers from a priority level instead of --layers",
    )

    # Show recommended targets
    parser.add_argument(
        "--show-targets",
        action="store_true",
        help="Show recommended SAE targets and exit",
    )

    return parser.parse_args()


def show_recommended_targets():
    """Show recommended SAE targets based on circuit analysis."""
    config = SAEConfig()
    targets = config.get_recommended_targets()

    print("\nRecommended SAE Targets (from circuit analysis):")
    print("=" * 60)
    print(f"{'#':<3} {'Layer':<6} {'Component':<12} {'Position':<18} {'Priority'}")
    print("-" * 60)

    high_priority = [
        (21, "resid_post", "dest"),
        (31, "mlp_out", "dest"),
        (24, "resid_post", "dest"),
    ]
    medium_priority = [
        (19, "resid_pre", "source"),
        (21, "resid_post", "source"),
        (24, "attn_out", "dest"),
        (34, "resid_post", "dest"),
    ]

    for i, t in enumerate(targets, 1):
        key = (t.layer, t.component, t.position_name)
        if key in high_priority:
            priority = "HIGH"
        elif key in medium_priority:
            priority = "MEDIUM"
        else:
            priority = "lower"
        print(f"{i:<3} L{t.layer:<5} {t.component:<12} {t.position_name:<18} {priority}")

    print("\nPosition meanings:")
    print("  dest:             Choice position (where model decides)")
    print("  source:           Time horizon position (input feature)")
    print("  secondary_source: Adjacent to source (for comparison)")

    print("\nComponent meanings:")
    print("  resid_post: Residual stream after layer (integration point)")
    print("  mlp_out:    MLP output (transformation)")
    print("  attn_out:   Attention output (routing)")
    print("  resid_pre:  Residual stream before layer (baseline)")


def main():
    args = get_args()

    if args.show_targets:
        show_recommended_targets()
        return 0

    # Import pipeline modules only when needed (they have heavy dependencies)
    from src.intertemporal.sae.pipeline_state import (
        PipelineConfig,
        PipelineState,
        find_state,
        find_latest_state,
        show_status,
    )
    from src.intertemporal.sae.sae_pipeline import run_pipeline, run_test_iteration

    # Load config from JSON
    with open(args.config) as f:
        cfg = json.load(f)

    # Override config with CLI arguments
    if args.priority:
        cfg["layers"] = PRIORITY_LAYERS[args.priority]
    elif args.layers:
        cfg["layers"] = args.layers

    if args.components:
        cfg["components"] = args.components

    if args.positions:
        cfg["position_names"] = args.positions

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

    # Show configuration summary
    print("\nConfiguration:")
    print(f"  Layers: {current_state.config.layers}")
    print(f"  Components: {current_state.config.components}")
    print(f"  Positions: {current_state.config.position_names}")
    n_targets = (
        len(current_state.config.layers)
        * len(current_state.config.components)
        * len(current_state.config.position_names)
    )
    print(f"  Total target combinations: {n_targets}")

    show_status(current_state)

    if args.test_iter:
        run_test_iteration(current_state)
    else:
        run_pipeline(current_state)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 60)
        print("PIPELINE CRASHED WITH EXCEPTION:")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)
