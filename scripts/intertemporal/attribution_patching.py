#!/usr/bin/env python
"""
Attribution patching for intertemporal preference analysis.

Core computation uses run_attribution_patching from src/experiments.
This script adds standalone CLI, positions.json output, and detailed summary.

Usage:
    # Quick test with defaults
    python scripts/intertemporal/attribution_patching.py

    # Full run
    python scripts/intertemporal/attribution_patching.py --max-pairs 5 --ig-steps 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import ModelRunner
from src.data import (
    load_pref_data_with_prompts,
    build_prompt_pairs,
    find_preference_data,
    get_preference_data_id,
)
from src.common.io import ensure_dir, save_json, get_timestamp
from src.analysis import find_top_attributions
from src.common.positions_schema import PositionsFile, PositionSpec
from src.profiler import P
from src.viz import plot_layer_position_heatmap
from src.experiments import run_attribution_patching

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Default config for standalone runs (small but meaningful)
DEFAULT_CONFIG = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "max_pairs": 1,
    "ig_steps": 3,
    "positions_threshold": 0.1,  # Attribution score threshold for positions.json
}

# Method metadata for display and grouping
METHOD_INFO = {
    "resid": {"name": "Residual Stream", "group": "standard"},
    "attn": {"name": "Attention Output", "group": "standard"},
    "mlp": {"name": "MLP Output", "group": "standard"},
    "eap_attn": {"name": "EAP Attention", "group": "eap"},
    "eap_mlp": {"name": "EAP MLP", "group": "eap"},
    "eap_ig_attn": {"name": "EAP-IG Attention", "group": "eap_ig"},
    "eap_ig_mlp": {"name": "EAP-IG MLP", "group": "eap_ig"},
}


def save_positions_json(
    aggregated: dict[str, np.ndarray],
    layers: list[int],
    token_labels: list[str],
    section_markers: dict[str, int],
    model: str,
    dataset_id: str,
    threshold: float,
    output_dir: Path,
) -> None:
    """Save standardized positions.json from attribution results."""
    if "resid" not in aggregated:
        return

    scores = aggregated["resid"]
    positions = []
    for layer_idx, layer in enumerate(layers):
        for pos in range(scores.shape[1]):
            score = float(scores[layer_idx, pos])
            if abs(score) > threshold:
                section = None
                for sec_name, sec_pos in section_markers.items():
                    if pos >= sec_pos:
                        section = sec_name.replace("before_", "")
                positions.append(PositionSpec(
                    position=pos,
                    token=token_labels[pos] if pos < len(token_labels) else f"pos{pos}",
                    score=score,
                    layer=layer,
                    section=section,
                ))

    PositionsFile(
        model=model,
        method="attribution_patching",
        positions=sorted(positions, key=lambda p: abs(p.score), reverse=True),
        dataset_id=dataset_id,
        threshold=threshold,
        component="resid_post",
    ).save(output_dir / "positions.json")


def print_summary(
    aggregated: dict[str, np.ndarray],
    layers: list[int],
    token_labels: list[str],
) -> dict:
    """Print summary of results and return metadata."""
    print("\n=== Attribution Summary ===")

    summary = {}
    for group_name in ["standard", "eap", "eap_ig"]:
        group_methods = [k for k, v in METHOD_INFO.items() if v["group"] == group_name]
        if not group_methods:
            continue

        print(f"\n{group_name.upper().replace('_', '-')} Methods:")
        for key in group_methods:
            if key not in aggregated:
                continue
            scores = aggregated[key]
            info = METHOD_INFO[key]

            top = find_top_attributions(scores, layers, n_top=1)
            if top:
                layer, pos, score = top[0]
                pos_label = token_labels[pos] if pos < len(token_labels) else f"pos{pos}"
                print(f"  {info['name']:20s}: max={scores.max():.4f}, min={scores.min():.4f}, top=L{layer}@{pos_label}({score:.4f})")

                summary[key] = {
                    "max": float(scores.max()),
                    "min": float(scores.min()),
                    "top_layer": layer,
                    "top_position": pos,
                    "top_score": float(score),
                }

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attribution patching analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--preference-data", type=str, help="Preference data ID")
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_CONFIG["max_pairs"])
    parser.add_argument("--ig-steps", type=int, default=DEFAULT_CONFIG["ig_steps"])
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "out" / "attribution_patching")
    return parser.parse_args()


def main():
    args = parse_args()

    with P("total"):
        # Load preference data
        pref_dir = PROJECT_ROOT / "out" / "preference_data"
        data_dir = PROJECT_ROOT / "out" / "datasets"

        if args.preference_data:
            pref_id = args.preference_data
        else:
            recent = find_preference_data(pref_dir)
            if not recent:
                print("Error: No preference data found in", pref_dir)
                return 1
            pref_id = get_preference_data_id(recent)

        with P("load_data"):
            pref_data = load_pref_data_with_prompts(pref_id, pref_dir, data_dir)
            pairs = build_prompt_pairs(pref_data, max_pairs=args.max_pairs, include_response=True)

        print(f"Preference data: {pref_id}")
        print(f"Model: {pref_data.model}")
        print(f"Samples: {len(pref_data.preferences)}, Pairs: {len(pairs)}")

        if not pairs:
            print("Error: No valid pairs found")
            return 1

        # Load model
        with P("load_model"):
            runner = ModelRunner(pref_data.model)

        print(f"Layers: {runner.n_layers}, IG steps: {args.ig_steps}")

        # Run attribution using src/ function (no duplication)
        print(f"\n=== Running Attribution Patching ({args.max_pairs} pairs) ===")
        with P("attribution"):
            aggregated, token_labels, section_markers = run_attribution_patching(
                runner, pref_data,
                max_pairs=args.max_pairs,
                ig_steps=args.ig_steps,
            )

        # Create output directory
        run_dir = args.output / get_timestamp()
        ensure_dir(run_dir)

        # Save heatmaps
        layers = list(range(runner.n_layers))
        model_name = pref_data.model.split("/")[-1]
        n_pairs = min(len(pairs), args.max_pairs)

        print("\n=== Saving Heatmaps ===")
        for key, scores in aggregated.items():
            info = METHOD_INFO.get(key, {"name": key})
            plot_layer_position_heatmap(
                scores, layers, token_labels,
                run_dir / f"heatmap_{key}.png",
                title=info["name"],
                subtitle=f"{model_name} | n={n_pairs} | range=[{scores.min():.3f}, {scores.max():.3f}]",
                cbar_label="Attribution (normalized)",
                cmap="RdBu_r",
                section_markers=section_markers,
            )
            np.save(run_dir / f"{key}.npy", scores)

        # Print summary
        summary = print_summary(aggregated, layers, token_labels)

        # Save positions.json
        save_positions_json(
            aggregated, layers, token_labels, section_markers,
            pref_data.model, pref_data.dataset_id,
            DEFAULT_CONFIG["positions_threshold"], run_dir,
        )

        # Save metadata
        save_json({
            "timestamp": get_timestamp(),
            "model": pref_data.model,
            "dataset_id": pref_data.dataset_id,
            "n_pairs": n_pairs,
            "ig_steps": args.ig_steps,
            "n_layers": runner.n_layers,
            "n_positions": len(token_labels),
            "section_markers": section_markers,
            "methods": list(aggregated.keys()),
            "summary": summary,
        }, run_dir / "metadata.json")

        print(f"\nResults saved to: {run_dir}")

    P.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
