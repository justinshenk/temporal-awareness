#!/usr/bin/env python
"""
Attribution patching for intertemporal preference analysis.

Methods:
- Attribution Patching: (clean - corrupted) · grad
- EAP (Edge Attribution Patching): edges between components
- EAP-IG: EAP with Integrated Gradients for accuracy

Usage:
    python scripts/circuits/attribution_patching_intertemporal.py
    python scripts/circuits/attribution_patching_intertemporal.py --max-pairs 5
    python scripts/circuits/attribution_patching_intertemporal.py --ig-steps 20
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
from src.analysis import (
    build_position_mapping,
    create_metric,
    find_section_markers,
    get_token_labels,
    run_all_attribution_methods,
    aggregate_attribution_results,
    find_top_attributions,
)
from src.common.positions_schema import PositionsFile, PositionSpec
from src.profiler import P
from src.viz import plot_layer_position_heatmap

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "out" / "attribution_patching"

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


def run_attribution_on_pairs(
    runner: ModelRunner,
    pairs: list[tuple],
    max_pairs: int,
    ig_steps: int = 10,
) -> tuple[list[dict], list[str], dict[str, int]]:
    """Run attribution methods on multiple pairs, return per-pair results."""
    all_results = []
    token_labels = None
    section_markers = None

    for i, (clean_text, corr_text, clean_sample, corr_sample) in enumerate(pairs[:max_pairs]):
        print(f"  Pair {i + 1}/{min(len(pairs), max_pairs)}")

        clean_labels = [clean_sample.short_term_label, clean_sample.long_term_label]
        corr_labels = [corr_sample.short_term_label, corr_sample.long_term_label]

        with P("build_mapping"):
            pos_mapping, _, _ = build_position_mapping(
                runner, clean_text, corr_text, clean_labels, corr_labels
            )

        with P("create_metric"):
            metric = create_metric(runner, clean_sample, corr_sample, clean_text, corr_text)

        print(f"    logit_diff: clean={metric.clean_val:.3f}, corr={metric.corr_val:.3f}, delta={metric.diff:.3f}")

        with P("run_methods"):
            results = run_all_attribution_methods(
                runner, clean_text, corr_text, metric, pos_mapping, ig_steps
            )

        # Normalize by metric difference
        if abs(metric.diff) > 1e-8:
            results = {k: v / abs(metric.diff) for k, v in results.items()}

        all_results.append(results)

        # Get labels from first pair
        if token_labels is None:
            token_labels = get_token_labels(runner, clean_text)
            section_markers = find_section_markers(
                runner, clean_text, clean_sample.short_term_label, clean_sample.long_term_label
            )

    return all_results, token_labels or [], section_markers or {}


def save_heatmap(
    scores: np.ndarray,
    layers: list[int],
    token_labels: list[str],
    save_path: Path,
    method_key: str,
    model_name: str,
    n_pairs: int,
    section_markers: dict[str, int],
) -> None:
    """Save heatmap with consistent formatting."""
    info = METHOD_INFO.get(method_key, {"name": method_key, "group": "other"})
    name = info["name"]

    # Stats for subtitle
    abs_max = np.abs(scores).max()
    max_val = scores.max()
    min_val = scores.min()

    plot_layer_position_heatmap(
        scores,
        layers,
        token_labels,
        save_path,
        title=f"{name}",
        subtitle=f"{model_name} | n={n_pairs} | range=[{min_val:.3f}, {max_val:.3f}]",
        cbar_label="Attribution (normalized)",
        cmap="RdBu_r",
        section_markers=section_markers,
    )


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

            # Find top attribution
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
        epilog="""
Methods computed:
  Standard:  resid, attn, mlp - (clean - corrupted) · grad
  EAP:       eap_attn, eap_mlp - Edge attribution to residual stream
  EAP-IG:    eap_ig_attn, eap_ig_mlp - EAP with integrated gradients
        """,
    )
    parser.add_argument("--preference-data", type=str, help="Preference data ID")
    parser.add_argument("--max-pairs", type=int, default=3, help="Number of pairs to process (default: 3)")
    parser.add_argument("--ig-steps", type=int, default=10, help="Integrated gradients steps (default: 10)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
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

        # Run attribution
        print(f"\n=== Running Attribution Patching ({args.max_pairs} pairs) ===")
        with P("attribution"):
            all_results, token_labels, section_markers = run_attribution_on_pairs(
                runner, pairs, args.max_pairs, args.ig_steps
            )

        # Aggregate results
        with P("aggregate"):
            aggregated = aggregate_attribution_results(all_results, runner.n_layers)

        # Create output directory
        run_dir = args.output / get_timestamp()
        ensure_dir(run_dir)

        # Save heatmaps
        layers = list(range(runner.n_layers))
        model_name = pref_data.model.split("/")[-1]
        n_pairs = min(len(pairs), args.max_pairs)

        print("\n=== Saving Heatmaps ===")
        with P("save_heatmaps"):
            for key, scores in aggregated.items():
                save_heatmap(
                    scores, layers, token_labels,
                    run_dir / f"heatmap_{key}.png",
                    key, model_name, n_pairs, section_markers,
                )
                np.save(run_dir / f"{key}.npy", scores)

        # Print summary
        summary = print_summary(aggregated, layers, token_labels)

        # Save standardized positions.json (using resid as primary method)
        if "resid" in aggregated:
            scores = aggregated["resid"]
            threshold = 0.1  # Attribution score threshold
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

            positions_file = PositionsFile(
                model=pref_data.model,
                method="attribution_patching",
                positions=sorted(positions, key=lambda p: abs(p.score), reverse=True),
                dataset_id=pref_data.dataset_id,
                threshold=threshold,
                component="resid_post",
            )
            positions_file.save(run_dir / "positions.json")

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
            "methods": list(METHOD_INFO.keys()),
            "summary": summary,
        }, run_dir / "metadata.json")

        print(f"\nResults saved to: {run_dir}")

    P.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
