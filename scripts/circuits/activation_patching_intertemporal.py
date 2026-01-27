#!/usr/bin/env python
"""
Activation patching for intertemporal preference analysis.

Produces:
- Position sweep (tall vertical heatmap)
- Full layer x position heatmap with section markers

Usage:
    python scripts/circuits/activation_patching_intertemporal.py
    python scripts/circuits/activation_patching_intertemporal.py --components  # Include MLP/attention
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import ModelRunner
from src.data import load_pref_data_with_prompts, build_prompt_pairs, find_preference_data, get_preference_data_id
from src.common.io import ensure_dir, save_json, get_timestamp
from src.analysis import (
    build_position_mapping,
    create_metric,
    find_section_markers,
    get_token_labels,
)
from src.common.positions_schema import PositionsFile, PositionSpec
from src.viz import plot_layer_position_heatmap
from src.profiler import P

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "out" / "activation_patching"
POSITION_THRESHOLD = 0.05  # Only include positions with recovery > threshold


def run_position_sweep(
    runner: ModelRunner,
    clean_text: str,
    corrupted_text: str,
    metric,
    pos_mapping: dict[int, int],
    clean_len: int,
    component: str = "resid_post",
) -> np.ndarray:
    """Patch ALL layers at each position. Returns 1D array of recovery values."""
    layers = list(range(runner.n_layers))
    hook_names = [f"blocks.{l}.hook_{component}" for l in layers]
    _, clean_cache = runner.run_with_cache(clean_text, names_filter=lambda n: n in hook_names)

    formatted = runner._apply_chat_template(corrupted_text)
    input_ids = runner.tokenize(formatted)
    corr_len = input_ids.shape[1]

    results = np.zeros(clean_len)

    for clean_pos in range(clean_len):
        corr_pos = pos_mapping.get(clean_pos, clean_pos)

        def make_hook(layer_idx):
            hook_name = f"blocks.{layer_idx}.hook_{component}"
            clean_act = clean_cache[hook_name]
            def hook(act, hook=None):
                if corr_pos < act.shape[1] and clean_pos < clean_act.shape[1]:
                    act[:, corr_pos, :] = clean_act[0, clean_pos, :].detach()
                return act
            return hook

        hooks = [(f"blocks.{l}.hook_{component}", make_hook(l)) for l in layers]
        with torch.no_grad():
            logits = runner.model.run_with_hooks(input_ids, fwd_hooks=hooks)
        results[clean_pos] = metric(logits)

    return results


def run_full_sweep(
    runner: ModelRunner,
    clean_text: str,
    corrupted_text: str,
    metric,
    pos_mapping: dict[int, int],
    positions: list[int],
    layers: list[int],
    component: str = "resid_post",
) -> np.ndarray:
    """Full layer x position patching sweep for specified positions."""
    hook_names = [f"blocks.{l}.hook_{component}" for l in layers]
    _, clean_cache = runner.run_with_cache(clean_text, names_filter=lambda n: n in hook_names)

    formatted = runner._apply_chat_template(corrupted_text)
    input_ids = runner.tokenize(formatted)

    results = np.zeros((len(layers), len(positions)))

    for li, layer in enumerate(layers):
        hook_name = f"blocks.{layer}.hook_{component}"
        clean_act = clean_cache[hook_name]

        for pi, clean_pos in enumerate(positions):
            corr_pos = pos_mapping.get(clean_pos, clean_pos)

            def hook(act, hook=None, cp=clean_pos, rp=corr_pos):
                if rp < act.shape[1] and cp < clean_act.shape[1]:
                    act[:, rp, :] = clean_act[0, cp, :].detach()
                return act

            with torch.no_grad():
                logits = runner.model.run_with_hooks(input_ids, fwd_hooks=[(hook_name, hook)])
            results[li, pi] = metric(logits)

        print(f"  Layer {layer}: max={results[li].max():.3f}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Activation patching analysis")
    parser.add_argument("--preference-data", type=str, help="Preference data ID")
    parser.add_argument("--max-pairs", type=int, default=2, help="Max pairs to process")
    parser.add_argument("--components", action="store_true", help="Include MLP/attention")
    parser.add_argument("--threshold", type=float, default=POSITION_THRESHOLD,
                        help="Position threshold for full sweep")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()

    with P("total"):
        pref_dir = PROJECT_ROOT / "out" / "preference_data"
        data_dir = PROJECT_ROOT / "out" / "datasets"

        if args.preference_data:
            pref_id = args.preference_data
        else:
            recent = find_preference_data(pref_dir)
            if not recent:
                print("No preference data found")
                return 1
            pref_id = get_preference_data_id(recent)

        with P("load_data"):
            pref_data = load_pref_data_with_prompts(pref_id, pref_dir, data_dir)
            print(f"Loaded {len(pref_data.preferences)} samples, model: {pref_data.model}")
            pairs = build_prompt_pairs(pref_data, max_pairs=args.max_pairs, include_response=True)
            print(f"Built {len(pairs)} pairs")

        if not pairs:
            print("No pairs!")
            return 1

        with P("load_model"):
            runner = ModelRunner(pref_data.model)
            print(f"Model: {runner.n_layers} layers")

        run_dir = args.output / get_timestamp()
        ensure_dir(run_dir)

        clean_text, corrupted_text, clean_sample, corrupted_sample = pairs[0]
        clean_labels = [clean_sample.short_term_label, clean_sample.long_term_label]
        corrupted_labels = [corrupted_sample.short_term_label, corrupted_sample.long_term_label]

        with P("build_mapping"):
            pos_mapping, clean_len, corr_len = build_position_mapping(
                runner, clean_text, corrupted_text, clean_labels, corrupted_labels,
            )
            print(f"Sequence lengths: clean={clean_len}, corrupted={corr_len}")

        token_labels = get_token_labels(runner, clean_text)
        section_markers = find_section_markers(
            runner, clean_text,
            clean_sample.short_term_label,
            clean_sample.long_term_label,
        )
        print(f"Section markers: {section_markers}")

        with P("create_metric"):
            metric = create_metric(runner, clean_sample, corrupted_sample, clean_text, corrupted_text)
            print(f"Baselines: clean={metric.clean_val:.3f}, corrupted={metric.corr_val:.3f}, diff={metric.diff:.3f}")

        # Components to analyze
        components = ["resid_post"]
        if args.components:
            components.extend(["attn_out", "mlp_out"])

        # Use 12 evenly spaced layers
        n_layers = runner.n_layers
        layers = [int(i * (n_layers - 1) / 11) for i in range(12)]

        # Phase 1: Run position sweeps for ALL components
        print("\n=== Position Sweeps ===")
        pos_sweep_results = {}

        for component in components:
            comp_name = {"resid_post": "Residual", "attn_out": "Attention", "mlp_out": "MLP"}.get(component, component)
            print(f"{comp_name}...")
            with P(f"position_sweep_{component}"):
                pos_results = run_position_sweep(
                    runner, clean_text, corrupted_text, metric,
                    pos_mapping, clean_len, component,
                )
            pos_sweep_results[component] = pos_results
            above = np.where(pos_results > args.threshold)[0]
            print(f"  Max: {pos_results.max():.3f} at pos {pos_results.argmax()}, {len(above)} above threshold")

        # Use resid_post to determine filtered positions (master list)
        resid_above = np.where(pos_sweep_results["resid_post"] > args.threshold)[0]
        filtered_positions = sorted(resid_above.tolist())
        if not filtered_positions:
            print("No positions above threshold for resid_post!")
            return 1

        print(f"\nFiltered: {len(filtered_positions)} positions above threshold {args.threshold}")
        filtered_labels = [token_labels[i] for i in filtered_positions]

        # Remap section markers to filtered positions (find nearest if exact not present)
        filtered_markers = {}
        for name, pos in section_markers.items():
            if pos in filtered_positions:
                filtered_markers[name] = filtered_positions.index(pos)
            else:
                # Find the nearest filtered position that's <= the marker position
                candidates = [i for i, fp in enumerate(filtered_positions) if fp <= pos]
                if candidates:
                    filtered_markers[name] = max(candidates)

        # Save position sweep heatmaps - all same shape (layers x filtered_positions)
        # Broadcast 1D position sweep to 2D (same value across all layers)
        for component in components:
            comp_name = {"resid_post": "Residual", "attn_out": "Attention", "mlp_out": "MLP"}.get(component, component)
            pos_results = pos_sweep_results[component]

            # Extract values for filtered positions and broadcast to (n_layers x n_positions)
            filtered_values = np.array([pos_results[i] for i in filtered_positions])
            broadcast_matrix = np.tile(filtered_values, (len(layers), 1))

            # For non-resid components, mask positions below that component's threshold as NaN
            if component != "resid_post":
                for pi, pos in enumerate(filtered_positions):
                    if pos_results[pos] <= args.threshold:
                        broadcast_matrix[:, pi] = np.nan

            # Determine filename - resid_post is just "position_sweep", others include component
            if component == "resid_post":
                filename = "position_sweep.png"
                title = "Position Sweep (All Layers)"
            else:
                short_name = {"attn_out": "attn", "mlp_out": "mlp"}.get(component, component)
                filename = f"position_sweep_{short_name}.png"
                title = f"Position Sweep ({comp_name})"

            plot_layer_position_heatmap(
                broadcast_matrix, layers, filtered_labels,
                run_dir / filename,
                title=title,
                cbar_label="Recovery", vmin=0.0, vmax=1.0,
                section_markers=filtered_markers,
            )

        # Phase 2: Run full sweeps for all components using SAME positions
        print("\n=== Full Sweeps ===")
        for component in components:
            comp_name = {"resid_post": "Residual", "attn_out": "Attention", "mlp_out": "MLP"}.get(component, component)
            short_name = {"resid_post": "resid", "attn_out": "attn", "mlp_out": "mlp"}.get(component, component)
            print(f"{comp_name}...")

            with P(f"full_sweep_{component}"):
                full_results = run_full_sweep(
                    runner, clean_text, corrupted_text, metric,
                    pos_mapping, filtered_positions, layers, component,
                )

            # For non-resid components, mask positions below that component's threshold as NaN
            if component != "resid_post":
                pos_results = pos_sweep_results[component]
                for pi, pos in enumerate(filtered_positions):
                    if pos_results[pos] <= args.threshold:
                        full_results[:, pi] = np.nan

            valid_max = np.nanmax(full_results) if not np.all(np.isnan(full_results)) else 0.0
            print(f"  Max: {valid_max:.3f}")

            plot_layer_position_heatmap(
                full_results, layers, filtered_labels,
                run_dir / f"heatmap_{short_name}.png",
                title=f"Activation Patching ({comp_name})",
                cbar_label="Recovery", vmin=0.0, vmax=1.0,
                section_markers=filtered_markers,
            )

            # Save standardized positions.json for downstream scripts
            positions = []
            for li, layer in enumerate(layers):
                for pi, pos in enumerate(filtered_positions):
                    recovery = float(full_results[li, pi])
                    if not np.isnan(recovery) and recovery > args.threshold:
                        # Determine section for this position
                        section = None
                        for sec_name, sec_pos in section_markers.items():
                            if pos >= sec_pos:
                                section = sec_name.replace("before_", "")
                        positions.append(PositionSpec(
                            position=pos,
                            token=token_labels[pos],
                            score=recovery,
                            layer=layer,
                            section=section,
                        ))

            positions_file = PositionsFile(
                model=pref_data.model,
                method="activation_patching",
                positions=sorted(positions, key=lambda p: p.score, reverse=True),
                dataset_id=pref_data.dataset_id,
                threshold=args.threshold,
                component=component,
            )
            positions_file.save(run_dir / f"positions_{short_name}.json")

        save_json({
            "timestamp": get_timestamp(),
            "model": pref_data.model,
            "dataset_id": pref_data.dataset_id,
            "section_markers": section_markers,
            "components": components,
            "threshold": args.threshold,
        }, run_dir / "metadata.json")

        print(f"\nSaved to: {run_dir}")

    P.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
