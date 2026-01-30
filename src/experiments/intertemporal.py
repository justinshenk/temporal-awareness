"""
Intertemporal preference experiment module.

Provides configs and the main experiment runner for analyzing
temporal preferences in language models.

IMPORTANT DESIGN PRINCIPLES:
1. NEVER use TransformerLens/NNsight/Pyvene directly - ALWAYS use ModelRunner API
2. All analysis functions use ModelRunner methods (run_with_cache, forward_with_intervention, etc.)
3. No magic numbers in configs - use named constants or explicit config values
4. Experiments should be reproducible (set random seeds, log all configs)
"""

from __future__ import annotations

import gc
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..common.io import ensure_dir, save_json, get_timestamp
from ..common.token_positions import build_position_labels
from ..data import (
    generate_preference_data,
    load_pref_data_with_prompts,
    build_prompt_pairs,
    PreferenceData,
)
from ..models import ModelRunner
from ..viz import plot_layer_position_heatmap, plot_position_sweep
from ..profiler import P

from .activation_patching import run_activation_patching
from .attribution_patching import run_attribution_patching
from .steering import compute_steering_vector, apply_steering
from .probe_training import run_probe_training


@dataclass
class ExperimentArgs:
    """Arguments for running the experiment."""

    config: dict  # Experiment configuration (model, dataset_config, etc.)
    preference_data: Optional[str] = None
    output: Optional[Path] = None
    project_root: Optional[Path] = None
    config_name: str = "custom"  # For display purposes


@dataclass
class ExperimentContext:
    """Shared state passed between experiment steps."""

    config: dict
    pref_data: PreferenceData
    runner: ModelRunner
    viz_dir: Path
    data_dir: Path


@dataclass
class PatchingResult:
    """Results from activation patching."""

    best_layer: int
    best_position: int
    best_token: str
    position_sweep: np.ndarray
    full_sweeps: dict[str, np.ndarray]  # component name -> [n_layers, n_positions]
    filtered_positions: list[int]
    layer_indices: list[int]


@dataclass
class SteeringDirection:
    """A named steering direction with its metadata."""

    name: str  # e.g. "contrastive", "probe_choice"
    direction: np.ndarray  # (d_model,) vector
    layer: int
    metadata: dict  # Source-specific stats (norm, accuracy, etc.)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _get_memory_usage() -> dict:
    stats = {}
    if torch.cuda.is_available():
        stats["cuda_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        stats["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
    if hasattr(torch.mps, "current_allocated_memory"):
        try:
            stats["mps_allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
        except Exception:
            pass
    return stats


def _log_memory(stage: str):
    mem = _get_memory_usage()
    if mem:
        mem_str = ", ".join(f"{k}={v:.2f}" for k, v in mem.items())
        print(f"  [Memory @ {stage}] {mem_str}")


def _print_step(number: int, title: str):
    print(f"\n{'=' * 60}")
    print(f"STEP {number}: {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def step_preference_data(
    args: ExperimentArgs,
    project_root: Path,
    data_dir: Path,
) -> PreferenceData:
    """Step 1: Load or generate preference data."""
    _print_step(1, "PREFERENCE DATA")
    config = args.config

    pref_dir = project_root / "out" / "preference_data"
    datasets_dir = project_root / "out" / "datasets"

    if args.preference_data:
        with P("load_data"):
            pref_data = load_pref_data_with_prompts(
                args.preference_data, pref_dir, datasets_dir
            )
        print(f"Loaded existing: {args.preference_data}")
    else:
        with P("generate_data"):
            pref_data = generate_preference_data(
                model=config["model"],
                dataset_config=config["dataset_config"],
                max_samples=config["max_samples"],
                verbose=True,
                pref_dir=pref_dir,
                datasets_dir=datasets_dir,
                internals=config.get("internals"),
            )

    print(f"Model: {pref_data.model}")
    print(f"Samples: {len(pref_data.preferences)}")

    save_json(
        {
            "model": pref_data.model,
            "dataset_id": pref_data.dataset_id,
            "n_samples": len(pref_data.preferences),
            "choices": {
                "short_term": sum(
                    1 for p in pref_data.preferences if p.choice == "short_term"
                ),
                "long_term": sum(
                    1 for p in pref_data.preferences if p.choice == "long_term"
                ),
            },
        },
        data_dir / "preference_summary.json",
    )
    _log_memory("after_data")
    return pref_data


def step_load_model(pref_data: PreferenceData) -> ModelRunner:
    """Step 2: Load the model."""
    _print_step(2, "LOAD MODEL")

    with P("load_model"):
        runner = ModelRunner(pref_data.model)
        print(f"Model loaded: {runner.n_layers} layers, d_model={runner.d_model}")
    _log_memory("after_model_load")
    return runner


def step_activation_patching(ctx: ExperimentContext) -> PatchingResult:
    """Step 3: Activation patching to identify important positions/layers."""
    _print_step(3, "ACTIVATION PATCHING")
    config = ctx.config

    act_n_layers = config.get("act_patch_n_layers_sample", 12)
    act_pos_sweep_component = "resid_post"
    act_full_sweep_components = config.get(
        "act_patch_full_sweep_components", ["resid_post", "attn_out", "mlp_out"]
    )
    act_pos_step = config.get("act_patch_position_step", 1)
    max_pairs = config.get("max_pairs", 1)
    position_threshold = config.get("position_threshold", 0.01)
    # None = use DefaultPromptFormat().get_interesting_positions()
    act_token_positions = config.get("act_patch_token_positions", None)

    with P("activation_patching"):
        pos_sweep, full_sweeps, filtered_pos, token_labels, section_markers, pair_metadata = (
            run_activation_patching(
                ctx.runner,
                ctx.pref_data,
                max_pairs=config["max_pairs"],
                threshold=config["position_threshold"],
                position_sweep_component=act_pos_sweep_component,
                full_sweep_components=act_full_sweep_components,
                n_layers_sample=act_n_layers,
                position_step=act_pos_step,
                token_positions=act_token_positions,
            )
        )

    # Compute layer indices (must match run_activation_patching logic)
    actual_n_layers = min(act_n_layers, ctx.runner.n_layers)
    if actual_n_layers > 1:
        layer_indices = [
            int(i * (ctx.runner.n_layers - 1) / (actual_n_layers - 1))
            for i in range(actual_n_layers)
        ]
    else:
        layer_indices = [ctx.runner.n_layers // 2]

    # Find best position/layer using resid_post (primary component)
    primary_component = act_full_sweep_components[0]
    primary_sweep = full_sweeps[primary_component]
    best_pos_idx = int(np.argmax(pos_sweep))
    if best_pos_idx in filtered_pos:
        col_idx = filtered_pos.index(best_pos_idx)
        best_layer_idx = int(np.argmax(primary_sweep[:, col_idx]))
    else:
        best_layer_idx = (
            int(np.argmax(primary_sweep[:, 0])) if primary_sweep.size > 0 else 0
        )
    best_layer = layer_indices[best_layer_idx] if layer_indices else 0
    best_token = (
        token_labels[best_pos_idx]
        if best_pos_idx < len(token_labels)
        else f"pos{best_pos_idx}"
    )

    print(f"Position sweep: max={pos_sweep.max():.3f}, argmax={best_pos_idx}")
    print(f"Filtered positions: {len(filtered_pos)} (threshold={config['position_threshold']})")
    print(f"Components: {act_full_sweep_components}")
    for comp, sweep in full_sweeps.items():
        print(f"  {comp} sweep shape: {sweep.shape}")
    print(f"Layers sampled: {act_n_layers}, Position step: {act_pos_step}")
    print(f"Best position: {best_pos_idx} ({best_token})")
    print(f"Best layer: {best_layer}")

    # Compute label probabilities for each pair
    choice_prefix = "I select:"
    pairs = build_prompt_pairs(ctx.pref_data, max_pairs=max_pairs, include_response=False)
    for pm, pair in zip(pair_metadata, pairs):
        clean_text, corrupted_text, clean_sample, corrupted_sample = pair
        clean_labels = (clean_sample.short_term_label, clean_sample.long_term_label)
        corr_labels = (corrupted_sample.short_term_label, corrupted_sample.long_term_label)

        # Get probs for clean prompt with its own labels
        clean_probs = ctx.runner.get_label_probs(clean_text, choice_prefix, clean_labels)
        pm["clean_label_probs"] = {
            clean_labels[0]: float(clean_probs[0]),
            clean_labels[1]: float(clean_probs[1]),
        }

        # Get probs for corrupted prompt with its own labels
        corr_probs = ctx.runner.get_label_probs(corrupted_text, choice_prefix, corr_labels)
        pm["corrupted_label_probs"] = {
            corr_labels[0]: float(corr_probs[0]),
            corr_labels[1]: float(corr_probs[1]),
        }

        # If clean and corrupted use different label formats, also get cross-format probs
        if clean_labels != corr_labels:
            clean_with_corr = ctx.runner.get_label_probs(clean_text, choice_prefix, corr_labels)
            pm["clean_label_probs_alt_format"] = {
                corr_labels[0]: float(clean_with_corr[0]),
                corr_labels[1]: float(clean_with_corr[1]),
            }
            corr_with_clean = ctx.runner.get_label_probs(corrupted_text, choice_prefix, clean_labels)
            pm["corrupted_label_probs_alt_format"] = {
                clean_labels[0]: float(corr_with_clean[0]),
                clean_labels[1]: float(corr_with_clean[1]),
            }

    # Save results
    np.save(ctx.data_dir / "position_sweep.npy", pos_sweep)
    for comp, sweep in full_sweeps.items():
        np.save(ctx.data_dir / f"full_sweep_{comp}.npy", sweep)
    save_json(
        {
            "filtered_positions": filtered_pos,
            "layer_indices": layer_indices,
            "token_labels": token_labels,
            "section_markers": section_markers,
            "position_sweep": [float(v) for v in pos_sweep],
            "position_sweep_max": float(pos_sweep.max()),
            "position_sweep_argmax": best_pos_idx,
            "position_sweep_component": act_pos_sweep_component,
            "full_sweep_components": act_full_sweep_components,
            "full_sweep": {
                comp: [[float(v) for v in row] for row in sweep]
                for comp, sweep in full_sweeps.items()
            },
            "n_layers_sample": act_n_layers,
            "position_step": act_pos_step,
            "best_position": best_pos_idx,
            "best_position_token": best_token,
            "best_layer": best_layer,
            "pairs": pair_metadata,
        },
        ctx.data_dir / "activation_patching.json",
    )

    # Plot position sweep (Phase 1)
    plot_position_sweep(
        pos_sweep,
        token_labels,
        ctx.viz_dir / "activation_patching_position_sweep.png",
        title=f"Position Sweep ({act_pos_sweep_component})",
        cbar_label="Recovery",
        vmin=0.0,
        vmax=max(1.0, float(pos_sweep.max())),
        section_markers=section_markers,
    )
    print(f"  Saved: {ctx.viz_dir / 'activation_patching_position_sweep.png'}")

    # Plot heatmap per component (Phase 2)
    filtered_labels = [
        token_labels[i] if i < len(token_labels) else f"pos{i}"
        for i in filtered_pos
    ]
    section_markers_filtered = {
        k: filtered_pos.index(v)
        for k, v in section_markers.items()
        if v in filtered_pos
    }
    for comp, sweep in full_sweeps.items():
        if len(filtered_pos) > 0 and sweep.size > 0:
            filename = f"activation_patching_{comp}.png"
            plot_layer_position_heatmap(
                sweep,
                layer_indices,
                filtered_labels,
                ctx.viz_dir / filename,
                title=f"Activation Patching ({comp})",
                cbar_label="Recovery",
                vmin=0.0,
                vmax=1.0,
                section_markers=section_markers_filtered,
            )
            print(f"  Saved: {ctx.viz_dir / filename}")

    _log_memory("after_act_patching")

    return PatchingResult(
        best_layer=best_layer,
        best_position=best_pos_idx,
        best_token=best_token,
        position_sweep=pos_sweep,
        full_sweeps=full_sweeps,
        filtered_positions=filtered_pos,
        layer_indices=layer_indices,
    )


def step_attribution_patching(ctx: ExperimentContext) -> dict:
    """Step 4: Attribution patching methods."""
    _print_step(4, "ATTRIBUTION PATCHING")
    config = ctx.config

    with P("attribution_patching"):
        attribution_results, attr_labels, attr_markers = run_attribution_patching(
            ctx.runner,
            ctx.pref_data,
            max_pairs=config["max_pairs"],
            ig_steps=config["ig_steps"],
        )

    # Save and plot each method
    layers = list(range(ctx.runner.n_layers))
    for key, scores in attribution_results.items():
        np.save(ctx.data_dir / f"attribution_{key}.npy", scores)
        plot_layer_position_heatmap(
            scores,
            layers,
            attr_labels,
            ctx.viz_dir / f"attribution_{key}.png",
            title=f"Attribution: {key}",
            cbar_label="Attribution",
            cmap="RdBu_r",
            section_markers=attr_markers,
        )
        print(f"  {key}: max={scores.max():.4f}, min={scores.min():.4f}")
        print(f"    Saved: {ctx.viz_dir / f'attribution_{key}.png'}")

    save_json(
        {
            "methods": list(attribution_results.keys()),
            "token_labels": attr_labels,
            "section_markers": attr_markers,
        },
        ctx.data_dir / "attribution_metadata.json",
    )
    _log_memory("after_attr_patching")
    return attribution_results


def step_contrastive_steering(
    ctx: ExperimentContext,
    patching: PatchingResult,
) -> SteeringDirection:
    """Step 5: Compute steering vector from contrastive activations."""
    _print_step(5, "CONTRASTIVE STEERING VECTOR")

    with P("contrastive_steering"):
        direction, stats = compute_steering_vector(
            ctx.runner,
            ctx.pref_data,
            layer=patching.best_layer,
            position=patching.best_position,
            max_samples=ctx.config["contrastive_max_samples"],
        )

    print(f"Layer: {stats['layer']}, Position: {stats['position']}")
    print(f"Direction norm: {stats['direction_norm']:.4f}")
    print(f"Samples: class0={stats['n_class0']}, class1={stats['n_class1']}")

    save_json(
        {
            "type": "steering_vector",
            "source": "contrastive",
            "model": ctx.pref_data.model,
            "layer": patching.best_layer,
            "position": patching.best_position,
            "token": patching.best_token,
            "direction": direction.tolist(),
            **stats,
        },
        ctx.data_dir / "steering_contrastive.json",
    )
    print(f"  Saved: {ctx.data_dir / 'steering_contrastive.json'}")
    _log_memory("after_contrastive_steering")

    return SteeringDirection(
        name="contrastive",
        direction=direction,
        layer=patching.best_layer,
        metadata=stats,
    )


def step_steering_eval(
    ctx: ExperimentContext,
    directions: list[SteeringDirection],
) -> dict:
    """Step 7: Evaluate all steering directions on preference data prompts."""
    _print_step(7, "STEERING EVALUATION")
    config = ctx.config

    max_eval = config.get("steering_eval_max_samples", 10)
    strengths = config["steering_strengths"]
    eval_prefs = ctx.pref_data.preferences[:max_eval]

    dir_names = [d.name for d in directions]
    print(f"Directions: {dir_names}")
    print(f"Evaluating {len(eval_prefs)} prompts x {len(strengths)} strengths each")

    all_eval_results = {}
    with P("steering_eval"):
        for sd in directions:
            print(f"\n--- {sd.name} (layer {sd.layer}) ---")
            eval_results = []

            for pref in eval_prefs:
                prompt = pref.prompt_text
                print(f"\nPrompt: {prompt[:60]}...")
                print(f"  Original choice: {pref.choice}")
                prompt_results = {
                    "prompt": prompt,
                    "original_choice": pref.choice,
                    "responses": [],
                }

                for strength in strengths:
                    try:
                        response = apply_steering(
                            ctx.runner,
                            prompt,
                            sd.direction,
                            layer=sd.layer,
                            strength=strength,
                            max_new_tokens=50,
                        )
                        prompt_results["responses"].append(
                            {
                                "strength": strength,
                                "response": response,
                            }
                        )
                        display = response.replace("\n", " ")[:60]
                        print(f"  strength={strength:+.1f}: {display}...")
                    except Exception as e:
                        print(f"  strength={strength:+.1f}: ERROR - {e}")
                        prompt_results["responses"].append(
                            {
                                "strength": strength,
                                "error": str(e),
                            }
                        )

                eval_results.append(prompt_results)

            all_eval_results[sd.name] = eval_results
            save_json(
                eval_results,
                ctx.data_dir / f"steering_eval_{sd.name}.json",
            )
            print(f"\n  Saved: {ctx.data_dir / f'steering_eval_{sd.name}.json'}")

    return all_eval_results


def step_probe_training(ctx: ExperimentContext) -> tuple[dict, Optional[SteeringDirection]]:
    """Step 6: Train linear probes and extract probe steering vector."""
    _print_step(6, "PROBE TRAINING")
    config = ctx.config

    probe_layers = config.get("probe_layers", None)
    if probe_layers is None:
        n = ctx.runner.n_layers
        probe_layers = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))

    # None means "use DefaultPromptFormat().get_interesting_positions()"
    probe_positions = config.get("probe_positions", None)
    probe_max_samples = config.get("probe_max_samples", 200)

    with P("probe_training"):
        probe_results, probes = run_probe_training(
            ctx.runner,
            ctx.pref_data,
            layers=probe_layers,
            token_positions=probe_positions,
            test_split=0.2,
            random_seed=42,
            max_samples=probe_max_samples,
        )

    # Save results and visualizations per probe type
    for probe_type in ["choice", "time_horizon"]:
        type_results = probe_results.get(probe_type, [])
        if not type_results:
            continue

        best = max(type_results, key=lambda r: r.test_accuracy)
        print(
            f"  {probe_type}: Best L{best.layer} P{best.token_position} = {best.test_accuracy:.3f}"
        )

        save_json(
            {
                "probe_type": probe_type,
                "model": ctx.pref_data.model,
                "layers": probe_results["_meta"]["layers"],
                "best": {
                    "layer": best.layer,
                    "position": best.token_position,
                    "test_accuracy": best.test_accuracy,
                },
                "results": [
                    {
                        "layer": r.layer,
                        "position": r.token_position,
                        "test_accuracy": r.test_accuracy,
                        "train_accuracy": r.train_accuracy,
                    }
                    for r in type_results
                ],
            },
            ctx.data_dir / f"probe_{probe_type}_results.json",
        )

        # Heatmap
        resolved_layers = probe_results["_meta"]["layers"]
        actual_positions = probe_results["_meta"]["token_positions"]
        pos_info = probe_results["_meta"].get(
            f"{probe_type}_position_info"
        ) or probe_results["_meta"].get("choice_position_info")
        pos_labels = build_position_labels(actual_positions, pos_info)

        matrix = np.full((len(resolved_layers), len(actual_positions)), np.nan)
        layer_idx_map = {l: i for i, l in enumerate(resolved_layers)}
        for r in type_results:
            matrix[layer_idx_map[r.layer], r.token_position] = r.test_accuracy

        title_map = {"choice": "Choice Probe", "time_horizon": "Time Horizon Probe"}
        plot_layer_position_heatmap(
            matrix,
            resolved_layers,
            pos_labels,
            ctx.viz_dir / f"probe_{probe_type}.png",
            title=f"{title_map.get(probe_type, probe_type)}",
            cbar_label="Test Accuracy",
            vmin=0.5,
            vmax=1.0,
        )
        print(f"    Saved: {ctx.viz_dir / f'probe_{probe_type}.png'}")

        # Save best probe's steering vector
        best_probe = probes.get((probe_type, best.layer, best.token_position))
        if best_probe:
            save_json(
                {
                    "type": "steering_vector",
                    "source": "linear_probe",
                    "probe_type": probe_type,
                    "model": ctx.pref_data.model,
                    "layer": best.layer,
                    "position": best.token_position,
                    "test_accuracy": best.test_accuracy,
                    "direction": best_probe.get_steering_vector().tolist(),
                    "bias": best_probe.get_bias(),
                },
                ctx.data_dir / f"steering_probe_{probe_type}.json",
            )

    # Extract best choice probe's direction as a SteeringDirection
    probe_steering = None
    choice_results = probe_results.get("choice", [])
    if choice_results:
        best_choice = max(choice_results, key=lambda r: r.test_accuracy)
        best_choice_probe = probes.get(("choice", best_choice.layer, best_choice.token_position))
        if best_choice_probe:
            probe_steering = SteeringDirection(
                name="probe_choice",
                direction=best_choice_probe.get_steering_vector(),
                layer=best_choice.layer,
                metadata={
                    "test_accuracy": best_choice.test_accuracy,
                    "train_accuracy": best_choice.train_accuracy,
                    "token_position": best_choice.token_position,
                },
            )
            print(f"  Probe steering: layer {probe_steering.layer}, "
                  f"accuracy {best_choice.test_accuracy:.3f}")

    _log_memory("after_probes")
    return probe_results, probe_steering


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_experiment(args: ExperimentArgs) -> int:
    """Run the full intertemporal preference experiment.

    Steps:
        1. Load/generate preference data
        2. Load model
        3. Activation patching
        4. Attribution patching
        5. Contrastive steering vector
        6. Probe training (+ probe steering vector)
        7. Evaluate all steering directions on preference prompts
    """
    config = args.config
    project_root = args.project_root or Path(__file__).parent.parent.parent
    output_dir = args.output or project_root / "out" / "experiments"

    print(f"Config: {args.config_name}")
    print(f"Model: {config['model']}")
    print(f"Max samples: {config['max_samples']}")

    with P("total"):
        ts = get_timestamp()
        run_dir = output_dir / ts

        viz_dir = run_dir / "viz"
        data_dir = run_dir / "data"
        ensure_dir(viz_dir)
        ensure_dir(data_dir)

        print(f"Output: {run_dir}")

        save_json(
            {
                "config": config,
                "args": {"config_name": args.config_name},
                "timestamp": ts,
            },
            run_dir / "config.json",
        )

        try:
            skip = set(config.get("skip", []))
            if skip:
                print(f"Skipping: {', '.join(sorted(skip))}")

            pref_data = step_preference_data(args, project_root, data_dir)
            runner = step_load_model(pref_data)

            ctx = ExperimentContext(
                config=config,
                pref_data=pref_data,
                runner=runner,
                viz_dir=viz_dir,
                data_dir=data_dir,
            )

            patching = None
            if "activation_patching" not in skip:
                patching = step_activation_patching(ctx)

            if "attribution_patching" not in skip:
                step_attribution_patching(ctx)

            # Compute steering vectors from available sources
            contrastive_sd = None
            if "contrastive_steering" not in skip and patching is not None:
                contrastive_sd = step_contrastive_steering(ctx, patching)

            probe_sd = None
            if "probe_training" not in skip:
                _, probe_sd = step_probe_training(ctx)

            # Evaluate all available steering directions
            directions = []
            if contrastive_sd is not None:
                directions.append(contrastive_sd)
            if probe_sd is not None:
                directions.append(probe_sd)
            if directions and "steering_eval" not in skip:
                step_steering_eval(ctx, directions)

            # Save final metadata
            results_meta = {}
            if patching is not None:
                results_meta.update({
                    "best_position": patching.best_position,
                    "best_layer": patching.best_layer,
                    "position_token": patching.best_token,
                    "position_sweep_max": float(patching.position_sweep.max()),
                    "n_filtered_positions": len(patching.filtered_positions),
                })
            if contrastive_sd is not None:
                results_meta["contrastive_norm"] = contrastive_sd.metadata["direction_norm"]
            results_meta["steering_directions"] = [d.name for d in directions]

            save_json(
                {
                    "timestamp": ts,
                    "model": pref_data.model,
                    "dataset_id": pref_data.dataset_id,
                    "config": config,
                    "results": results_meta,
                },
                run_dir / "metadata.json",
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"\n{'=' * 60}")
            print("COMPLETE")
            print(f"{'=' * 60}")
            print(f"Output directory: {run_dir}")
            print("  viz/  - Heatmaps and plots")
            print("  data/ - JSON and numpy arrays")

        except Exception as e:
            print(f"\nERROR: {e}")
            traceback.print_exc()
            save_json(
                {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                run_dir / "error.json",
            )
            return 1

    P.report()
    return 0
