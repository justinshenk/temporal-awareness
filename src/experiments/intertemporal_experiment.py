"""
Intertemporal preference experiment module.

Provides configs and the main experiment runner for analyzing
temporal preferences in language models.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import numpy as np

from ..common.device import get_device
from ..common.io import ensure_dir, save_json, get_timestamp
from ..common.paths import get_pref_dataset_dir, get_experiment_dir
from ..common.token_positions import build_position_labels
from ..data import (
    generate_preference_data,
    PreferenceDataset,
    load_and_merge_preference_data,
)
from ..models import ModelRunner
from ..viz import plot_layer_position_heatmap, plot_position_sweep
from ..profiler import P, profile_fn

from .activation_patching import (
    run_activation_patching,
    PatchingResult,
    compute_layer_indices,
    find_best_position_layer,
    compute_pair_label_probs,
)
from .attribution_patching import run_attribution_patching
from .steering import compute_steering_vector, apply_steering
from .probe_training import run_probe_training
from ..common.positions_schema import PositionSpec
from ..prompt_datasets import PromptDatasetConfig


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Configuration for intertemporal experiments."""

    # Important
    model: str
    dataset_config: dict

    # Data generation
    max_samples: Optional[int] = 50

    # Patching
    max_pairs: int = 3
    ig_steps: int = 10
    position_threshold: float = 0.05

    # Activation patching
    act_patch_n_layers_sample: int = 12
    act_patch_full_sweep_components: list[str] = field(
        default_factory=lambda: ["resid_post", "attn_out", "mlp_out"]
    )
    act_patch_position_step: int = 1
    act_patch_token_positions: Optional[list] = None

    # Contrastive
    contrastive_max_samples: int = 500
    top_n_positions: int = 1

    # Steering evaluation
    steering_strengths: list[float] = field(
        default_factory=lambda: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    )
    steering_eval_max_samples: int = 10

    # Probe training
    probe_layers: Optional[list[int]] = None
    probe_positions: Optional[list] = None
    probe_max_samples: int = 200

    # Execution control
    skip: list[str] = field(default_factory=list)
    internals: Optional[dict] = None

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentConfig:
        """Create from a dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @property
    def name(self) -> str:
        """Human-readable name derived from dataset_config."""
        return self.dataset_config.get("name", "experiment")

    def get_preference_dataset_prefix(self) -> str:
        """Get the prefix for preference dataset files: {dataset_id}_{model_name}."""
        dataset_cfg = PromptDatasetConfig.load_from_dict(self.dataset_config)
        return PreferenceDataset.make_prefix(dataset_cfg.get_id(), self.model)


@dataclass
class ExperimentResults:
    """Results from a full experiment run."""

    pref_data: PreferenceDataset
    runner: ModelRunner
    output_dir: Path

    # Patching results
    position_sweep: Optional[np.ndarray] = None
    activation_patching: Optional[np.ndarray] = None
    attribution_results: Optional[dict[str, np.ndarray]] = None

    # Positions
    top_positions: list[PositionSpec] = field(default_factory=list)

    # Steering vectors
    steering_vectors: dict = field(default_factory=dict)


@dataclass
class ExperimentContext:
    """Shared state passed between experiment steps."""

    config: ExperimentConfig
    pref_data: PreferenceDataset
    runner: ModelRunner

    # Output
    output_dir: Path = get_experiment_dir()
    timestamp: Optional[str] = None

    @property
    def ts(self) -> Path:
        if not self.timestamp:
            self.timestamp = get_timestamp()
        return self.timestamp

    @property
    def viz_dir(self) -> Path:
        viz_dir = self.output_dir / "viz"
        ensure_dir(viz_dir)
        return viz_dir

    @property
    def data_dir(self) -> Path:
        data_dir = self.output_dir / "data"
        ensure_dir(data_dir)
        return data_dir

    @property
    def viz_dir(self) -> Path:
        viz_dir = self.output_dir / "viz"
        ensure_dir(viz_dir)
        return viz_dir

    @property
    def run_dir(self) -> Path:
        run_dir = self.output_dir / self.ts
        ensure_dir(run_dir)
        return run_dir


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------


@profile_fn("step_load_model")
def step_load_model(config: ExperimentConfig) -> ModelRunner:
    runner = ModelRunner(config.model, device=get_device())
    return runner


@profile_fn("step_preference_data")
def step_preference_data(config: ExperimentConfig) -> PreferenceDataset:
    # Check if preference data already exists
    pref_dataset_prefix = config.get_preference_dataset_prefix()

    if pref_data := load_and_merge_preference_data(pref_dataset_prefix, get_pref_dataset_dir()):
        pass  # Loaded and merged existing data
    else:
        with P("generate_data"):
            pref_data = generate_preference_data(
                model=config.model,
                dataset_config=config.dataset_config,
                internals=config.internals,
                max_samples=config.max_samples,
            )

    pref_data.save_as_json(get_experiment_dir() / "preference_data.json")

    print(f"Model: {pref_data.model}")
    print(f"Samples: {len(pref_data.preferences)}")
    return pref_data


@profile_fn("step_activation_patching")
def step_activation_patching(ctx: ExperimentContext) -> PatchingResult:
    config = ctx.config

    act_n_layers = config.act_patch_n_layers_sample
    act_pos_sweep_component = "resid_post"
    act_full_sweep_components = config.act_patch_full_sweep_components
    act_pos_step = config.act_patch_position_step
    max_pairs = config.max_pairs
    position_threshold = config.position_threshold
    act_token_positions = config.act_patch_token_positions

    with P("activation_patching"):
        (
            pos_sweep,
            full_sweeps,
            filtered_pos,
            token_labels,
            section_markers,
            pair_metadata,
        ) = run_activation_patching(
            ctx.runner,
            ctx.pref_data,
            max_pairs=config.max_pairs,
            threshold=config.position_threshold,
            position_sweep_component=act_pos_sweep_component,
            full_sweep_components=act_full_sweep_components,
            n_layers_sample=act_n_layers,
            position_step=act_pos_step,
            token_positions=act_token_positions,
        )

    layer_indices = compute_layer_indices(act_n_layers, ctx.runner.n_layers)

    primary_component = act_full_sweep_components[0]
    best_pos_idx, best_layer, best_token = find_best_position_layer(
        pos_sweep,
        full_sweeps,
        filtered_pos,
        token_labels,
        layer_indices,
        primary_component,
    )

    print(f"Position sweep: max={pos_sweep.max():.3f}, argmax={best_pos_idx}")
    print(
        f"Filtered positions: {len(filtered_pos)} (threshold={config.position_threshold})"
    )
    print(f"Components: {act_full_sweep_components}")
    for comp, sweep in full_sweeps.items():
        print(f"  {comp} sweep shape: {sweep.shape}")
    print(f"Layers sampled: {act_n_layers}, Position step: {act_pos_step}")
    print(f"Best position: {best_pos_idx} ({best_token})")
    print(f"Best layer: {best_layer}")

    compute_pair_label_probs(ctx.runner, ctx.pref_data, pair_metadata, max_pairs)

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
        token_labels[i] if i < len(token_labels) else f"pos{i}" for i in filtered_pos
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

    return PatchingResult(
        best_layer=best_layer,
        best_position=best_pos_idx,
        best_token=best_token,
        position_sweep=pos_sweep,
        full_sweeps=full_sweeps,
        filtered_positions=filtered_pos,
        layer_indices=layer_indices,
    )


@profile_fn("step_attribution_patching")
def step_attribution_patching(ctx: ExperimentContext) -> dict:
    config = ctx.config

    with P("attribution_patching"):
        attribution_results, attr_labels, attr_markers = run_attribution_patching(
            ctx.runner,
            ctx.pref_data,
            max_pairs=config.max_pairs,
            ig_steps=config.ig_steps,
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
    return attribution_results


@profile_fn("step_contrastive_steering")
def step_contrastive_steering(
    ctx: ExperimentContext,
    patching: PatchingResult,
) -> dict:
    """Step 5: Compute steering vector from contrastive activations."""
    with P("contrastive_steering"):
        direction, stats = compute_steering_vector(
            ctx.runner,
            ctx.pref_data,
            layer=patching.best_layer,
            position=patching.best_position,
            max_samples=ctx.config.contrastive_max_samples,
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

    return {
        "name": "contrastive",
        "direction": direction,
        "layer": patching.best_layer,
        "metadata": stats,
    }


@profile_fn("step_steering_eval")
def step_steering_eval(
    ctx: ExperimentContext,
    directions: list[dict],
) -> dict:
    """Step 7: Evaluate all steering directions on preference data prompts."""
    config = ctx.config

    max_eval = config.steering_eval_max_samples
    strengths = config.steering_strengths
    eval_prefs = ctx.pref_data.preferences[:max_eval]

    dir_names = [d["name"] for d in directions]
    print(f"Directions: {dir_names}")
    print(f"Evaluating {len(eval_prefs)} prompts x {len(strengths)} strengths each")

    all_eval_results = {}
    with P("steering_eval"):
        for sd in directions:
            print(f"\n--- {sd['name']} (layer {sd['layer']}) ---")
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
                            sd["direction"],
                            layer=sd["layer"],
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

            all_eval_results[sd["name"]] = eval_results
            save_json(
                eval_results,
                ctx.data_dir / f"steering_eval_{sd['name']}.json",
            )
            sd_name = sd["name"]
            print(f"\n  Saved: {ctx.data_dir / f'steering_eval_{sd_name}.json'}")

    return all_eval_results


@profile_fn("step_probe_training")
def step_probe_training(
    ctx: ExperimentContext,
) -> tuple[dict, Optional[dict]]:
    """Step 6: Train linear probes and extract probe steering vector."""
    config = ctx.config

    probe_layers = config.probe_layers
    if probe_layers is None:
        n = ctx.runner.n_layers
        probe_layers = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))

    probe_positions = config.probe_positions
    probe_max_samples = config.probe_max_samples

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

    # Extract best choice probe's direction as a steering dict
    probe_steering = None
    choice_results = probe_results.get("choice", [])
    if choice_results:
        best_choice = max(choice_results, key=lambda r: r.test_accuracy)
        best_choice_probe = probes.get(
            ("choice", best_choice.layer, best_choice.token_position)
        )
        if best_choice_probe:
            probe_steering = {
                "name": "probe_choice",
                "direction": best_choice_probe.get_steering_vector(),
                "layer": best_choice.layer,
                "metadata": {
                    "test_accuracy": best_choice.test_accuracy,
                    "train_accuracy": best_choice.train_accuracy,
                    "token_position": best_choice.token_position,
                },
            }
            print(
                f"  Probe steering: layer {probe_steering['layer']}, "
                f"accuracy {best_choice.test_accuracy:.3f}"
            )

    return probe_results, probe_steering


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


@profile_fn("run_experiment")
def run_experiment(config: ExperimentConfig) -> None:
    """Run the full intertemporal preference experiment."""

    # Step 1: Load Model
    runner = step_load_model(config)

    # Step 2: Get Preference Dataset (Prompt Dataset + Model's responses)
    pref_data = step_preference_data(config)

    return 0

    ctx = ExperimentContext(
        config=config,
        pref_data=pref_data,
        runner=runner,
    )

    patching = None
    if "activation_patching" not in config.skip:
        patching = step_activation_patching(ctx)

    if "attribution_patching" not in config.skip:
        step_attribution_patching(ctx)

    # Compute steering vectors from available sources
    contrastive_sd = None
    if "contrastive_steering" not in config.skip and patching is not None:
        contrastive_sd = step_contrastive_steering(ctx, patching)

    probe_sd = None
    if "probe_training" not in config.skip:
        _, probe_sd = step_probe_training(ctx)

    # Evaluate all available steering directions
    directions = []
    if contrastive_sd is not None:
        directions.append(contrastive_sd)
    if probe_sd is not None:
        directions.append(probe_sd)
    if directions and "steering_eval" not in config.skip:
        step_steering_eval(ctx, directions)

    # Save final metadata
    results_meta = {}
    if patching is not None:
        results_meta.update(
            {
                "best_position": patching.best_position,
                "best_layer": patching.best_layer,
                "position_token": patching.best_token,
                "position_sweep_max": float(patching.position_sweep.max()),
                "n_filtered_positions": len(patching.filtered_positions),
            }
        )
    if contrastive_sd is not None:
        results_meta["contrastive_norm"] = contrastive_sd["metadata"]["direction_norm"]
    results_meta["steering_directions"] = [d["name"] for d in directions]

    return 0
