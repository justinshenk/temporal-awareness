"""Pipeline orchestration: iterative training loop with crash recovery.

This pipeline trains SAEs on position-specific activations from LLM responses
to temporal preference questions.

Key changes from previous sentence-based approach:
- Extracts activations at specific token positions (source, dest, secondary_source)
- Supports multiple components (resid_pre, resid_post, mlp_out, attn_out)
- Organizes SAEs by (layer, component, position) tuples
"""

import copy
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.common import TimeValue
from src.common.device_utils import clear_gpu_memory, log_memory, check_memory_trend
from src.common.profiler import P

from .pipeline_state import PipelineStage, PipelineState
from .sae_paths import (
    ensure_dirs,
    reset_and_get_test_filepath_cfg,
    reset_and_get_special_filepath_cfg,
)
from .scenario_generator import generate_samples
from .sae_activations import horizon_bucket
from .sae_inference import (
    generate_and_extract,
    PositionActivations,
    form_training_data,
)
from .sae_analysis import (
    SAE,
    SAESpec,
    initialize_sae_models,
    load_sae_models,
    save_sae_model,
    train_sae,
)
from .sae_evaluation import cluster_analysis, baseline_cluster_analysis


# =============================================================================
# Path Helpers
# =============================================================================


def get_state_filepath(state: PipelineState) -> str:
    return str(state.filepath_cfg.data_dir / f"state_{state.pipeline_id}.json")


def get_samples_filepath(state: PipelineState) -> str:
    return str(
        state.filepath_cfg.data_dir
        / f"samples_{state.pipeline_id}_iter{state.iteration}.json"
    )


def get_activations_filepath(state: PipelineState, sample_idx: int) -> str:
    return str(
        state.filepath_cfg.data_dir
        / f"activations_{state.pipeline_id}_iter{state.iteration}_sample{sample_idx}.npz"
    )


def get_sae_dirpath(state: PipelineState) -> str:
    return str(state.filepath_cfg.sae_dir / state.pipeline_id)


def get_analysis_dirpath(state: PipelineState) -> str:
    return str(
        state.filepath_cfg.analysis_dir
        / f"state_{state.pipeline_id}_iter{state.iteration}"
    )


# =============================================================================
# State & Data I/O
# =============================================================================


def save_state(state: PipelineState, stage: PipelineStage | None) -> None:
    if stage:
        state.stage = stage
    state.save(get_state_filepath(state))


def save_samples(
    state: PipelineState,
    samples: list,
    activations: list[PositionActivations | None] | None = None,
) -> None:
    """Save samples and their position-based activations."""
    if activations:
        for sample_idx, (sample, act) in enumerate(zip(samples, activations)):
            if act is not None:
                path = get_activations_filepath(state, sample_idx)
                # Save activations as npz with position metadata
                np.savez(
                    path,
                    positions=json.dumps(act.positions.to_dict()),
                    **act.activations,
                )
                sample["activation_path"] = path

    path = get_samples_filepath(state)
    with open(path, "w") as f:
        json.dump({"samples": samples}, f, indent=2)
    state.samples_path = path


def load_samples(state: PipelineState) -> list:
    path = get_samples_filepath(state)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)["samples"]


def load_activations(samples: list) -> list[PositionActivations | None]:
    """Load position-based activations for samples."""
    from .sae_positions import ResolvedPositions

    result = []
    for sample in samples:
        act_path = sample.get("activation_path")
        if not act_path or not os.path.exists(act_path):
            result.append(None)
            continue

        with np.load(act_path, allow_pickle=True) as data:
            # Load position metadata
            positions_dict = json.loads(str(data["positions"]))
            positions = ResolvedPositions(**positions_dict)

            # Load all activation arrays
            activations = {
                k: data[k] for k in data.files if k != "positions"
            }

            result.append(PositionActivations(positions=positions, activations=activations))

    return result


# =============================================================================
# Sample Processing
# =============================================================================


def enrich_sample(sample: dict) -> dict:
    """Add derived fields to a PromptSample dict for analysis."""
    prompt = sample["prompt"]
    pair = prompt["preference_pair"]
    th = TimeValue.parse(prompt["time_horizon"]) if prompt["time_horizon"] else None

    sample["time_horizon_bucket"] = horizon_bucket(th)
    sample["time_horizon_months"] = th.to_months() if th else None
    sample["short_term_label"] = pair["short_term"]["label"]
    sample["long_term_label"] = pair["long_term"]["label"]
    sample["short_term_time_months"] = TimeValue.parse(
        pair["short_term"]["time"]
    ).to_months()
    sample["long_term_time_months"] = TimeValue.parse(
        pair["long_term"]["time"]
    ).to_months()
    sample["prompt_text"] = prompt["text"]
    return sample


# =============================================================================
# Pipeline Stages
# =============================================================================


def stage_generate_dataset(state: PipelineState) -> None:
    """Stage 1: Generate or load samples."""
    iter_seed = state.config.seed + state.iteration * 10000
    cache_path = state.filepath_cfg.data_dir / "all_generated_samples.json"

    with P("generate_samples"):
        if cache_path.exists():
            with open(cache_path) as f:
                samples = json.load(f)["samples"]
        else:
            raw = generate_samples()
            samples = [asdict(s) for s in raw]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"samples": samples}, f)

    with P("subsample"):
        rng = np.random.RandomState(iter_seed)
        n = min(state.config.samples_per_iter, len(samples))
        indices = rng.choice(len(samples), size=n, replace=False)
        samples = [enrich_sample(samples[i]) for i in sorted(indices)]

    save_samples(state, samples)
    save_state(state, PipelineStage.DATASET_GENERATED)


def stage_inference(state: PipelineState) -> None:
    """Stage 2: Run inference to get responses and position-specific activations."""
    samples = load_samples(state)

    with P("generate_and_extract"):
        updated, activations = generate_and_extract(
            samples=samples,
            model_name=state.config.model,
            max_new_tokens=state.config.max_new_tokens,
            layers=state.config.layers,
            components=state.config.components,
            position_names=state.config.position_names,
        )

    save_samples(state, updated, activations)
    save_state(state, PipelineStage.INFERENCE_DONE)


def stage_train_saes(
    state: PipelineState,
    samples: list,
    activations: list[PositionActivations | None],
    sae_models: list[SAE | SAESpec],
    tb_writer=None,
) -> list[SAE]:
    """Stage 3: Train SAEs on position-specific activations."""
    sae_dir = get_sae_dirpath(state)
    global_step = state.iteration * state.config.max_epochs

    # Group models by their target key (layer, component, position)
    models_by_target: dict[str, list[SAE | SAESpec]] = {}
    for sae in sae_models:
        key = sae.get_target_key()
        if key not in models_by_target:
            models_by_target[key] = []
        models_by_target[key].append(sae)

    trained = []
    results = []

    for target_key, target_models in models_by_target.items():
        # Parse target key to get layer, component, position
        # Format: L{layer}_{component}_P{position_name}
        parts = target_key.split("_")
        layer = int(parts[0][1:])  # Remove 'L' prefix
        component = "_".join(parts[1:-1])  # Handle components with underscores
        # Handle the P prefix in position name
        pos_part = parts[-1]
        if pos_part.startswith("P"):
            position_name = pos_part[1:]  # Remove 'P' prefix
        else:
            # Reconstruct if split incorrectly
            position_name = target_key.split("_P")[-1]
            component = target_key.split("_P")[0].split("_", 1)[-1]

        # Form training data for this target
        try:
            with P(f"form_training_data_{target_key}"):
                X, _ = form_training_data(activations, layer, component, position_name)
        except ValueError as e:
            print(f"  Skipping {target_key}: {e}")
            continue

        print(f"  Training {len(target_models)} SAEs for {target_key} (n={X.shape[0]})...")

        for sae in target_models:
            with P("train_sae"):
                trained_sae, result = train_sae(
                    x_norm=X,
                    sae=sae,
                    batch_size=state.config.batch_size,
                    max_epochs=state.config.max_epochs,
                    patience=state.config.patience,
                    tb_writer=tb_writer,
                    tb_prefix="",
                    tb_global_step=global_step,
                )
            trained.append(trained_sae)
            results.append(result)
            save_sae_model(sae_dir, trained_sae)

    state.sae_results = results
    save_state(state, PipelineStage.SAE_TRAINED)
    return trained


def stage_analyze(
    state: PipelineState,
    samples: list,
    activations: list[PositionActivations | None],
    sae_models: list[SAE],
) -> None:
    """Stage 4: Analyze SAE features."""
    analysis_dir = get_analysis_dirpath(state)

    results = []
    for sae in sae_models:
        sae_dir = os.path.join(analysis_dir, sae.get_name())
        os.makedirs(sae_dir, exist_ok=True)

        # Get training data for this SAE's target
        try:
            X, indices = form_training_data(
                activations, sae.layer, sae.component, sae.position_name
            )
        except ValueError:
            continue

        # Build sentence-like dicts for cluster_analysis with all available fields
        filtered = [
            {
                "sample_idx": samples[i].get("sample_idx"),
                "time_horizon_bucket": samples[i].get("time_horizon_bucket", -1),
                "time_horizon_months": samples[i].get("time_horizon_months"),
                "llm_choice": samples[i].get("llm_choice", -1),
                "llm_choice_time_months": samples[i].get("llm_choice_time_months"),
                "formatting_id": samples[i].get("formatting_id"),
                "short_term_first": samples[i].get("short_term_first"),
                "short_term_label": samples[i].get("short_term_label"),
                "long_term_label": samples[i].get("long_term_label"),
                "short_term_time_months": samples[i].get("short_term_time_months"),
                "long_term_time_months": samples[i].get("long_term_time_months"),
                "activations": {sae.get_target_key(): X[j]},
            }
            for j, i in enumerate(indices)
        ]

        # Get SAE features
        import torch
        from ...common.device_utils import get_device

        device = get_device()
        X_tensor = torch.from_numpy(X).float().to(device)
        sae_model = sae.to(device)
        with torch.no_grad():
            features = sae_model.get_all_activations(X_tensor)

        with P("cluster_analysis"):
            result = {"cluster": cluster_analysis(filtered, features, sae_dir)}
        results.append(result)

    state.analysis_results = results
    save_state(state, PipelineStage.EVALUATED)


# =============================================================================
# Iteration Runner
# =============================================================================


def run_iteration(
    state: PipelineState, tb_writer=None, skip_analysis: bool = False
) -> None:
    """Run one complete iteration."""
    ensure_dirs(state.filepath_cfg)
    log_memory("iter_start", state.iteration)

    if state.stage < PipelineStage.DATASET_GENERATED:
        stage_generate_dataset(state)
        log_memory("after_dataset", state.iteration)

    if state.stage < PipelineStage.INFERENCE_DONE:
        stage_inference(state)
        log_memory("after_inference", state.iteration)

    samples = load_samples(state)
    activations = load_activations(samples)

    sae_models = (
        load_sae_models(state, get_sae_dirpath(state))
        if state.iteration
        else initialize_sae_models(state)
    )
    log_memory("after_load", state.iteration)

    if state.stage < PipelineStage.SAE_TRAINED:
        sae_models = stage_train_saes(state, samples, activations, sae_models, tb_writer)
        log_memory("after_train", state.iteration)

    if not skip_analysis and state.stage < PipelineStage.EVALUATED:
        stage_analyze(state, samples, activations, sae_models)
        log_memory("after_analysis", state.iteration)

    del activations
    clear_gpu_memory()


# =============================================================================
# Special Iteration (Full Retrain)
# =============================================================================


def load_subsampled_data(
    state: PipelineState, max_samples: int = 4096
) -> tuple[list, list[PositionActivations | None]]:
    """Load random subsample of accumulated data."""
    pattern = f"samples_{state.pipeline_id}_iter*.json"
    sample_files = sorted(state.filepath_cfg.data_dir.glob(pattern))

    all_samples, all_acts = [], []
    for sf in sample_files:
        with open(sf) as f:
            samples = json.load(f)["samples"]
        acts = load_activations(samples)
        all_samples.extend(samples)
        all_acts.extend(acts)
        if len(all_samples) >= max_samples * 2:
            break

    if len(all_samples) > max_samples:
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(len(all_samples), size=max_samples, replace=False))
        all_samples = [all_samples[i] for i in idx]
        all_acts = [all_acts[i] for i in idx]

    return all_samples, all_acts


def run_special_iteration(main_state: PipelineState) -> None:
    """Retrain SAEs from scratch on accumulated data."""
    from torch.utils.tensorboard import SummaryWriter

    state = copy.deepcopy(main_state)
    state.filepath_cfg = reset_and_get_special_filepath_cfg()
    ensure_dirs(state.filepath_cfg)
    state.config.max_epochs = 300
    state.config.patience = 10

    tb_writer = SummaryWriter(
        log_dir=str(state.filepath_cfg.tensorboard_dir / state.pipeline_id)
    )
    try:
        samples, activations = load_subsampled_data(state)
        sae_models = initialize_sae_models(state)
        sae_models = stage_train_saes(state, samples, activations, sae_models, tb_writer)
        stage_analyze(state, samples, activations, sae_models)
    finally:
        tb_writer.close()
        clear_gpu_memory()


# =============================================================================
# Main Entry Points
# =============================================================================


def run_pipeline(state: PipelineState, retrain_every_n_iter: int = 50) -> None:
    """Run the iterative pipeline."""
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(
        log_dir=str(state.filepath_cfg.tensorboard_dir / state.pipeline_id)
    )

    try:
        for i in range(state.iteration, state.config.max_iterations):
            print(
                f"\n{'=' * 60}\nITERATION {i}/{state.config.max_iterations}\n{'=' * 60}"
            )
            state.iteration = i
            run_iteration(state, tb_writer=writer, skip_analysis=True)
            state.stage = PipelineStage.INIT
            P.report()

            if i > 0 and i % 10 == 0:
                check_memory_trend()
            if i > 0 and i % retrain_every_n_iter == 0:
                run_special_iteration(state)
    finally:
        writer.close()
        check_memory_trend()
        P.report()


def run_test_iteration(state: PipelineState) -> None:
    """Run a minimal test iteration."""
    from torch.utils.tensorboard import SummaryWriter

    state.filepath_cfg = reset_and_get_test_filepath_cfg()
    state.config.samples_per_iter = 8
    state.config.batch_size = 4
    state.config.max_iterations = 1
    state.config.max_epochs = 1
    state.config.patience = 1
    # Use only high-priority targets for testing
    state.config.layers = [21, 31]
    state.config.components = ["resid_post", "mlp_out"]
    state.config.position_names = ["dest"]
    state.iteration += 1
    state.stage = PipelineStage.INIT

    writer = SummaryWriter(
        log_dir=str(state.filepath_cfg.tensorboard_dir / state.pipeline_id)
    )
    try:
        run_iteration(state, tb_writer=writer)
    finally:
        writer.close()
        P.report()
