"""Pipeline orchestration: iterative training loop with crash recovery."""

import copy
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.common import TimeValue
from src.common.device_utils import clear_gpu_memory, log_memory, check_memory_trend
from src.common.profiler import P

from .pipeline_state import PipelineStage, PipelineState
from .sae_paths import ensure_dirs, reset_and_get_test_filepath_cfg, reset_and_get_special_filepath_cfg
from .scenario_generator import generate_samples
from .sae_activations import Sentence, horizon_bucket, get_sentences, form_training_datasets, get_normalized_vectors_for_sentences
from .sae_inference import generate_and_extract
from .sae import SAE, initialize_sae_models, load_sae_models, save_sae_model, train_sae, get_sae_features_for_sentences
from .sae_evaluation import cluster_analysis, baseline_cluster_analysis


# =============================================================================
# Path Helpers
# =============================================================================


def get_state_filepath(state: PipelineState) -> str:
    return str(state.filepath_cfg.data_dir / f"state_{state.pipeline_id}.json")


def get_samples_filepath(state: PipelineState) -> str:
    return str(state.filepath_cfg.data_dir / f"samples_{state.pipeline_id}_iter{state.iteration}.json")


def get_activations_filepath(state: PipelineState, sample_idx: int, sentence_idx: int) -> str:
    return str(state.filepath_cfg.data_dir / f"activations_{state.pipeline_id}_iter{state.iteration}_sample{sample_idx}_sentence{sentence_idx}.npz")


def get_sae_dirpath(state: PipelineState) -> str:
    return str(state.filepath_cfg.sae_dir / state.pipeline_id)


def get_analysis_dirpath(state: PipelineState) -> str:
    return str(state.filepath_cfg.analysis_dir / f"state_{state.pipeline_id}_iter{state.iteration}")


def get_section_means_filepath(state: PipelineState) -> str:
    return str(state.filepath_cfg.data_dir / f"section_means_{state.pipeline_id}.npz")


# =============================================================================
# Filter
# =============================================================================


def filter_sentence(sentence: Sentence) -> bool:
    return sentence.source == "response" and sentence.section == "choice"


# =============================================================================
# State & Data I/O
# =============================================================================


def save_state(state: PipelineState, stage: PipelineStage | None) -> None:
    if stage:
        state.stage = stage
    state.save(get_state_filepath(state))


def save_samples(state: PipelineState, samples: list, activations: list | None = None) -> None:
    if activations:
        for sample_idx, sample in enumerate(samples):
            paths = []
            for sentence_idx, sentence_acts in activations[sample_idx].items():
                path = get_activations_filepath(state, sample_idx, sentence_idx)
                np.savez(path, **sentence_acts)
                paths.append(path)
            sample["activation_paths"] = paths

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


def load_activations(samples: list) -> list | None:
    if not samples or not samples[0].get("activation_paths"):
        return None
    activations = []
    for sample in samples:
        sample_acts = {}
        for sentence_idx, path in enumerate(sample["activation_paths"]):
            with np.load(path) as data:
                sample_acts[sentence_idx] = {k: data[k] for k in data.files}
        activations.append(sample_acts)
    return activations


# =============================================================================
# Section Activation Means
# =============================================================================


def compute_section_means_streaming(state: PipelineState) -> dict[int, dict[str, np.ndarray]]:
    """Compute activation means by streaming through files to avoid OOM."""
    sections = Sentence.get_sections()
    layers = state.config.layers

    sums: dict[int, dict[str, np.ndarray | None]] = {l: {s: None for s in sections} for l in layers}
    counts: dict[int, dict[str, int]] = {l: {s: 0 for s in sections} for l in layers}

    pattern = f"samples_{state.pipeline_id}_iter*.json"
    sample_files = sorted(state.filepath_cfg.data_dir.glob(pattern))

    for sf in sample_files:
        with open(sf) as f:
            samples = json.load(f)["samples"]
        acts = load_activations(samples)
        if not acts:
            continue

        for sample_idx, sample in enumerate(samples):
            if sample_idx >= len(acts):
                continue
            sample_acts = acts[sample_idx]
            sentences = sample.get("sentences", [])

            for sent_idx in sorted(sample_acts.keys(), key=int):
                if sent_idx >= len(sentences):
                    continue
                section = sentences[sent_idx].get("section")
                if section not in sections:
                    continue

                for layer in layers:
                    key = f"layer_{layer}"
                    if key in sample_acts[sent_idx]:
                        vec = sample_acts[sent_idx][key]
                        if sums[layer][section] is None:
                            sums[layer][section] = vec.copy()
                        else:
                            sums[layer][section] += vec
                        counts[layer][section] += 1

    # Find d_in from first available vector
    d_in = next(
        (sums[l][s].shape[0] for l in layers for s in sections if sums[l][s] is not None),
        None
    )
    if d_in is None:
        raise ValueError("No activation vectors found")

    result: dict[int, dict[str, np.ndarray]] = {}
    for layer in layers:
        result[layer] = {}
        for s in sections:
            if sums[layer][s] is not None and counts[layer][s] > 0:
                result[layer][s] = sums[layer][s] / counts[layer][s]
            else:
                result[layer][s] = np.zeros(d_in)

    return result


def save_section_means(state: PipelineState, means: dict[int, dict[str, np.ndarray]]) -> None:
    path = get_section_means_filepath(state)
    arrays = {f"layer_{l}_{s}": arr for l, sects in means.items() for s, arr in sects.items()}
    np.savez(path, **arrays)
    state.section_means_path = path


def load_section_means(state: PipelineState) -> dict[int, dict[str, np.ndarray]] | None:
    if not state.section_means_path or not os.path.exists(state.section_means_path):
        return None

    sections = Sentence.get_sections()
    layers = state.config.layers

    with np.load(state.section_means_path) as data:
        d_in = next((data[k].shape[0] for k in data.files), 1)
        result: dict[int, dict[str, np.ndarray]] = {}
        for layer in layers:
            result[layer] = {}
            for section in sections:
                key = f"layer_{layer}_{section}"
                result[layer][section] = data[key] if key in data else np.zeros(d_in)
    return result


def get_section_means(state: PipelineState) -> dict[int, dict[str, np.ndarray]]:
    cached = load_section_means(state)
    if cached:
        return cached
    means = compute_section_means_streaming(state)
    save_section_means(state, means)
    save_state(state, None)
    return means


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
    sample["short_term_time_months"] = TimeValue.parse(pair["short_term"]["time"]).to_months()
    sample["long_term_time_months"] = TimeValue.parse(pair["long_term"]["time"]).to_months()
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
    """Stage 2: Run inference to get responses and activations."""
    samples = load_samples(state)

    with P("generate_and_extract"):
        updated, activations = generate_and_extract(
            samples=samples,
            model_name=state.config.model,
            max_new_tokens=state.config.max_new_tokens,
        )

    save_samples(state, updated, activations)
    save_state(state, PipelineStage.INFERENCE_DONE)


def stage_train_saes(
    state: PipelineState,
    samples: list,
    activations: list,
    sae_models: list[SAE],
    section_means: dict,
    tb_writer=None,
) -> list[SAE]:
    """Stage 3: Train SAEs on activations."""
    sae_dir = get_sae_dirpath(state)
    global_step = state.iteration * state.config.max_epochs

    with P("form_training_datasets"):
        x_by_layer = {
            layer: form_training_datasets(samples, activations, layer, section_means, filter_sentence)
            for layer in state.config.layers
        }

    trained = []
    results = []
    for sae in sae_models:
        with P("train_sae"):
            sae, result = train_sae(
                x_norm=x_by_layer[sae.layer],
                sae=sae,
                batch_size=state.config.batch_size,
                max_epochs=state.config.max_epochs,
                patience=state.config.patience,
                tb_writer=tb_writer,
                tb_prefix="",
                tb_global_step=global_step,
            )
        trained.append(sae)
        results.append(result)
        save_sae_model(sae_dir, sae)

    state.sae_results = results
    save_state(state, PipelineStage.SAE_TRAINED)
    return trained


def stage_analyze(
    state: PipelineState,
    samples: list,
    activations: list,
    sae_models: list[SAE],
    section_means: dict,
) -> None:
    """Stage 4: Analyze SAE features and baseline clustering."""
    analysis_dir = get_analysis_dirpath(state)
    sentences = get_sentences(samples, activations, section_means)

    results = []
    for sae in sae_models:
        sae_dir = os.path.join(analysis_dir, sae.get_name())
        with P("get_sae_features"):
            features, filtered = get_sae_features_for_sentences(sae, sentences, filter_sentence)
        with P("cluster_analysis"):
            result = {"cluster": cluster_analysis(filtered, features, sae_dir)}
        results.append(result)

    # Baseline clustering per unique (layer, n_clusters)
    baseline_dir = os.path.join(analysis_dir, "cluster_baseline")
    seen = set()
    baselines = {}
    for sae in sae_models:
        key = (sae.layer, sae.num_latents)
        if key in seen:
            continue
        seen.add(key)
        X, filtered = get_normalized_vectors_for_sentences(sae.layer, sentences, filter_sentence)
        config_key = f"L{sae.layer}_k{sae.num_latents}"
        with P("baseline_cluster_analysis"):
            baselines[config_key] = baseline_cluster_analysis(X, filtered, sae.num_latents, os.path.join(baseline_dir, config_key))

    for sae, result in zip(sae_models, results):
        result["cluster_baseline"] = baselines.get(f"L{sae.layer}_k{sae.num_latents}", {})

    state.analysis_results = results
    save_state(state, PipelineStage.EVALUATED)


# =============================================================================
# Iteration Runner
# =============================================================================


def run_iteration(state: PipelineState, tb_writer=None, skip_analysis: bool = False) -> None:
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
    section_means = get_section_means(state)

    sae_models = (
        load_sae_models(state, get_sae_dirpath(state))
        if state.iteration else initialize_sae_models(state)
    )
    log_memory("after_load", state.iteration)

    if state.stage < PipelineStage.SAE_TRAINED:
        sae_models = stage_train_saes(state, samples, activations, sae_models, section_means, tb_writer)
        log_memory("after_train", state.iteration)

    if not skip_analysis and state.stage < PipelineStage.EVALUATED:
        stage_analyze(state, samples, activations, sae_models, section_means)
        log_memory("after_analysis", state.iteration)

    del activations
    clear_gpu_memory()


# =============================================================================
# Special Iteration (Full Retrain)
# =============================================================================


def load_subsampled_data(state: PipelineState, max_samples: int = 4096) -> tuple[list, list]:
    """Load random subsample of accumulated data."""
    pattern = f"samples_{state.pipeline_id}_iter*.json"
    sample_files = sorted(state.filepath_cfg.data_dir.glob(pattern))

    all_samples, all_acts = [], []
    for sf in sample_files:
        with open(sf) as f:
            samples = json.load(f)["samples"]
        acts = load_activations(samples)
        if not acts:
            continue
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
    # Recompute section means on all data
    means = compute_section_means_streaming(main_state)
    save_section_means(main_state, means)
    save_state(main_state, None)

    state = copy.deepcopy(main_state)
    state.filepath_cfg = reset_and_get_special_filepath_cfg()
    ensure_dirs(state.filepath_cfg)
    state.config.max_epochs = 300
    state.config.patience = 10

    tb_writer = SummaryWriter(log_dir=str(state.filepath_cfg.tensorboard_dir / state.pipeline_id))
    try:
        samples, activations = load_subsampled_data(state)
        sae_models = initialize_sae_models(state)
        sae_models = stage_train_saes(state, samples, activations, sae_models, means, tb_writer)
        stage_analyze(state, samples, activations, sae_models, means)
    finally:
        tb_writer.close()
        clear_gpu_memory()


# =============================================================================
# Main Entry Points
# =============================================================================


def run_pipeline(state: PipelineState, retrain_every_n_iter: int = 50) -> None:
    """Run the iterative pipeline."""
    writer = SummaryWriter(log_dir=str(state.filepath_cfg.tensorboard_dir / state.pipeline_id))

    try:
        for i in range(state.iteration, state.config.max_iterations):
            print(f"\n{'='*60}\nITERATION {i}/{state.config.max_iterations}\n{'='*60}")
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
    state.filepath_cfg = reset_and_get_test_filepath_cfg()
    state.config.samples_per_iter = 8
    state.config.batch_size = 4
    state.config.max_iterations = 1
    state.config.max_epochs = 1
    state.config.patience = 1
    state.iteration += 1
    state.stage = PipelineStage.INIT

    writer = SummaryWriter(log_dir=str(state.filepath_cfg.tensorboard_dir / state.pipeline_id))
    try:
        run_iteration(state, tb_writer=writer)
    finally:
        writer.close()
        P.report()
