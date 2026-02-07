"""Pipeline orchestration: iterative training loop with crash recovery."""

import json
import os
from dataclasses import asdict

import numpy as np
from src.common.types import TimeValue
from .utils import (
    clear_gpu_memory,
    ensure_dirs,
    reset_and_get_test_filepath_cfg,
    reset_and_get_special_filepath_cfg,
)
from .state import PipelineStage, PipelineState
from .data import generate_samples
from .activations import (
    Sentence,
    _horizon_bucket,
    calculate_activation_means_by_section,
    form_training_datasets,
    get_normalized_vectors_for_sentences,
    get_sentences,
)
from .inference import generate_and_extract
from .sae import (
    SAE,
    initialize_sae_models,
    load_sae_models,
    save_sae_model,
    train_sae,
    get_sae_features_for_sentences,
)
from .evaluate import (
    cluster_analysis,
    baseline_cluster_analysis,
)
from torch.utils.tensorboard import SummaryWriter
import copy

from src.profiler import P

# ── Path Helpers ─────────────────────────────


def get_state_filepath(state: PipelineState) -> str:
    return str(state.filepath_cfg.data_dir / f"state_{state.pipeline_id}.json")


def get_samples_filepath(state: PipelineState) -> str:
    return str(
        state.filepath_cfg.data_dir
        / f"samples_{state.pipeline_id}_iter{state.iteration}.json"
    )


def get_activations_filepath(
    state: PipelineState, sample_idx: int, sentence_idx: int
) -> str:
    return str(
        state.filepath_cfg.data_dir
        / f"activations_{state.pipeline_id}_iter{state.iteration}_sample{sample_idx}_sentence{sentence_idx}.npz"
    )


def get_sae_dirpath(state: PipelineState) -> str:
    return str(state.filepath_cfg.sae_dir / state.pipeline_id)


def get_analysis_dirpath(state: PipelineState) -> str:
    return str(
        state.filepath_cfg.analysis_dir
        / f"state_{state.pipeline_id}_iter{state.iteration}"
    )


def filter_sentence(sentence: Sentence) -> bool:
    return sentence.source == "response" and sentence.section == "choice"


# def filter_sentence(sentence: Sentence) -> bool:
#     return True


# ── Data Helpers ─────────────────────────────


def update_state(state: PipelineState, stage: PipelineStage | None) -> None:
    if stage:
        state.stage = stage
    state.save(get_state_filepath(state))
    print(f"\n\n\n\n\n    Saved at {stage}:\n\n{state}\n\n\n\n")


def update_samples(
    state: PipelineState, samples: list, activations: list | None = None
) -> None:
    if activations:
        print(f"  Saving activations for {len(samples)} samples")
        for sample_idx, sample in enumerate(samples):
            all_activation_filepaths = []
            for sentence_idx, sentence_activations in activations[sample_idx].items():
                activations_filepath = get_activations_filepath(
                    state, sample_idx, sentence_idx
                )
                # sentence_activations = {layer_idx: layer}
                np.savez(activations_filepath, **sentence_activations)
                all_activation_filepaths.append(activations_filepath)
            sample["activation_paths"] = all_activation_filepaths
        print(f"  Saved activations for {len(samples)} samples")

    if samples:
        samples_filepath = get_samples_filepath(state)
        with open(samples_filepath, "w") as f:
            json.dump({"samples": samples}, f, indent=4)
        state.samples_path = samples_filepath


def load_samples(state: PipelineState) -> list:
    samples = []

    samples_filepath = get_samples_filepath(state)
    if os.path.exists(samples_filepath):
        with open(samples_filepath, "r") as f:
            data = json.load(f)
            samples = data["samples"]
        state.samples_path = samples_filepath

    return samples


def load_activations(samples: list) -> list | None:
    activations = None
    if samples and samples[0].get("activation_paths"):
        print(f"  Loading activations for {len(samples)} samples")
        activations = []
        for sample_idx, sample in enumerate(samples):
            sample_activations = {}
            for sentence_idx, activations_filepath in enumerate(
                sample["activation_paths"]
            ):
                with np.load(activations_filepath) as data:
                    # sentence_activations = {layer_idx: layer}
                    sentence_activations = {key: data[key] for key in data.files}
                sample_activations[sentence_idx] = sentence_activations
            activations.append(sample_activations)
        print(f"  Loaded activations for {len(samples)} samples")

    return activations


# ── Load all accumulated data ────────────────────────────────────────────────


def _load_all_data(state: PipelineState) -> tuple[list, list]:
    """Load samples and activations from all completed iterations.

    Scans the data directory for all samples_{pipeline_id}_iter*.json files,
    loads each one together with its activations, and concatenates them.

    Returns (all_samples, all_activations).
    """
    with P("load_all_data"):
        data_dir = state.filepath_cfg.data_dir
        pattern = f"samples_{state.pipeline_id}_iter*.json"
        sample_files = sorted(data_dir.glob(pattern))

        all_samples = []
        all_activations = []

        for sf in sample_files:
            with open(sf) as f:
                data = json.load(f)
            samples = data["samples"]
            acts = load_activations(samples)
            if acts is None:
                continue
            all_samples.extend(samples)
            all_activations.extend(acts)

        n_samples = len(all_samples)
        n_iters = len(sample_files)
        print(f"  Loaded {n_samples} samples from {n_iters} iterations")
    return all_samples, all_activations


def _load_samples(state: PipelineState) -> list:
    """Loads generation (and maybe inference) results for the current iteration."""
    with P("load_samples"):
        samples = load_samples(state)
    return samples


def _load_activations(state: PipelineState, samples: list) -> list:
    """Loads inference result for the current iteration."""
    with P("load_activations"):
        activations = load_activations(samples)
    return activations


def _calculate_section_activation_means(state: PipelineState):
    with P("calculate_section_activation_means"):
        samples, activations = _load_all_data(state)
        activation_means = calculate_activation_means_by_section(
            samples, activations, state.config.layers
        )
    return activation_means


# ── Stage 1: Dataset generation (per iteration) ─────────────────────────────


def process_sample(sample: dict) -> dict:
    """Convert a serialized PromptSample dict to a flat JSON-serializable dict."""
    prompt = sample["prompt"]
    pair = prompt["preference_pair"]
    th_raw = prompt["time_horizon"]
    th = TimeValue.parse_time_value(th_raw) if th_raw else None
    st_time = TimeValue.parse_time_value(pair["short_term"]["time"])
    lt_time = TimeValue.parse_time_value(pair["long_term"]["time"])
    return {
        "sample_id": sample["sample_id"],
        "prompt_text": prompt["text"],
        "time_horizon_bucket": _horizon_bucket(th),
        "time_horizon_months": th.to_months() if th else None,
        "short_term_label": pair["short_term"]["label"],
        "long_term_label": pair["long_term"]["label"],
        "short_term_time_months": st_time.to_months(),
        "long_term_time_months": lt_time.to_months(),
    }


def _generate_dataset(state: PipelineState):
    config = state.config
    iteration = state.iteration

    # Different seed per iteration for diversity
    iter_seed = config.seed + iteration * 10000

    print(f"  Generating {config.samples_per_iter} samples (seed={iter_seed})...")

    with P("generate_samples"):
        cache_path = state.filepath_cfg.data_dir / "all_generated_samples.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                serialized_samples = json.load(f)["samples"]
                samples = serialized_samples
                print(json.dumps(samples[0], indent=4))
            print(f"  Loaded {len(samples)} cached samples from {cache_path}")
        else:
            samples = generate_samples()
            serialized_samples = [asdict(s) for s in samples]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"samples": serialized_samples}, f)
            print(f"  Saved {len(serialized_samples)} samples to {cache_path}")

    with P("subsample"):
        rng = np.random.RandomState(iter_seed)
        n = min(config.samples_per_iter, len(samples))
        indices = rng.choice(len(samples), size=n, replace=False)
        samples = [samples[i] for i in sorted(indices)]

    print(f"  Generated {len(samples)} samples")

    samples = [process_sample(s) for s in samples]
    with P("save_samples"):
        update_samples(state, samples)
        update_state(state, PipelineStage.DATASET_GENERATED)


# ── Stage 2: Combined inference (per iteration) ─────────────────────────────


def _run_inference(
    state: PipelineState,
) -> None:
    """Run inference for the current iteration."""

    samples = _load_samples(state)

    with P("generate_and_extract"):
        updated_samples, activations = generate_and_extract(
            samples=samples,
            model_name=state.config.model,
            max_new_tokens=state.config.max_new_tokens,
        )

    with P("save_inference"):
        update_samples(state, updated_samples, activations)
        update_state(state, PipelineStage.INFERENCE_DONE)


# ── Stage 3: SAE training ───────────────


def _get_sae_models(state: PipelineState):
    with P("get_sae_models"):
        if state.iteration:
            sae_dir = get_sae_dirpath(state)
            sae_models = load_sae_models(state, sae_dir)
        else:
            sae_models = initialize_sae_models(state)
    return sae_models


def _train_saes(
    state: PipelineState,
    samples: list,
    activations: list,
    sae_models: list,
    section_activation_means: dict,
    tb_writer=None,
) -> list[SAE]:
    """Train SAEs on new samples/activations, logging to TensorBoard."""
    sae_dir = get_sae_dirpath(state)

    # Use iteration as global step so TB loss curves are continuous across
    # iterations (each iteration = 1 gradient step in online-SGD mode).
    global_step = state.iteration * state.config.max_epochs

    normalized_x_by_layer = {}
    with P("form_training_datasets"):
        for layer in state.config.layers:
            normalized_x_by_layer[layer] = form_training_datasets(
                samples,
                activations,
                layer,
                section_activation_means,
                filter_sentence=filter_sentence,
            )

    all_sae = []
    all_results = []
    for sae in sae_models:
        with P("train_single_sae"):
            trained_sae, training_results = train_sae(
                x_norm=normalized_x_by_layer[sae.layer],
                sae=sae,
                batch_size=state.config.batch_size,
                max_epochs=state.config.max_epochs,
                patience=state.config.patience,
                tb_writer=tb_writer,
                tb_prefix="",
                tb_global_step=global_step,
            )
        all_sae.append(trained_sae)
        all_results.append(training_results)

        save_sae_model(sae_dir, trained_sae)

    state.sae_results = all_results
    update_state(state, PipelineStage.SAE_TRAINED)

    return all_sae


# ── Stage 4: Evaluation (per iteration) ──────────────────────────────────────


def _analyze_sae(
    sae_model: SAE,
    sentences: list[dict],
    analysis_dir: str,
) -> dict:
    result = {}

    with P("get_sae_features"):
        sentence_features, filtered = get_sae_features_for_sentences(
            sae_model,
            sentences,
            filter_sentence=filter_sentence,
        )

    # SAE cluster analysis
    with P("cluster_analysis"):
        result["cluster"] = cluster_analysis(filtered, sentence_features, analysis_dir)

    return result


def _run_baseline_analysis(
    sae_models: list[SAE],
    sentences: list[dict],
    analysis_dir: str,
) -> dict:
    """Run baseline clustering once per unique (layer, n_clusters) pair.

    Saves results to analysis_dir/cluster_baseline/{layer}_{n_clusters}/.
    Returns a dict keyed by "L{layer}_k{n_clusters}" with baseline metrics.
    """
    baseline_dir = os.path.join(analysis_dir, "cluster_baseline")

    # Collect unique (layer, n_clusters) pairs
    seen = set()
    unique_configs = []
    for sae_model in sae_models:
        key = (sae_model.layer, sae_model.num_latents)
        if key not in seen:
            seen.add(key)
            unique_configs.append(sae_model)

    all_baseline_results = {}
    for sae_model in unique_configs:
        layer = sae_model.layer
        n_clusters = sae_model.num_latents
        config_key = f"L{layer}_k{n_clusters}"

        X_norm, baseline_filtered = get_normalized_vectors_for_sentences(
            layer=layer,
            sentences=sentences,
            filter_sentence=filter_sentence,
        )

        config_dir = os.path.join(baseline_dir, config_key)
        with P("baseline_cluster_analysis"):
            all_baseline_results[config_key] = baseline_cluster_analysis(
                X_norm,
                baseline_filtered,
                n_clusters,
                config_dir,
            )

    return all_baseline_results


def _analyze_results(
    state: PipelineState,
    samples: list[dict],
    activations: list[dict],
    sae_models: list[SAE],
    section_activation_means: dict,
) -> None:
    all_results = []
    with P("_analyze_results"):
        analysis_dir = get_analysis_dirpath(state)
        sentences = get_sentences(samples, activations, section_activation_means)
        for sae_model in sae_models:
            name = sae_model.get_name()
            sae_analysis_dir = os.path.join(analysis_dir, name)
            result = _analyze_sae(sae_model, sentences, sae_analysis_dir)
            all_results.append(result)

        # Run baseline clustering once per unique (layer, n_clusters) — not per SAE
        baseline_results = _run_baseline_analysis(sae_models, sentences, analysis_dir)

        # Attach the relevant baseline to each SAE result
        for sae_model, result in zip(sae_models, all_results):
            config_key = f"L{sae_model.layer}_k{sae_model.num_latents}"
            result["cluster_baseline"] = baseline_results.get(config_key, {})

    state.analysis_results = all_results
    update_state(state, PipelineStage.EVALUATED)


# ── Special iteration ─────────────────────────────────────────────────────────


def _run_special_iter(state: PipelineState):
    """Retrain SAEs from scratch on all accumulated data.

    Runs in an isolated directory (special_iter/) so it doesn't interfere
    with the main pipeline's state or checkpoints.  Uses the running mean
    from the main pipeline (already accounts for all past data).
    """
    # Snapshot the main pipeline's SAE dir (where running_mean lives).
    main_sae_dir = get_sae_dirpath(state)

    # Work on a deep copy so we don't mutate the real pipeline state.
    state = copy.deepcopy(state)
    state.filepath_cfg = reset_and_get_special_filepath_cfg()
    ensure_dirs(state.filepath_cfg)

    # Override config for full retrain.
    config = state.config
    config.max_epochs = 300
    config.patience = 10

    # TensorBoard in the special dir.
    tb_log_dir = state.filepath_cfg.tensorboard_dir / state.pipeline_id
    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

    print(
        f"\n  === Special iteration: retraining from scratch (iter {state.iteration}) ==="
    )

    try:
        # Load ALL data from all completed iterations (from the main data dir,
        # which was copied into special_iter/ by reset_and_get_special_filepath_cfg).
        samples, activations = _load_all_data(state)
        activation_means = calculate_activation_means_by_section(
            samples, activations, state.config.layers
        )
        sae_models = initialize_sae_models(state)
        trained_sae_models = _train_saes(
            state,
            samples,
            activations,
            sae_models,
            activation_means,
            tb_writer=tb_writer,
        )
        _analyze_results(
            state, samples, activations, trained_sae_models, activation_means
        )
    finally:
        tb_writer.close()
        clear_gpu_memory()


# ── Single iteration ─────────────────────────────────────────────────────────


def _run_iteration(state: PipelineState, tb_writer=None, skip_analysis: bool = False):
    """Run one complete iteration: generate -> infer -> train -> evaluate."""

    ensure_dirs(state.filepath_cfg)
    iteration = state.iteration

    # Stage 1: Dataset
    samples = None
    if state.stage < PipelineStage.DATASET_GENERATED:
        print(f"\n  iter:{iteration}, Stage 1: Dataset Generation")
        _generate_dataset(state)

    # Stage 2: Inference
    if state.stage < PipelineStage.INFERENCE_DONE:
        print(f"\n  iter:{iteration}, Stage 2: Running Inference")
        _run_inference(state)

    samples = _load_samples(state)
    activations = _load_activations(state, samples)
    activation_means = _calculate_section_activation_means(state)
    sae_models = _get_sae_models(state)

    # Stage 3: SAE Training
    if state.stage < PipelineStage.SAE_TRAINED:
        print(f"\n  iter:{iteration}, Stage 3: SAE Training")
        sae_models = _train_saes(
            state,
            samples,
            activations,
            sae_models,
            activation_means,
            tb_writer=tb_writer,
        )

    if not skip_analysis:
        # Stage 4: Analysis
        if state.stage < PipelineStage.EVALUATED:
            print(f"\n  iter:{iteration}, Stage 4: Analysis")
            _analyze_results(state, samples, activations, sae_models, activation_means)

    del activations
    clear_gpu_memory()


# ── Main entry point ─────────────────────────────────────────────────────────


def run_pipeline(state: PipelineState, retrain_every_n_iter: int = 50):
    """Run the iterative pipeline from current state."""

    tb_log_dir = state.filepath_cfg.tensorboard_dir / state.pipeline_id
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    try:
        for i in range(state.iteration, state.config.max_iterations):
            state.iteration = i
            _run_iteration(state, tb_writer=writer, skip_analysis=True)
            state.stage = PipelineStage.INIT
            P.report()

            # Every N iterations, retrain SAEs from scratch on all data.
            if i > 0 and i % retrain_every_n_iter == 0:
                _run_special_iter(state)
    finally:
        writer.close()
        P.report()


# ── Test iteration ────────────────────────────────────────────────────────────


def run_test_iteration(state: PipelineState):
    """Run a single test iteration end-to-end, continuing from the real run's state.

    Copies run_data/ → test_iter/ (wiping test_iter/ first), switches output root
    to test_iter/, runs one small iteration there, then restores output root.
    All test output lives under test_iter/ and is overwritten each time.
    """

    # Change output dir
    state.filepath_cfg = reset_and_get_test_filepath_cfg()

    # Override config for minimal test
    config = state.config
    config.samples_per_iter = 8
    config.batch_size = 4
    config.max_iterations = 1
    config.max_epochs = 1
    config.patience = 1

    # Start fresh iter
    state.iteration = state.iteration + 1
    state.stage = PipelineStage.INIT  # force full iteration

    # Run
    tb_log_dir = state.filepath_cfg.tensorboard_dir / state.pipeline_id
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    try:
        _run_iteration(state, tb_writer=writer)
    finally:
        P.report()
        writer.close()
