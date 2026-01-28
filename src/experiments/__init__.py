"""Experiment orchestration for intertemporal preference analysis.

This module provides unified functions for running the full analysis pipeline:
- Data generation
- Activation patching
- Attribution patching
- Contrastive analysis (steering vectors)
- Steering evaluation

IMPORTANT DESIGN PRINCIPLES:
1. NEVER use TransformerLens/NNsight/Pyvene APIs directly - ALWAYS use ModelRunner
2. All experiment functions must work with any backend (configurable via ModelRunner)
3. No magic numbers - use config parameters or named constants
4. Check for existing code before adding new functions (avoid duplication)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..common.io import ensure_dir, save_json, get_timestamp
from ..common.positions_schema import PositionsFile, PositionSpec
from ..data import (
    PreferenceData,
    generate_preference_data,
    build_prompt_pairs,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_MODEL,
)
from ..models import ModelRunner
from ..models.intervention_utils import patch, steering
from ..analysis import (
    build_position_mapping,
    create_metric,
    find_section_markers,
    get_token_labels,
    run_all_attribution_methods,
    aggregate_attribution_results,
)
from ..profiler import P


@dataclass
class ExperimentConfig:
    """Configuration for intertemporal experiments."""

    # Data generation
    model: str = DEFAULT_MODEL
    dataset_config: dict = field(default_factory=lambda: DEFAULT_DATASET_CONFIG.copy())
    max_samples: int = 50

    # Patching
    max_pairs: int = 3
    ig_steps: int = 10
    position_threshold: float = 0.05

    # Contrastive
    contrastive_max_samples: int = 500
    top_n_positions: int = 1

    # Output
    output_dir: Optional[Path] = None


@dataclass
class ExperimentResults:
    """Results from a full experiment run."""

    pref_data: PreferenceData
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


def run_activation_patching(
    runner: ModelRunner,
    pref_data: PreferenceData,
    max_pairs: int = 3,
    threshold: float = 0.05,
    component: str = "resid_post",
) -> tuple[np.ndarray, np.ndarray, list[int], list[str], dict[str, int]]:
    """Run activation patching to identify important positions.

    Args:
        runner: Model runner
        pref_data: Preference data with prompt_text populated
        max_pairs: Number of pairs to process
        threshold: Position threshold for filtering
        component: Component to patch (resid_post, attn_out, mlp_out)

    Returns:
        Tuple of (position_sweep, full_sweep, filtered_positions, token_labels, section_markers)
    """
    pairs = build_prompt_pairs(pref_data, max_pairs=max_pairs, include_response=True)
    if not pairs:
        raise ValueError("No valid pairs found")

    clean_text, corrupted_text, clean_sample, corrupted_sample = pairs[0]
    clean_labels = [clean_sample.short_term_label, clean_sample.long_term_label]
    corrupted_labels = [corrupted_sample.short_term_label, corrupted_sample.long_term_label]

    with P("build_mapping"):
        pos_mapping, clean_len, _ = build_position_mapping(
            runner, clean_text, corrupted_text, clean_labels, corrupted_labels
        )

    token_labels = get_token_labels(runner, clean_text)
    section_markers = find_section_markers(
        runner, clean_text, clean_sample.short_term_label, clean_sample.long_term_label
    )

    with P("create_metric"):
        metric = create_metric(runner, clean_sample, corrupted_sample, clean_text, corrupted_text)

    # Position sweep: patch ALL layers at each position
    layers = list(range(runner.n_layers))
    hook_names = [f"blocks.{l}.hook_{component}" for l in layers]
    _, clean_cache = runner.run_with_cache(clean_text, names_filter=lambda n: n in hook_names)

    position_sweep = np.zeros(clean_len)
    for clean_pos in range(clean_len):
        corr_pos = pos_mapping.get(clean_pos, clean_pos)
        interventions = []
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_{component}"
            clean_act = clean_cache[hook_name]
            if clean_pos < clean_act.shape[1]:
                act_values = clean_act[0, clean_pos].detach().cpu().numpy()
                interventions.append(patch(
                    layer=layer,
                    values=act_values,
                    positions=[corr_pos],
                    component=component,
                ))
        if interventions:
            logits = runner.forward_with_intervention(corrupted_text, interventions)
            position_sweep[clean_pos] = metric(logits)

    # Filter positions above threshold
    filtered_positions = np.where(position_sweep > threshold)[0].tolist()
    if not filtered_positions:
        filtered_positions = [int(np.argmax(position_sweep))]

    # Full sweep: individual (layer, position) patching
    n_layers_sample = min(12, runner.n_layers)
    layer_indices = [int(i * (runner.n_layers - 1) / (n_layers_sample - 1)) for i in range(n_layers_sample)]

    full_sweep = np.zeros((len(layer_indices), len(filtered_positions)))
    for li, layer in enumerate(layer_indices):
        hook_name = f"blocks.{layer}.hook_{component}"
        clean_act = clean_cache[hook_name]
        for pi, clean_pos in enumerate(filtered_positions):
            corr_pos = pos_mapping.get(clean_pos, clean_pos)
            if clean_pos < clean_act.shape[1]:
                act_values = clean_act[0, clean_pos].detach().cpu().numpy()
                intervention = patch(
                    layer=layer,
                    values=act_values,
                    positions=[corr_pos],
                    component=component,
                )
                logits = runner.forward_with_intervention(corrupted_text, intervention)
                full_sweep[li, pi] = metric(logits)

    return position_sweep, full_sweep, filtered_positions, token_labels, section_markers


def run_attribution_patching(
    runner: ModelRunner,
    pref_data: PreferenceData,
    max_pairs: int = 3,
    ig_steps: int = 10,
) -> tuple[dict[str, np.ndarray], list[str], dict[str, int]]:
    """Run attribution patching methods.

    Args:
        runner: Model runner
        pref_data: Preference data
        max_pairs: Number of pairs to process
        ig_steps: Integration steps for EAP-IG

    Returns:
        Tuple of (aggregated_results, token_labels, section_markers)
    """
    pairs = build_prompt_pairs(pref_data, max_pairs=max_pairs, include_response=True)
    if not pairs:
        raise ValueError("No valid pairs found")

    all_results = []
    token_labels = None
    section_markers = None

    for i, (clean_text, corr_text, clean_sample, corr_sample) in enumerate(pairs[:max_pairs]):
        clean_labels = [clean_sample.short_term_label, clean_sample.long_term_label]
        corr_labels = [corr_sample.short_term_label, corr_sample.long_term_label]

        with P("build_mapping"):
            pos_mapping, _, _ = build_position_mapping(
                runner, clean_text, corr_text, clean_labels, corr_labels
            )

        with P("create_metric"):
            metric = create_metric(runner, clean_sample, corr_sample, clean_text, corr_text)

        with P("run_methods"):
            results = run_all_attribution_methods(
                runner, clean_text, corr_text, metric, pos_mapping, ig_steps
            )

        # Normalize by metric difference
        if abs(metric.diff) > 1e-8:
            results = {k: v / abs(metric.diff) for k, v in results.items()}

        all_results.append(results)

        if token_labels is None:
            token_labels = get_token_labels(runner, clean_text)
            section_markers = find_section_markers(
                runner, clean_text, clean_sample.short_term_label, clean_sample.long_term_label
            )

    aggregated = aggregate_attribution_results(all_results, runner.n_layers)

    # Ensure token_labels matches aggregated result length (may be padded)
    if aggregated and token_labels:
        max_len = max(v.shape[1] for v in aggregated.values())
        if len(token_labels) < max_len:
            # Pad with position indices for missing labels
            token_labels = token_labels + [f"pos{i}" for i in range(len(token_labels), max_len)]

    return aggregated, token_labels or [], section_markers or {}


def compute_steering_vector(
    runner: ModelRunner,
    pref_data: PreferenceData,
    layer: int,
    position: int,
    max_samples: int = 500,
) -> tuple[np.ndarray, dict]:
    """Compute steering vector at a specific (layer, position).

    Args:
        runner: Model runner
        pref_data: Preference data
        layer: Target layer
        position: Target position
        max_samples: Max samples per class

    Returns:
        Tuple of (direction vector, stats dict)
    """
    import torch
    from ..probes import prepare_samples

    samples, labels = prepare_samples(pref_data, "choice", "choice", random_seed=42)

    # Subsample if needed
    if len(samples) > max_samples * 2:
        np.random.seed(42)
        idx_0 = np.where(labels == 0)[0]
        idx_1 = np.where(labels == 1)[0]
        if len(idx_0) > max_samples:
            idx_0 = np.random.choice(idx_0, max_samples, replace=False)
        if len(idx_1) > max_samples:
            idx_1 = np.random.choice(idx_1, max_samples, replace=False)
        selected = np.concatenate([idx_0, idx_1])
        samples = [samples[i] for i in selected]
        labels = labels[selected]

    hook_name = f"blocks.{layer}.hook_resid_post"

    def names_filter(name: str) -> bool:
        return hook_name in name

    # Extract activations
    acts_list = []
    for sample in samples:
        text = sample.prompt_text + (sample.response or "")
        with torch.no_grad():
            _, cache = runner.run_with_cache(text, names_filter=names_filter)
        acts = cache[hook_name]
        if isinstance(acts, torch.Tensor):
            acts = acts[0].cpu().numpy()
        pos_idx = min(position, acts.shape[0] - 1)
        acts_list.append(acts[pos_idx])
        del cache

    activations = np.array(acts_list)
    acts_0 = activations[labels == 0]
    acts_1 = activations[labels == 1]

    # Compute direction
    mean_0, mean_1 = np.mean(acts_0, axis=0), np.mean(acts_1, axis=0)
    direction = mean_1 - mean_0
    norm = np.linalg.norm(direction)

    stats = {
        "layer": layer,
        "position": position,
        "direction_norm": float(norm),
        "n_class0": len(acts_0),
        "n_class1": len(acts_1),
    }

    return direction, stats


def apply_steering(
    runner: ModelRunner,
    prompt: str,
    direction: np.ndarray,
    layer: int,
    strength: float = 1.0,
    max_new_tokens: int = 100,
) -> str:
    """Apply steering vector and generate text.

    Args:
        runner: Model runner
        prompt: Input prompt
        direction: Steering direction vector
        layer: Target layer
        strength: Steering strength (can be negative to reverse)
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    intervention = steering(
        layer=layer,
        direction=direction,
        strength=strength,
        normalize=False,  # Assume already normalized
    )
    return runner.generate(prompt, max_new_tokens=max_new_tokens, intervention=intervention)


from .intertemporal import (
    ExperimentArgs,
    run_experiment,
)


def run_probe_training(
    runner: ModelRunner,
    pref_data: PreferenceData,
    layers: list[int],
    token_positions: list,
    test_split: float = 0.2,
    random_seed: int = 42,
    max_samples: int = 200,
) -> tuple[dict, dict]:
    """Train linear probes for choice and time horizon prediction.

    Trains two types of probes:
    1. Choice probe: predicts model's short_term vs long_term choice
    2. Time horizon probe: predicts prompt's time horizon (<1yr vs >1yr)

    Args:
        runner: ModelRunner instance
        pref_data: Preference data with samples
        layers: Layer indices to probe (negative indices count from end)
        token_positions: Position specs for probing
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        max_samples: Max samples to use (0 for all)

    Returns:
        Tuple of (results_dict, probes_dict) where:
        - results_dict maps probe_type to list of ProbeResult
        - probes_dict maps (probe_type, layer, pos_idx) to trained LinearProbe
    """
    from sklearn.model_selection import train_test_split
    from ..probes import LinearProbe, ProbeResult, prepare_samples, extract_activations

    # Resolve negative layer indices
    resolved_layers = []
    for l in layers:
        if l < 0:
            resolved_layers.append(runner.n_layers + l)
        else:
            resolved_layers.append(l)

    # Prepare samples for both probe types
    with P("prepare_probe_samples"):
        choice_samples, choice_labels = prepare_samples(
            pref_data, "choice", "choice", random_seed
        )
        horizon_samples, horizon_labels = prepare_samples(
            pref_data, "time_horizon", "time_horizon", random_seed
        )

    # Subsample if needed
    if max_samples > 0 and len(choice_samples) > max_samples:
        _, choice_samples, _, choice_labels = train_test_split(
            choice_samples, choice_labels, test_size=max_samples,
            stratify=choice_labels, random_state=random_seed
        )
        choice_samples = list(choice_samples)

    if max_samples > 0 and len(horizon_samples) > max_samples:
        _, horizon_samples, _, horizon_labels = train_test_split(
            horizon_samples, horizon_labels, test_size=max_samples,
            stratify=horizon_labels, random_state=random_seed
        )
        horizon_samples = list(horizon_samples)

    print(f"  Choice samples: {len(choice_samples)}")
    print(f"  Horizon samples: {len(horizon_samples)}")

    # Early return if no choice samples (critical)
    if len(choice_samples) < 4:  # Need at least 4 for train/test split
        print("  WARNING: Insufficient choice samples for probe training")
        return {}, {}

    # Extract activations
    with P("extract_probe_activations"):
        choice_extraction = extract_activations(
            runner, choice_samples, resolved_layers, token_positions
        )
        # Only extract for horizon if we have samples
        if len(horizon_samples) >= 4:
            horizon_extraction = extract_activations(
                runner, horizon_samples, resolved_layers, token_positions
            )
        else:
            horizon_extraction = None

    # Train probes
    results = {}
    probes = {}

    def train_probes_for_type(probe_type, X, y):
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=test_split,
            stratify=y, random_state=random_seed,
        )

        type_results = []
        for (layer, pos_idx), X_lp in sorted(X.items()):
            X_train, X_test = X_lp[train_idx], X_lp[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            probe = LinearProbe(random_state=random_seed)
            cv_mean, cv_std, train_acc = probe.train(X_train, y_train, n_cv_folds=0)
            test_acc, test_prec, test_rec, test_f1 = probe.evaluate(X_test, y_test)

            result = ProbeResult(
                layer=layer, token_position=pos_idx,
                cv_accuracy_mean=cv_mean, cv_accuracy_std=cv_std,
                train_accuracy=train_acc, test_accuracy=test_acc,
                test_precision=test_prec, test_recall=test_rec,
                test_f1=test_f1, n_train=len(y_train), n_test=len(y_test),
                n_features=X_train.shape[1],
            )
            type_results.append(result)
            probes[(probe_type, layer, pos_idx)] = probe

        type_results.sort(key=lambda r: (r.layer, r.token_position))
        return type_results

    with P("train_choice_probe"):
        results["choice"] = train_probes_for_type(
            "choice", choice_extraction.X, choice_labels
        )

    # Only train time_horizon probe if we have enough samples
    if horizon_extraction is not None and len(horizon_samples) >= 4:
        with P("train_horizon_probe"):
            results["time_horizon"] = train_probes_for_type(
                "time_horizon", horizon_extraction.X, horizon_labels
            )
    else:
        print("  Skipping time_horizon probe (insufficient samples)")
        results["time_horizon"] = []

    # Add extraction info for visualization
    results["_meta"] = {
        "layers": resolved_layers,
        "token_positions": token_positions,
        "choice_position_info": choice_extraction.position_info,
        "horizon_position_info": horizon_extraction.position_info if horizon_extraction else None,
    }

    return results, probes


__all__ = [
    "ExperimentConfig",
    "ExperimentResults",
    "run_activation_patching",
    "run_attribution_patching",
    "compute_steering_vector",
    "apply_steering",
    "run_probe_training",
    # Intertemporal experiment
    "ExperimentArgs",
    "run_experiment",
]
