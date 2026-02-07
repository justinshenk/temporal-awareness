"""Linear probe training for choice and time horizon prediction."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split

from ..data import PreferenceDataset
from ..formatting.configs.default_prompt_format import DefaultPromptFormat
from ..models import ModelRunner
from ..probes import LinearProbe, ProbeResult, prepare_samples, extract_activations
from ..profiler import P


def run_probe_training(
    runner: ModelRunner,
    pref_data: PreferenceDataset,
    layers: list[int],
    token_positions: Optional[list] = None,
    test_split: float = 0.2,
    random_seed: int = 42,
    max_samples: int = 0,
) -> tuple[dict, dict]:
    """Train linear probes for choice and time horizon prediction.

    Trains two types of probes:
    1. Choice probe: predicts model's short_term vs long_term choice
    2. Time horizon probe: predicts prompt's time horizon (<1yr vs >1yr)

    Args:
        runner: ModelRunner instance
        pref_data: Preference data with samples
        layers: Layer indices to probe (negative indices count from end)
        token_positions: Position specs for probing.  If None, uses
            ``DefaultPromptFormat().get_interesting_positions()``.
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        max_samples: Max samples to use (0 for all)

    Returns:
        Tuple of (results_dict, probes_dict) where:
        - results_dict maps probe_type to list of ProbeResult
        - probes_dict maps (probe_type, layer, pos_idx) to trained LinearProbe
    """
    if token_positions is None:
        token_positions = DefaultPromptFormat().get_interesting_positions()

    # Resolve negative layer indices (e.g. -1 -> last layer)
    resolved_layers = []
    for l in layers:
        if l < 0:
            resolved_layers.append(runner.n_layers + l)
        else:
            resolved_layers.append(l)

    # Prepare labeled samples for both probe types
    with P("prepare_probe_samples"):
        # choice: binary label based on model's short_term vs long_term selection
        choice_samples, choice_labels = prepare_samples(
            pref_data, "choice", "choice", random_seed
        )
        # choice_labels: ndarray [n_samples] with values 0 or 1

        # time_horizon: binary label based on prompt's time horizon
        horizon_samples, horizon_labels = prepare_samples(
            pref_data, "time_horizon", "time_horizon", random_seed
        )
        # horizon_labels: ndarray [n_samples] with values 0 or 1

    # Stratified subsample to cap at max_samples while preserving class balance
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

    # Need at least 4 samples for a meaningful train/test split
    if len(choice_samples) < 4:
        print("  WARNING: Insufficient choice samples for probe training")
        return {}, {}

    # Extract activations at each (layer, position) for all samples
    with P("extract_probe_activations"):
        # choice_extraction.X: dict[(layer, pos_idx) -> ndarray of shape [n_samples, d_model]]
        choice_extraction = extract_activations(
            runner, choice_samples, resolved_layers, token_positions
        )
        if len(horizon_samples) >= 4:
            # horizon_extraction.X: dict[(layer, pos_idx) -> ndarray [n_samples, d_model]]
            horizon_extraction = extract_activations(
                runner, horizon_samples, resolved_layers, token_positions
            )
        else:
            horizon_extraction = None

    results = {}
    probes = {}

    def train_probes_for_type(probe_type, X, y):
        """Train a linear probe at each (layer, position) for a given label type.

        Args:
            probe_type: Name of the probe (e.g. "choice", "time_horizon")
            X: dict[(layer, pos_idx) -> ndarray [n_samples, d_model]]
            y: ndarray [n_samples] binary labels
        """
        # Split sample indices into train/test (stratified by label)
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),  # [n_samples]
            test_size=test_split,
            stratify=y,
            random_state=random_seed,
        )

        type_results = []
        for (layer, pos_idx), X_lp in sorted(X.items()):
            # X_lp: [n_samples, d_model] - activations for this (layer, position)
            X_train, X_test = X_lp[train_idx], X_lp[test_idx]  # [n_train, d_model], [n_test, d_model]
            y_train, y_test = y[train_idx], y[test_idx]         # [n_train], [n_test]

            probe = LinearProbe(random_state=random_seed)
            cv_mean, cv_std, train_acc = probe.train(X_train, y_train, n_cv_folds=0)
            test_acc, test_prec, test_rec, test_f1 = probe.evaluate(X_test, y_test)

            result = ProbeResult(
                layer=layer, token_position=pos_idx,
                cv_accuracy_mean=cv_mean, cv_accuracy_std=cv_std,
                train_accuracy=train_acc, test_accuracy=test_acc,
                test_precision=test_prec, test_recall=test_rec,
                test_f1=test_f1, n_train=len(y_train), n_test=len(y_test),
                n_features=X_train.shape[1],  # d_model
            )
            type_results.append(result)
            probes[(probe_type, layer, pos_idx)] = probe

        type_results.sort(key=lambda r: (r.layer, r.token_position))
        return type_results

    with P("train_choice_probe"):
        results["choice"] = train_probes_for_type(
            "choice", choice_extraction.X, choice_labels
        )

    if horizon_extraction is not None and len(horizon_samples) >= 4:
        with P("train_horizon_probe"):
            results["time_horizon"] = train_probes_for_type(
                "time_horizon", horizon_extraction.X, horizon_labels
            )
    else:
        print("  Skipping time_horizon probe (insufficient samples)")
        results["time_horizon"] = []

    # Store metadata for downstream visualization
    results["_meta"] = {
        "layers": resolved_layers,
        "token_positions": token_positions,
        "choice_position_info": choice_extraction.position_info,
        "horizon_position_info": horizon_extraction.position_info if horizon_extraction else None,
    }

    return results, probes
