"""Activation patching for identifying important token positions and layers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..data import (
    PreferenceData,
    build_prompt_pairs,
)
from ..models import ModelRunner
from ..models.intervention_utils import patch
from ..analysis import (
    build_position_mapping,
    create_metric,
    find_section_markers,
    get_token_labels,
)
from ..profiler import P


def _run_pair_position_sweep(
    runner: ModelRunner,
    clean_text: str,
    corrupted_text: str,
    clean_sample,
    corrupted_sample,
    position_sweep_component: str,
    all_hook_names: set[str],
    position_step: int,
) -> tuple[np.ndarray, dict, int, dict]:
    """Run Phase 1 position sweep for a single clean/corrupted pair.

    Returns:
        Tuple of (position_sweep, clean_cache, clean_len, pos_mapping)
    """
    clean_labels = [clean_sample.short_term_label, clean_sample.long_term_label]
    corrupted_labels = [corrupted_sample.short_term_label, corrupted_sample.long_term_label]

    with P("build_mapping"):
        pos_mapping, clean_len, _ = build_position_mapping(
            runner, clean_text, corrupted_text, clean_labels, corrupted_labels
        )

    with P("create_metric"):
        metric = create_metric(runner, clean_sample, corrupted_sample, clean_text, corrupted_text)

    layers = list(range(runner.n_layers))
    # clean_cache[hook_name] shape: [batch, seq_len, d_model]
    _, clean_cache = runner.run_with_cache(clean_text, names_filter=lambda n: n in all_hook_names)

    position_sweep = np.zeros(clean_len)  # [seq_len]
    for clean_pos in range(0, clean_len, position_step):
        corr_pos = pos_mapping.get(clean_pos, clean_pos)
        interventions = []
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_{position_sweep_component}"
            clean_act = clean_cache[hook_name]  # [batch, seq_len, d_model]
            if clean_pos < clean_act.shape[1]:
                act_values = clean_act[0, clean_pos].detach().cpu().numpy()  # [d_model]
                interventions.append(patch(
                    layer=layer,
                    values=act_values,
                    positions=[corr_pos],
                    component=position_sweep_component,
                ))
        if interventions:
            logits = runner.forward_with_intervention(corrupted_text, interventions)
            position_sweep[clean_pos] = metric(logits)

    # Normalize by metric gap so scores are comparable across pairs
    if abs(metric.diff) > 1e-8:
        position_sweep = position_sweep / abs(metric.diff)

    return position_sweep, clean_cache, clean_len, pos_mapping, metric


def _run_pair_full_sweep(
    runner: ModelRunner,
    corrupted_text: str,
    clean_cache: dict,
    pos_mapping: dict,
    metric,
    filtered_positions: list[int],
    layer_indices: list[int],
    full_sweep_components: list[str],
) -> dict[str, np.ndarray]:
    """Run Phase 2 full sweep for a single pair using pre-computed cache.

    Returns:
        dict[component -> ndarray of shape [n_sampled_layers, n_filtered_positions]]
    """
    pair_sweeps = {}
    for component in full_sweep_components:
        full_sweep = np.zeros((len(layer_indices), len(filtered_positions)))
        for li, layer in enumerate(layer_indices):
            hook_name = f"blocks.{layer}.hook_{component}"
            clean_act = clean_cache[hook_name]  # [batch, seq_len, d_model]
            for pi, clean_pos in enumerate(filtered_positions):
                corr_pos = pos_mapping.get(clean_pos, clean_pos)
                if clean_pos < clean_act.shape[1]:
                    act_values = clean_act[0, clean_pos].detach().cpu().numpy()  # [d_model]
                    intervention = patch(
                        layer=layer,
                        values=act_values,
                        positions=[corr_pos],
                        component=component,
                    )
                    logits = runner.forward_with_intervention(corrupted_text, intervention)
                    full_sweep[li, pi] = metric(logits)
        # Normalize by metric gap
        if abs(metric.diff) > 1e-8:
            full_sweep = full_sweep / abs(metric.diff)
        pair_sweeps[component] = full_sweep
    return pair_sweeps


def run_activation_patching(
    runner: ModelRunner,
    pref_data: PreferenceData,
    max_pairs: int = 3,
    threshold: float = 0.05,
    position_sweep_component: str = "resid_post",
    full_sweep_components: list[str] = ["resid_post", "attn_out", "mlp_out"],
    n_layers_sample: int = 12,
    position_step: int = 1,
    token_positions: Optional[list] = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[int], list[str], dict[str, int]]:
    """Run activation patching to identify important positions.

    Results are averaged across multiple clean/corrupted pairs for robustness.
    Each pair's scores are normalized by its metric gap before averaging, so
    pairs with different metric scales contribute equally.

    Args:
        runner: Model runner
        pref_data: Preference data with prompt_text populated
        max_pairs: Number of pairs to process
        threshold: Position threshold for filtering
        position_sweep_component: Component to patch in position sweep (resid_post, attn_out, mlp_out)
        full_sweep_components: Components to patch in full sweep (resid_post, attn_out, mlp_out)
        n_layers_sample: Number of layers to sample in sweep (evenly spaced)
        position_step: Stride for position sweep (1 = every position, 2 = every other, etc.)
        token_positions: Additional position specs (keyword/text/relative) to always
            include in the Phase 2 full sweep.  If None, uses
            ``DefaultPromptFormat().get_interesting_positions()``.
            These are resolved against the tokenized prompt and merged with the
            threshold-filtered positions from Phase 1.

    Returns:
        Tuple of (position_sweep, all_full_sweeps, filtered_positions, token_labels, section_markers)
        all_full_sweeps is a dict mapping component name -> [n_sampled_layers, n_filtered_positions]
    """
    pairs = build_prompt_pairs(pref_data, max_pairs=max_pairs, include_response=True)
    if not pairs:
        raise ValueError("No valid pairs found")

    layers = list(range(runner.n_layers))
    all_hook_names = set(
        [f"blocks.{l}.hook_{position_sweep_component}" for l in layers]
        + [f"blocks.{l}.hook_{c}" for l in layers for c in full_sweep_components]
    )

    # === Phase 1: Position sweep across all pairs ===
    # Each pair produces a [seq_len] array; we pad to max length and average.
    position_sweeps = []  # list of (sweep_array, clean_cache, clean_len, pos_mapping, metric)
    token_labels = None
    section_markers = None

    for i, (clean_text, corrupted_text, clean_sample, corrupted_sample) in enumerate(pairs[:max_pairs]):
        print(f"  Activation patching pair {i+1}/{min(max_pairs, len(pairs))}")
        sweep, clean_cache, clean_len, pos_mapping, metric = _run_pair_position_sweep(
            runner, clean_text, corrupted_text, clean_sample, corrupted_sample,
            position_sweep_component, all_hook_names, position_step,
        )
        position_sweeps.append((sweep, clean_cache, clean_len, pos_mapping, metric, corrupted_text))

        # Capture token labels and section markers from the first pair only
        if token_labels is None:
            token_labels = get_token_labels(runner, clean_text)
            section_markers = find_section_markers(
                runner, clean_text, clean_sample.short_term_label, clean_sample.long_term_label
            )

    # Average position sweeps with zero-padding for different lengths
    max_len = max(s[0].shape[0] for s in position_sweeps)
    padded_sweeps = []
    for sweep, _, _, _, _, _ in position_sweeps:
        if sweep.shape[0] < max_len:
            padded = np.zeros(max_len)
            padded[:sweep.shape[0]] = sweep
            padded_sweeps.append(padded)
        else:
            padded_sweeps.append(sweep)
    position_sweep = np.mean(padded_sweeps, axis=0)  # [max_len]

    # Pad token_labels if averaged sweep is longer
    if token_labels and len(token_labels) < max_len:
        token_labels = token_labels + [f"pos{i}" for i in range(len(token_labels), max_len)]

    # Keep positions where averaged patching score exceeds threshold
    filtered_positions = np.where(position_sweep > threshold)[0].tolist()
    if not filtered_positions:
        filtered_positions = [int(np.argmax(position_sweep))]

    # Merge in semantically interesting positions from the prompt format
    if token_positions is None:
        from ..formatting.configs.default_prompt_format import DefaultPromptFormat
        token_positions = DefaultPromptFormat().get_interesting_positions()
    if token_positions:
        from ..common.token_positions import resolve_positions
        # Resolve against first pair's tokenization
        first_clean_text = pairs[0][0]
        token_ids = runner.tokenize(first_clean_text)
        clean_token_strs = [runner.tokenizer.decode([t]) for t in token_ids[0].tolist()]
        resolved = resolve_positions(token_positions, clean_token_strs)
        first_clean_len = position_sweeps[0][2]
        for pos in resolved:
            if pos.found and 0 <= pos.index < first_clean_len and pos.index not in filtered_positions:
                filtered_positions.append(pos.index)
        filtered_positions.sort()

    # === Phase 2: Full (layer x position) sweep across all pairs ===
    actual_n_layers = min(n_layers_sample, runner.n_layers)
    if actual_n_layers > 1:
        layer_indices = [int(i * (runner.n_layers - 1) / (actual_n_layers - 1)) for i in range(actual_n_layers)]
    else:
        layer_indices = [runner.n_layers // 2]

    # Run full sweep for each pair using cached activations, then average
    all_pair_sweeps = []  # list of dict[component -> ndarray]
    for sweep, clean_cache, clean_len, pos_mapping, metric, corrupted_text in position_sweeps:
        pair_sweeps = _run_pair_full_sweep(
            runner, corrupted_text, clean_cache, pos_mapping, metric,
            filtered_positions, layer_indices, full_sweep_components,
        )
        all_pair_sweeps.append(pair_sweeps)

    # Average full sweep results across pairs
    all_full_sweeps = {}
    for component in full_sweep_components:
        component_arrays = [ps[component] for ps in all_pair_sweeps]
        all_full_sweeps[component] = np.mean(component_arrays, axis=0)

    return position_sweep, all_full_sweeps, filtered_positions, token_labels or [], section_markers or {}
