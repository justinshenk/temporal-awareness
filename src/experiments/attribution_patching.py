"""Attribution patching methods (EAP, EAP-IG)."""

from __future__ import annotations

import numpy as np

from ..preference import (
    PreferenceDataset,
    build_prompt_pairs,
)
from ..models import ModelRunner
from ..analysis import (
    build_position_mapping,
    create_metric,
    find_section_markers,
    get_token_labels,
    run_all_attribution_methods,
    aggregate_attribution_results,
)
from ..common.profiler import P


def run_attribution_patching(
    runner: ModelRunner,
    pref_data: PreferenceDataset,
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

    # Accumulate results across multiple clean/corrupted pairs for averaging
    all_results = []  # list of dict[method_name -> ndarray]
    token_labels = None
    section_markers = None

    for i, (clean_text, corr_text, clean_sample, corr_sample) in enumerate(pairs[:max_pairs]):
        clean_labels = [clean_sample.short_term_label, clean_sample.long_term_label]
        corr_labels = [corr_sample.short_term_label, corr_sample.long_term_label]

        with P("build_mapping"):
            # Align token positions between clean and corrupted prompts
            pos_mapping, _, _ = build_position_mapping(
                runner, clean_text, corr_text, clean_labels, corr_labels
            )

        with P("create_metric"):
            metric = create_metric(runner, clean_sample, corr_sample, clean_text, corr_text)

        with P("run_methods"):
            # Run EAP, EAP-IG, and other attribution methods
            # results: dict[method_name -> ndarray of shape [n_layers, seq_len]]
            results = run_all_attribution_methods(
                runner, clean_text, corr_text, metric, pos_mapping, ig_steps
            )

        # Normalize attribution scores by the clean-vs-corrupted metric gap
        # so values are comparable across pairs with different metric scales
        if abs(metric.diff) > 1e-8:
            results = {k: v / abs(metric.diff) for k, v in results.items()}

        all_results.append(results)

        # Capture token labels and section markers from the first pair only
        if token_labels is None:
            token_labels = get_token_labels(runner, clean_text)
            section_markers = find_section_markers(
                runner, clean_text, clean_sample.short_term_label, clean_sample.long_term_label
            )

    # Average attribution scores across pairs
    # aggregated: dict[method_name -> ndarray of shape [n_layers, max_seq_len]]
    aggregated = aggregate_attribution_results(all_results, runner.n_layers)

    # Pad token_labels if aggregated arrays are longer (due to varying seq lengths)
    if aggregated and token_labels:
        max_len = max(v.shape[1] for v in aggregated.values())
        if len(token_labels) < max_len:
            token_labels = token_labels + [f"pos{i}" for i in range(len(token_labels), max_len)]

    return aggregated, token_labels or [], section_markers or {}
