"""Generate preference data on-the-fly by querying a model.

Useful for activation patching, attribution patching, probe training,
contrastive steering, and other analysis workflows that need preference
data without pre-existing saved files.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ..common.io import save_json
from .preference_loader import PreferenceData, PreferenceItem


# Default dataset configuration for quick experiments
DEFAULT_DATASET_CONFIG = {
    "name": "standalone_default",
    "context": {
        "reward_unit": "housing units",
        "role": "the city administration",
        "situation": "Plan for housing development in the city.",
    },
    "options": {
        "short_term": {
            "reward_range": [1000, 5000],
            "time_range": [[1, "months"], [6, "months"]],
            "reward_steps": [2, "linear"],
            "time_steps": [2, "linear"],
        },
        "long_term": {
            "reward_range": [8000, 30000],
            "time_range": [[2, "years"], [10, "years"]],
            "reward_steps": [2, "logarithmic"],
            "time_steps": [2, "logarithmic"],
        },
    },
    "time_horizons": [None, [1, "years"], [3, "years"]],
    "add_formatting_variations": True,
}

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def generate_preference_data(
    model: Optional[str] = None,
    dataset_config: Optional[dict] = None,
    max_samples: int = 50,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    verbose: bool = True,
) -> PreferenceData:
    """
    Generate preference data on-the-fly by querying a model.

    This function creates a dataset, queries the model, and returns
    PreferenceData with prompt_text already merged. Useful for quick
    experiments without needing to save/load intermediate files.

    Args:
        model: Model name to query (default: Qwen/Qwen2.5-1.5B-Instruct)
        dataset_config: Dataset configuration dict (default: housing scenario)
        max_samples: Maximum samples to generate
        temperature: Sampling temperature (0.0 for deterministic)
        max_new_tokens: Maximum tokens in model response
        verbose: Whether to print progress

    Returns:
        PreferenceData with prompt_text populated for each preference

    Example:
        >>> pref_data = generate_preference_data(max_samples=20)
        >>> short_term, long_term = pref_data.split_by_choice()
        >>> print(f"Short-term: {len(short_term)}, Long-term: {len(long_term)}")
    """
    from ..datasets import DatasetGenerator
    from ..models import QueryRunner, QueryConfig

    model = model or DEFAULT_MODEL
    config_dict = dataset_config or DEFAULT_DATASET_CONFIG

    if verbose:
        print(f"\n{'=' * 60}")
        print("Generating preference data on-the-fly")
        print(f"{'=' * 60}")
        print(f"  Model: {model}")
        print(f"  Dataset config: {config_dict.get('name', 'default')}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Generate dataset
        if verbose:
            print("\nStep 1: Generating dataset...")
        dataset_cfg = DatasetGenerator.load_dataset_config_from_dict(config_dict)
        generator = DatasetGenerator(dataset_cfg)
        samples = generator.generate()
        if verbose:
            print(f"  Generated {len(samples)} samples")

        # Save temporarily for QueryRunner
        dataset_id = dataset_cfg.get_id()
        dataset_data = {
            "dataset_id": dataset_id,
            "config": dataset_cfg.to_dict(),
            "samples": [asdict(s) for s in samples],
        }
        dataset_path = tmpdir / f"dataset_{dataset_id}.json"
        save_json(dataset_data, dataset_path)

        # Step 2: Query model
        if verbose:
            print(f"\nStep 2: Querying model ({model})...")
        subsample = min(1.0, max_samples / len(samples)) if samples else 1.0
        query_config = QueryConfig(
            models=[model],
            datasets=[dataset_id],
            internals=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            subsample=subsample,
        )
        runner = QueryRunner(query_config, tmpdir)
        output = runner.query_dataset(dataset_id, model)
        if verbose:
            output.print_summary()

        # Step 3: Convert to PreferenceData
        if verbose:
            print("\nStep 3: Converting to PreferenceData...")
        preferences = []
        for p in output.preferences:
            preferences.append(
                PreferenceItem(
                    sample_id=p.sample_id,
                    time_horizon=p.time_horizon,
                    short_term_label=p.short_term_label,
                    long_term_label=p.long_term_label,
                    choice=p.choice,
                    choice_prob=p.choice_prob,
                    alt_prob=p.alt_prob,
                    response=p.response,
                    internals=None,  # Not needed for most analyses
                )
            )

        pref_data = PreferenceData(
            dataset_id=dataset_id,
            model=model,
            preferences=preferences,
        )

        # Merge prompt text
        prompts_by_id = {
            s["sample_id"]: s["prompt"]["text"]
            for s in dataset_data["samples"]
        }
        for pref in pref_data.preferences:
            pref.prompt_text = prompts_by_id.get(pref.sample_id, "")

        if verbose:
            print(f"  Created {len(pref_data.preferences)} preferences with prompt text")

    return pref_data
