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

from ..common.io import ensure_dir, save_json
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
    max_samples: Optional[int] = 50,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    verbose: bool = True,
    save_data: bool = True,
    pref_dir: Optional[Path] = None,
    datasets_dir: Optional[Path] = None,
    internals: Optional[dict] = None,
) -> PreferenceData:
    """
    Generate preference data on-the-fly by querying a model.

    This function creates a dataset, queries the model, and returns
    PreferenceData with prompt_text already merged.

    When save_data is True (default), the generated dataset and preference
    files are saved to pref_dir and datasets_dir so they can be reloaded
    later with load_pref_data_with_prompts.

    Args:
        model: Model name to query (default: Qwen/Qwen2.5-1.5B-Instruct)
        dataset_config: Dataset configuration dict (default: housing scenario)
        max_samples: Maximum samples to generate (None/0/negative = use all)
        temperature: Sampling temperature (0.0 for deterministic)
        max_new_tokens: Maximum tokens in model response
        verbose: Whether to print progress
        save_data: Whether to save dataset and preference files to disk
        pref_dir: Directory to save preference JSON files (required if save_data=True)
        datasets_dir: Directory to save dataset JSON files (required if save_data=True)
        internals: Optional internals config dict (e.g. {"activations": {"resid_post": {"layers": [0,5,10]}}})

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

    # Use provided directories for saving, or a temporary directory if not saving
    use_tmpdir = not save_data or datasets_dir is None
    tmpdir_ctx = tempfile.TemporaryDirectory() if use_tmpdir else None
    working_datasets_dir = Path(tmpdir_ctx.name) if tmpdir_ctx else datasets_dir

    try:
        if save_data:
            if datasets_dir is not None:
                ensure_dir(datasets_dir)
            if pref_dir is not None:
                ensure_dir(pref_dir)

        # Step 1: Generate dataset
        if verbose:
            print("\nStep 1: Generating dataset...")
        dataset_cfg = DatasetGenerator.load_dataset_config_from_dict(config_dict)
        generator = DatasetGenerator(dataset_cfg)
        samples = generator.generate()
        if verbose:
            print(f"  Generated {len(samples)} samples")

        dataset_id = dataset_cfg.get_id()
        dataset_data = {
            "dataset_id": dataset_id,
            "config": dataset_cfg.to_dict(),
            "samples": [asdict(s) for s in samples],
        }
        # Always save dataset to working dir (QueryRunner needs it on disk)
        dataset_path = working_datasets_dir / f"dataset_{dataset_id}.json"
        save_json(dataset_data, dataset_path)

        # Also save to the real datasets_dir if it's different from working dir
        if save_data and datasets_dir is not None and working_datasets_dir != datasets_dir:
            save_json(dataset_data, datasets_dir / f"dataset_{dataset_id}.json")
            if verbose:
                print(f"  Saved dataset to {datasets_dir}")

        # Step 2: Query model
        if verbose:
            print(f"\nStep 2: Querying model ({model})...")
        if max_samples and max_samples > 0 and samples:
            subsample = min(1.0, max_samples / len(samples))
        else:
            subsample = 1.0
        internals_config = None
        if internals:
            from ..models.query_runner import InternalsConfig
            internals_config = InternalsConfig(**internals)
        query_config = QueryConfig(
            models=[model],
            datasets=[dataset_id],
            internals=internals_config,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            subsample=subsample,
        )
        runner = QueryRunner(query_config, working_datasets_dir)
        output = runner.query_dataset(dataset_id, model)
        if verbose:
            output.print_summary()

        # Step 3: Convert to PreferenceData and save internals
        if verbose:
            print("\nStep 3: Converting to PreferenceData...")
        model_name = model.split("/")[-1]
        internals_dir = pref_dir / "internals" if pref_dir else None
        has_internals = any(p.internals is not None for p in output.preferences)
        if has_internals and save_data and internals_dir:
            ensure_dir(internals_dir)

        preferences = []
        for p in output.preferences:
            internals_dict = None
            if p.internals is not None and save_data and internals_dir:
                import torch
                filename = f"{dataset_id}_{model_name}_sample_{p.sample_id}.pt"
                file_path = internals_dir / filename
                torch.save(p.internals.activations, file_path)
                internals_dict = {
                    "file_path": str(file_path),
                    "activations": p.internals.activation_names,
                }

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
                    internals=internals_dict,
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
            n_with_internals = sum(1 for p in preferences if p.internals is not None)
            msg = f"  Created {len(pref_data.preferences)} preferences with prompt text"
            if n_with_internals:
                msg += f" ({n_with_internals} with internals)"
            print(msg)

        # Save preference data
        if save_data and pref_dir is not None:
            # Build preference JSON matching the format load_preference_data expects
            pref_json = {
                "dataset_id": dataset_id,
                "model": model,
                "preferences": [asdict(p) for p in pref_data.preferences],
            }
            model_name = model.split("/")[-1]
            pref_path = pref_dir / f"{dataset_id}_{model_name}.json"
            save_json(pref_json, pref_path)
            if verbose:
                print(f"  Saved preferences to {pref_path}")

    finally:
        if tmpdir_ctx is not None:
            tmpdir_ctx.cleanup()

    return pref_data
