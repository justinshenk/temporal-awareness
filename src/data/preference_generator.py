"""Generate preference data on-the-fly by querying a model.

Useful for activation patching, attribution patching, probe training,
contrastive steering, and other analysis workflows that need preference
data without pre-existing saved files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import torch

from ..common.io import ensure_dir
from ..prompt_datasets import PromptDatasetGenerator, PromptDatasetConfig
from ..models import QueryRunner, QueryConfig
from ..models.query_runner import InternalsConfig
from .preference_loader import PreferenceDataset, PreferenceSample
from .default_configs import DEFAULT_MODEL, DEFAULT_PROMPT_DATASET_CONFIG


def generate_preference_data(
    model: Optional[str] = None,
    dataset_config: Optional[dict] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    max_samples: Optional[int] = None,
    internals: Optional[dict] = None,
    save_data: bool = True,
    datasets_dir: Optional[Path] = None,
    pref_dir: Optional[Path] = None,
    verbose: bool = True,
) -> PreferenceDataset:
    """
    Generate preference data on-the-fly by querying a model.

    This function creates a dataset, queries the model, and returns
    PreferenceDataset with prompt_text already merged.

    When save_data is True (default), the generated dataset and preference
    files are saved to pref_dir and datasets_dir so they can be reloaded
    later with load_pref_data_with_prompts.

    Args:
        model: Model name to query (default: Qwen/Qwen2.5-1.5B-Instruct)
        dataset_config: Dataset configuration dict (default: housing scenario)
        temperature: Sampling temperature (0.0 for deterministic)
        max_new_tokens: Maximum tokens in model response
        verbose: Whether to print progress
    Returns:
        PreferenceDataset with prompt_text populated for each preference
    """
    model = model or DEFAULT_MODEL
    config_dict = dataset_config or DEFAULT_PROMPT_DATASET_CONFIG

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
        dataset_cfg = PromptDatasetConfig.load_from_dict(config_dict)
        generator = PromptDatasetGenerator(dataset_cfg)
        prompt_dataset = generator.generate()
        if verbose:
            print(f"  Generated {len(prompt_dataset.samples)} samples")

        dataset_id = prompt_dataset.dataset_id
        dataset_filename = prompt_dataset.config.get_filename()

        # Always save dataset to working dir (QueryRunner needs it on disk)
        dataset_path = working_datasets_dir / dataset_filename
        prompt_dataset.save_as_json(dataset_path)

        # Also save to the real datasets_dir if it's different from working dir
        if (
            save_data
            and datasets_dir is not None
            and working_datasets_dir != datasets_dir
        ):
            prompt_dataset.save_as_json(datasets_dir / dataset_filename)
            if verbose:
                print(f"  Saved dataset to {datasets_dir}")

        # Step 2: Query model
        if verbose:
            print(f"\nStep 2: Querying model ({model})...")
        if max_samples and max_samples > 0 and prompt_dataset.samples:
            subsample = min(1.0, max_samples / len(prompt_dataset.samples))
        else:
            subsample = 1.0
        internals_config = None
        if internals:
            internals_config = InternalsConfig.from_dict(internals)
        query_config = QueryConfig(
            internals=internals_config,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            subsample=subsample,
        )
        runner = QueryRunner(query_config)
        output = runner.query_dataset(prompt_dataset, model)
        if verbose:
            print(f"  Queried {len(output.preferences)} samples")

        # Step 3: Convert to PreferenceDataset and save internals
        if verbose:
            print("\nStep 3: Converting to PreferenceDataset...")

        pref_data = PreferenceDataset(
            prompt_dataset_id=dataset_id,
            model=model,
            preferences=[],
            prompt_dataset_name=dataset_cfg.name,
        )

        internals_dir = pref_dir / "internals" if pref_dir else None
        has_internals = any(p.internals is not None for p in output.preferences)
        if has_internals and save_data and internals_dir:
            ensure_dir(internals_dir)

        for p in output.preferences:
            internals_dict = None
            if p.internals is not None and save_data and internals_dir:
                filename = f"{pref_data.get_prefix()}_sample_{p.sample_id}.pt"
                file_path = internals_dir / filename
                torch.save(p.internals.activations, file_path)
                internals_dict = {
                    "file_path": str(file_path),
                    "activations": p.internals.activation_names,
                }

            pref_data.preferences.append(
                PreferenceSample(
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

        # Merge prompt text
        prompts_by_id = prompt_dataset.get_prompts_by_id()
        for pref in pref_data.preferences:
            pref.prompt_text = prompts_by_id.get(pref.sample_id, "")

        if verbose:
            n_with_internals = sum(1 for p in pref_data.preferences if p.internals is not None)
            msg = f"  Created {len(pref_data.preferences)} preferences with prompt text"
            if n_with_internals:
                msg += f" ({n_with_internals} with internals)"
            print(msg)

        # Save preference data
        if save_data and pref_dir is not None:
            pref_path = pref_dir / f"{pref_data.get_filename()}.json"
            pref_data.save_as_json(pref_path)
            if verbose:
                print(f"  Saved preferences to {pref_path}")

    finally:
        if tmpdir_ctx is not None:
            tmpdir_ctx.cleanup()

    return pref_data
