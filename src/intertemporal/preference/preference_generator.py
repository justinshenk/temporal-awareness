"""Generate preference data on-the-fly by querying a model.

Useful for activation patching, attribution patching, probe training,
contrastive steering, and other analysis workflows that need preference
data without pre-existing saved files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ...common.file_io import ensure_dir
from ...common.device_utils import clear_gpu_memory
from ..common.project_paths import get_pref_dataset_dir, get_prompt_dataset_dir
from .preference_querier import PreferenceQuerier, PreferenceQueryConfig
from .preference_dataset import PreferenceDataset
from ..prompt import PromptDatasetGenerator, PromptDatasetConfig
from ..data.default_configs import FULL_EXPERIMENT_CONFIG
from ...common.profiler import P


def generate_preference_data(
    model: Optional[str] = None,
    dataset_config: Optional[dict] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    max_samples: Optional[int] = None,
    save_data: bool = False,
    prompt_datasets_dir: Optional[Path] = None,
    pref_datasets_dir: Optional[Path] = None,
) -> PreferenceDataset:
    """Generate preference data on-the-fly by querying a model."""

    model = model or FULL_EXPERIMENT_CONFIG["model"]
    config_dict = dataset_config or FULL_EXPERIMENT_CONFIG["dataset_config"]

    # Generate prompt dataset
    with P("generate_prompt_dataset"):
        prompt_dataset_cfg = PromptDatasetConfig.from_dict(config_dict)
        prompt_dataset = PromptDatasetGenerator(prompt_dataset_cfg).generate()

    # Build query config
    subsample = 1.0
    if max_samples and max_samples > 0 and prompt_dataset.samples:
        subsample = min(1.0, max_samples / len(prompt_dataset.samples))
    query_config = PreferenceQueryConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        subsample=subsample,
    )

    # Query model
    with P("generate_preference_dataset"):
        pref_data = PreferenceQuerier(query_config).query_dataset(prompt_dataset, model)

    # Strip heavy data (full_logits tensors) before saving to reduce file size
    pref_data.pop_heavy()
    clear_gpu_memory()

    # Save data
    if save_data:
        with P("saving_preference_dataset"):
            prompt_datasets_dir = prompt_datasets_dir or get_prompt_dataset_dir()
            pref_datasets_dir = pref_datasets_dir or get_pref_dataset_dir()
            ensure_dir(prompt_datasets_dir)
            ensure_dir(pref_datasets_dir)
            prompt_dataset.save_as_json(
                prompt_datasets_dir / prompt_dataset.config.get_filename()
            )
            pref_data.save_as_json(
                pref_datasets_dir / pref_data.get_filename(), with_internals=True
            )

    return pref_data
