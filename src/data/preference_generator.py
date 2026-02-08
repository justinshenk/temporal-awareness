"""Generate preference data on-the-fly by querying a model.

Useful for activation patching, attribution patching, probe training,
contrastive steering, and other analysis workflows that need preference
data without pre-existing saved files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..common.io import ensure_dir
from ..common.paths import get_pref_dataset_dir, get_prompt_dataset_dir
from ..models import QueryRunner, QueryConfig
from ..models.preference_dataset import PreferenceDataset
from ..models.query_runner import InternalsConfig
from ..prompt_datasets import PromptDatasetGenerator, PromptDatasetConfig
from .default_configs import DEFAULT_MODEL, DEFAULT_PROMPT_DATASET_CONFIG
from ..profiler import P


def generate_preference_data(
    model: Optional[str] = None,
    dataset_config: Optional[dict] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    max_samples: Optional[int] = None,
    internals: Optional[dict] = None,
    save_data: bool = True,
    prompt_datasets_dir: Optional[Path] = None,
    pref_datasets_dir: Optional[Path] = None,
) -> PreferenceDataset:
    """Generate preference data on-the-fly by querying a model."""

    model = model or DEFAULT_MODEL
    config_dict = dataset_config or DEFAULT_PROMPT_DATASET_CONFIG

    # Generate prompt dataset
    with P("generate_prompt_dataset"):
        prompt_dataset_cfg = PromptDatasetConfig.from_dict(config_dict)
        prompt_dataset = PromptDatasetGenerator(prompt_dataset_cfg).generate()

    # Build query config
    subsample = 1.0
    if max_samples and max_samples > 0 and prompt_dataset.samples:
        subsample = min(1.0, max_samples / len(prompt_dataset.samples))
    internals_config = InternalsConfig.from_dict(internals) if internals else None
    query_config = QueryConfig(
        internals=internals_config,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        subsample=subsample,
    )

    # Query model
    with P("generate_preference_dataset"):
        pref_data = QueryRunner(query_config).query_dataset(prompt_dataset, model)

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
