"""Dataset loading for activation caching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .cache_activations_paths import PROJECT_ROOT  # noqa: F401
except ImportError:
    from cache_activations_paths import PROJECT_ROOT  # noqa: F401

from src.intertemporal.data.default_datasets import (
    FULL_EXPERIMENT_DATASET_CONFIG,
    GEO_VIZ_CFG,
    MINIMAL_EXPERIMENT_DATASET_CONFIG,
    MULTILABEL_EXPERIMENT_DATASET_CONFIG,
)
from src.intertemporal.prompt import PromptDataset, PromptDatasetConfig  # type: ignore
from src.intertemporal.prompt.prompt_dataset_generator import PromptDatasetGenerator

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "geo_viz": GEO_VIZ_CFG,
    "full": FULL_EXPERIMENT_DATASET_CONFIG,
    "minimal": MINIMAL_EXPERIMENT_DATASET_CONFIG,
    "multilabel": MULTILABEL_EXPERIMENT_DATASET_CONFIG,
}


def load_prompt_dataset(dataset: str = "geo_viz") -> PromptDataset:
    """Load a saved PromptDataset JSON or generate a named default dataset."""
    dataset_path = Path(dataset)
    if dataset_path.exists():
        return PromptDataset.from_json(str(dataset_path))

    if dataset not in DATASET_CONFIGS:
        known = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(
            f"Unknown dataset '{dataset}'. Use one of {known}, or pass a JSON path."
        )

    config = PromptDatasetConfig.from_dict(DATASET_CONFIGS[dataset])
    return PromptDatasetGenerator(config).generate()


def get_time_horizon_months(sample: Any) -> float:
    """Return the sample horizon in months; use 60 months for no horizon."""
    if sample.prompt.time_horizon is None:
        return 60.0
    return sample.prompt.time_horizon.to_months()


def get_time_horizon_label(sample: Any) -> str:
    """Return a stable label for the raw horizon value."""
    if sample.prompt.time_horizon is None:
        return "none"
    return str(sample.prompt.time_horizon)


def get_sample_text(sample: Any) -> str:
    """Return rendered prompt text from a PromptSample."""
    text = getattr(sample, "text", None)
    if isinstance(text, str) and text:
        return text
    raise ValueError(f"Prompt sample {sample!r} does not contain rendered text.")
