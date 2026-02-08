"""Project path utilities."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent


def get_output_dir() -> Path:
    """Return the preference data output directory."""
    return get_project_root() / "out"


def get_experiment_dir() -> Path:
    """Return the preference data output directory."""
    return get_output_dir() / "experiments"


def get_pref_dataset_dir() -> Path:
    """Return the preference data output directory."""
    return get_output_dir() / "preference_datasets"


def get_internals_dir(pref_dataset_dir: Path = get_pref_dataset_dir()) -> Path:
    return pref_dataset_dir / "internals"


def get_prompt_dataset_dir() -> Path:
    """Return the datasets output directory."""
    return get_output_dir() / "prompt_datasets"


def get_intertemporal_configs_dir() -> Path:
    """Return the prompt dataset configs directory."""
    return get_project_root() / "scripts" / "intertemporal" / "configs"


def get_prompt_dataset_configs_dir() -> Path:
    """Return the prompt dataset configs directory."""
    return get_intertemporal_configs_dir() / "prompt_datasets"


def get_query_configs_dir() -> Path:
    """Return the query configs directory."""
    return get_intertemporal_configs_dir() / "query"
