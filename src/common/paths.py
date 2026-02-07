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


def get_prompt_dataset_dir() -> Path:
    """Return the datasets output directory."""
    return get_output_dir() / "prompt_datasets"


def get_prompt_dataset_configs_dir() -> Path:
    """Return the prompt dataset configs directory."""
    return (
        get_project_root() / "scripts" / "intertemporal" / "configs" / "prompt_datasets"
    )


def get_query_configs_dir() -> Path:
    """Return the query configs directory."""
    return get_project_root() / "scripts" / "intertemporal" / "configs" / "query"


def get_circuits_configs_dir() -> Path:
    """Return the circuits configs directory."""
    return get_project_root() / "scripts" / "intertemporal" / "configs" / "circuits"


def get_probes_configs_dir() -> Path:
    """Return the probes configs directory."""
    return get_project_root() / "scripts" / "intertemporal" / "configs" / "probes"
