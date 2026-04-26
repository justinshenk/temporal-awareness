"""Config loading for activation caching."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the EAP-IG YAML config."""
    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_settings(config_path: Path) -> tuple[str, int, str | None]:
    """Read only setup.model, setup.batch_size, and setup.dtype from config."""
    config = load_config(config_path)
    setup = config["setup"]
    return setup["model"], setup["batch_size"], setup.get("dtype")
