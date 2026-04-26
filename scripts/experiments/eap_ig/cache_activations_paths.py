"""Paths and small shared helpers for activation caching."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def find_project_root(start: Path) -> Path:
    """Find the repository root by walking upward until src/ is present."""
    for path in (start, *start.parents):
        if (path / "src").is_dir():
            return path
    raise RuntimeError(f"Could not find project root containing src/ from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_NODES_PATH = PROJECT_ROOT / "data" / "selected_nodes" / "final_node_list.pkl"
DEFAULT_CONFIG_PATH = (
    Path(__file__).parent
    / "config"
    / "final_configs"
    / "Q_A"
    / "(1)FM_4B_explicit_GC.yaml"
)
HF_REPO_ID = "Temporal_Awareness_Node_Scores"


def chunk_list(items: list[Any], batch_size: int) -> list[list[Any]]:
    """Split a list into fixed-size chunks."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def resolve_path(path: str | Path) -> Path:
    """Resolve repo-relative paths."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
