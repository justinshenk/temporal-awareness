"""
Base I/O utilities for saving and loading data.

Provides core JSON/JSONL utilities. Output-specific functions are in scripts/io.py.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path


# =============================================================================
# Base I/O Functions
# =============================================================================


def setup_project_paths(file: str, levels_to_root: int = 2) -> tuple[Path, Path]:
    """Set up project paths and add them to sys.path.

    Args:
        file: The __file__ of the calling script
        levels_to_root: Number of parent directories to reach project root

    Returns:
        Tuple of (project_root, script_dir)

    Example:
        # In scripts/data/generate_dataset.py
        PROJECT_ROOT, SCRIPTS_DIR = setup_project_paths(__file__, levels_to_root=3)
    """
    script_dir = Path(file).parent
    project_root = script_dir
    for _ in range(levels_to_root):
        project_root = project_root.parent

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    return project_root, script_dir


def parse_file_path(filename, default_ext=".json", default_dir_path=""):
    """Parse filename into full file path.

    Args:
        filename: Simple name, file name, or full path
        default_ext: Extension to add if filename is a simple name
        default_dir_path: Directory to prepend if filename is not a full path

    Returns:
        Full file path as Path object
    """
    default_ext = default_ext if default_ext.startswith(".") else f".{default_ext}"
    if is_simple_name(filename):
        return Path(default_dir_path) / f"{filename}{default_ext}"
    elif is_file_name(filename):
        return Path(default_dir_path) / filename
    elif is_file_path(filename):
        return Path(filename)
    else:
        raise Exception(f"{filename} is not valid")


def is_simple_name(s: str) -> bool:
    """Check if string is a simple name (no path separators or extensions)."""
    return "/" not in s and "\\" not in s and "." not in s


def is_path(s: str) -> bool:
    """Check if string is a simple name (no path separators or extensions)."""
    return "/" in s or "\\" in s


def is_file_name(s: str, ext: str | None = None) -> bool:
    """Check if string looks like a file path (has extension or path separators).

    Args:
        s: String to check
        ext: Optional extension to match (e.g., ".json" or "json")
    """
    if is_simple_name(s):
        return False
    if is_path(s):
        return False
    if s.count(".") != 1:
        return False
    if ext:
        ext = ext if ext.startswith(".") else f".{ext}"
        if not s.endswith(ext):
            return False
    return True


def is_file_path(s: str, ext: str | None = None) -> bool:
    """Check if string looks like a file path (has extension or path separators).

    Args:
        s: String to check
        ext: Optional extension to match (e.g., ".json" or "json")
    """
    is_path = ("/" in s or "\\" in s or "." in s) and not s.endswith("/")
    if not is_path or ext is None:
        return is_path
    ext = ext if ext.startswith(".") else f".{ext}"
    return s.endswith(ext)


def is_dir_path(s: str) -> bool:
    """Check if string looks like a directory path."""
    return s.endswith("/") or ("/" in s and "." not in s.split("/")[-1])


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_text_readable(obj):
    """Recursively convert long text fields to arrays of lines for readability."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in ("text", "raw_text", "trace") and isinstance(v, str) and "\n" in v:
                # Convert multiline text to array of lines
                result[k] = v.split("\n")
            else:
                result[k] = _make_text_readable(v)
        return result
    elif isinstance(obj, list):
        return [_make_text_readable(item) for item in obj]
    else:
        return obj


def save_json(data, path: Path, readable_text: bool = True) -> None:
    """Save dictionary as pretty JSON."""
    if readable_text:
        data = _make_text_readable(data)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=str, ensure_ascii=False)


def _restore_text_fields(obj):
    """Recursively restore text fields from arrays back to strings."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in ("text", "raw_text", "trace") and isinstance(v, list):
                # Join array of lines back to string
                result[k] = "\n".join(v)
            else:
                result[k] = _restore_text_fields(v)
        return result
    elif isinstance(obj, list):
        return [_restore_text_fields(item) for item in obj]
    else:
        return obj


def load_json(path: Path) -> dict:
    """Load JSON file. Robust to trailing commas."""
    with open(path) as f:
        s = f.read()
    s = re.sub(r",\s*([}\]])", r"\1", s)
    data = json.loads(s)
    return _restore_text_fields(data)


def save_jsonl(items: list[dict], path: Path) -> None:
    """Save list of dicts as JSONL."""
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, default=str) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items
