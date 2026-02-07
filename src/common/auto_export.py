"""Auto-export utilities for package __init__.py files."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def auto_export(
    init_file: str,
    package_name: str,
    globals_dict: dict[str, Any],
    recursive: bool = False,
) -> list[str]:
    """Auto-import all modules in a package and export their public names.

    Args:
        init_file: The __file__ of the calling __init__.py
        package_name: The __name__ of the calling package
        globals_dict: The globals() dict of the calling module
        recursive: Whether to recurse into subpackages

    Returns:
        List of exported names (for use as __all__)

    Usage in __init__.py:
        from src.common.auto_export import auto_export
        __all__ = auto_export(__file__, __name__, globals())
    """
    package_dir = Path(init_file).parent
    all_names: list[str] = []

    # Import .py files
    for module_path in sorted(package_dir.glob("*.py")):
        if module_path.name == "__init__.py":
            continue

        module_name = module_path.stem
        try:
            module = importlib.import_module(f".{module_name}", package=package_name)
        except ImportError:
            continue

        # Get public names from module
        module_all = getattr(module, "__all__", None)
        if module_all is not None:
            names = module_all
        else:
            names = [n for n in dir(module) if not n.startswith("_")]

        for name in names:
            if name not in globals_dict:
                globals_dict[name] = getattr(module, name)
                all_names.append(name)

    # Optionally recurse into subpackages
    if recursive:
        for subdir in sorted(package_dir.iterdir()):
            if subdir.is_dir() and (subdir / "__init__.py").exists():
                subpkg_name = subdir.name
                if subpkg_name.startswith("_"):
                    continue
                try:
                    subpkg = importlib.import_module(
                        f".{subpkg_name}", package=package_name
                    )
                    globals_dict[subpkg_name] = subpkg
                    all_names.append(subpkg_name)
                except ImportError:
                    continue

    return all_names
