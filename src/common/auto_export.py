"""Auto-export utilities for package __init__.py files."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

# Names to exclude from auto-export (stdlib modules, special names, etc.)
_EXCLUDE_NAMES = frozenset({
    # Special names
    "annotations",
    # Common stdlib modules that get imported
    "abc", "ast", "asyncio", "base64", "binascii", "builtins", "bz2",
    "calendar", "cmath", "codecs", "collections", "configparser", "contextlib",
    "copy", "copyreg", "csv", "ctypes", "dataclasses", "datetime", "decimal",
    "difflib", "dis", "email", "enum", "errno", "fcntl", "filecmp", "fileinput",
    "fnmatch", "fractions", "functools", "gc", "getopt", "getpass", "glob",
    "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http", "imaplib",
    "importlib", "inspect", "io", "ipaddress", "itertools", "json", "keyword",
    "linecache", "locale", "logging", "lzma", "mailbox", "math", "mimetypes",
    "mmap", "multiprocessing", "netrc", "numbers", "operator", "os", "pathlib",
    "pickle", "platform", "plistlib", "poplib", "posix", "posixpath", "pprint",
    "profile", "pwd", "py_compile", "pyclbr", "pydoc", "queue", "quopri",
    "random", "re", "readline", "reprlib", "resource", "rlcompleter", "runpy",
    "sched", "secrets", "select", "selectors", "shelve", "shlex", "shutil",
    "signal", "smtplib", "socket", "socketserver", "sqlite3", "ssl", "stat",
    "statistics", "string", "stringprep", "struct", "subprocess", "sys",
    "sysconfig", "syslog", "tarfile", "telnetlib", "tempfile", "termios",
    "textwrap", "threading", "time", "timeit", "token", "tokenize", "trace",
    "traceback", "tracemalloc", "tty", "turtle", "types", "typing", "unicodedata",
    "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref",
    "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
    "zipfile", "zipimport", "zlib",
    # Common third-party modules
    "numpy", "np", "torch", "pytest", "pandas", "pd",
})


def _should_export(name: str, obj: Any) -> bool:
    """Check if a name should be exported."""
    if name.startswith("_"):
        return False
    if name in _EXCLUDE_NAMES:
        return False
    # Exclude modules (we want classes, functions, constants - not module imports)
    if isinstance(obj, type(sys)):
        return False
    return True


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
            obj = getattr(module, name)
            if name not in globals_dict and _should_export(name, obj):
                globals_dict[name] = obj
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
