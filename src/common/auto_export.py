"""Auto-export utilities for package __init__.py files.

Provides automatic re-exporting of public names from submodules and subpackages,
reducing boilerplate in __init__.py files.

IMPORT PATTERNS
===============

Given this folder structure:
    src/
      common/
        __init__.py      <- explicit __all__ re-exports from subpackages
        choice/
          __init__.py    <- explicit __all__
          simple_binary_choice.py  <- defines SimpleBinaryChoice
        math/
          __init__.py              <- explicit __all__
          math_primitives.py       <- defines normalize, argmax

This enables these import styles:

1. FLAT IMPORTS - Import classes/functions directly from package:

   from src.common import SimpleBinaryChoice, normalize
   from src.common.math import normalize, argmax

   This works because __init__.py explicitly re-exports from subpackages.

2. SUBPACKAGE IMPORTS - Import subpackages as modules:

   from src.common import math
   math.normalize(...)

   Subpackages are available as attributes.

3. EXPLICIT MODULE IMPORTS - Still work as normal:

   from src.common.choice.simple_binary_choice import SimpleBinaryChoice
   from src.common.math.math_primitives import normalize

USAGE IN __init__.py
====================

Just add these two lines to any package's __init__.py:

    from src.common.auto_export import auto_export
    __all__ = auto_export(__file__, __name__, globals())

That's it. No need to modify __init__.py when adding new files.

WHAT GETS EXPORTED
==================

- All public names (not starting with _) from .py files in the directory
- All subpackages (directories with __init__.py)
- Excludes: stdlib names, third-party libs, typing imports, dataclass helpers
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

# =============================================================================
# Exclusion Configuration
# =============================================================================

STDLIB_MODULES = frozenset(
    {
        "abc",
        "ast",
        "asyncio",
        "base64",
        "binascii",
        "builtins",
        "bz2",
        "calendar",
        "cmath",
        "codecs",
        "collections",
        "configparser",
        "contextlib",
        "copy",
        "copyreg",
        "csv",
        "ctypes",
        "dataclasses",
        "datetime",
        "decimal",
        "difflib",
        "dis",
        "email",
        "enum",
        "errno",
        "fcntl",
        "filecmp",
        "fileinput",
        "fnmatch",
        "fractions",
        "functools",
        "gc",
        "getopt",
        "getpass",
        "glob",
        "grp",
        "gzip",
        "hashlib",
        "heapq",
        "hmac",
        "html",
        "http",
        "imaplib",
        "importlib",
        "inspect",
        "io",
        "ipaddress",
        "itertools",
        "json",
        "keyword",
        "linecache",
        "locale",
        "logging",
        "lzma",
        "mailbox",
        "math",
        "mimetypes",
        "mmap",
        "multiprocessing",
        "netrc",
        "numbers",
        "operator",
        "os",
        "pathlib",
        "pickle",
        "platform",
        "plistlib",
        "poplib",
        "posix",
        "posixpath",
        "pprint",
        "profile",
        "pwd",
        "py_compile",
        "pyclbr",
        "pydoc",
        "queue",
        "quopri",
        "random",
        "re",
        "readline",
        "reprlib",
        "resource",
        "rlcompleter",
        "runpy",
        "sched",
        "secrets",
        "select",
        "selectors",
        "shelve",
        "shlex",
        "shutil",
        "signal",
        "smtplib",
        "socket",
        "socketserver",
        "sqlite3",
        "ssl",
        "stat",
        "statistics",
        "string",
        "stringprep",
        "struct",
        "subprocess",
        "sys",
        "sysconfig",
        "syslog",
        "tarfile",
        "telnetlib",
        "tempfile",
        "termios",
        "textwrap",
        "threading",
        "time",
        "timeit",
        "token",
        "tokenize",
        "trace",
        "traceback",
        "tracemalloc",
        "tty",
        "turtle",
        "types",
        "typing",
        "unicodedata",
        "unittest",
        "urllib",
        "uu",
        "uuid",
        "venv",
        "warnings",
        "wave",
        "weakref",
        "webbrowser",
        "winreg",
        "winsound",
        "wsgiref",
        "xdrlib",
        "xml",
        "xmlrpc",
        "zipfile",
        "zipimport",
        "zlib",
    }
)

THIRDPARTY_MODULES = frozenset(
    {
        "numpy",
        "np",
        "torch",
        "pytest",
        "pandas",
        "pd",
        "sklearn",
        "matplotlib",
        "tqdm",
        "transformers",
        "datasets",
        "scipy",
        "PIL",
        "cv2",
    }
)

TYPING_NAMES = frozenset(
    {
        "annotations",
        "Any",
        "Callable",
        "Dict",
        "List",
        "Literal",
        "Optional",
        "Sequence",
        "Tuple",
        "Type",
        "Union",
        "TYPE_CHECKING",
        "TypeVar",
        "Generic",
        "Enum",  # from enum module but commonly imported alongside typing
        "F",  # torch.nn.functional alias
    }
)

DATACLASS_NAMES = frozenset(
    {
        "dataclass",
        "field",
        "asdict",
        "astuple",
        "fields",
    }
)

EXCLUDED_NAMES = STDLIB_MODULES | THIRDPARTY_MODULES | TYPING_NAMES | DATACLASS_NAMES


# =============================================================================
# Helpers
# =============================================================================


def _is_module(obj: Any) -> bool:
    return isinstance(obj, type(sys))


def _should_export(name: str, obj: Any) -> bool:
    """Determine if a name should be exported."""
    if name.startswith("_"):
        return False
    if name in EXCLUDED_NAMES:
        return False
    if _is_module(obj):
        return False
    return True


def _import_safe(name: str, package: str) -> Any | None:
    """Import a module/package safely, returning None on failure."""
    try:
        return importlib.import_module(f".{name}", package=package)
    except ImportError:
        return None


def _get_public_names(module: Any) -> list[str]:
    """Get public names from a module."""
    if hasattr(module, "__all__"):
        return list(module.__all__)
    return [n for n in dir(module) if not n.startswith("_")]


# =============================================================================
# Core Export Functions
# =============================================================================


def _export_module_contents(
    module: Any,
    into: dict[str, Any],
) -> list[str]:
    """Export public names from a module into a dict.

    If a name is already in the dict (e.g., from `from .xxx import *`),
    it will still be included in the returned list for __all__.
    """
    exported = []
    for name in _get_public_names(module):
        obj = getattr(module, name)
        if _should_export(name, obj):
            if name not in into:
                into[name] = obj
            exported.append(name)
    return exported


def _find_modules(directory: Path) -> list[str]:
    """Find Python module names in a directory."""
    return [p.stem for p in sorted(directory.glob("*.py")) if p.name != "__init__.py"]


def _find_packages(directory: Path) -> list[str]:
    """Find subpackage names in a directory."""
    return [
        p.name
        for p in sorted(directory.iterdir())
        if p.is_dir() and (p / "__init__.py").exists() and not p.name.startswith("_")
    ]


# =============================================================================
# Main API
# =============================================================================


def auto_export(
    init_file: str,
    package_name: str,
    globals_dict: dict[str, Any],
) -> list[str]:
    """Auto-import modules and subpackages, exporting their public names.

    This function:
    1. Imports all .py modules in the package directory
    2. Imports all subpackages (directories with __init__.py)
    3. Re-exports public names from modules into the package namespace
    4. Makes subpackages available as attributes

    Args:
        init_file: __file__ from the calling __init__.py
        package_name: __name__ from the calling __init__.py
        globals_dict: globals() from the calling __init__.py

    Returns:
        List of exported names for __all__

    Example:
        # In __init__.py
        from src.common.auto_export import auto_export
        __all__ = auto_export(__file__, __name__, globals())

        # Now you can do:
        # from package import SomeClass  (from a module)
        # from package import subpackage  (a subpackage)
    """
    directory = Path(init_file).parent
    all_names: list[str] = []

    # 1. Export contents from .py modules
    for module_name in _find_modules(directory):
        module = _import_safe(module_name, package_name)
        if module is not None:
            exported = _export_module_contents(module, globals_dict)
            all_names.extend(exported)

    # 2. Import subpackages and re-export their public contents
    for pkg_name in _find_packages(directory):
        pkg = _import_safe(pkg_name, package_name)
        if pkg is not None:
            # Make subpackage available as attribute
            if pkg_name not in globals_dict:
                globals_dict[pkg_name] = pkg
                all_names.append(pkg_name)
            # Re-export subpackage contents for flat access
            exported = _export_module_contents(pkg, globals_dict)
            all_names.extend(exported)

    return all_names
