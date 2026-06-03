"""Content output utilities for structured data logging."""

from __future__ import annotations

from .log_primitives import log


def log_params(**kwargs) -> None:
    """Print parameters as indented key-value pairs."""
    for key, value in kwargs.items():
        log(f"  {key}: {value}")


def log_kv(key: str, value: str, indent_str: str = "  ") -> None:
    """Log a key-value pair."""
    log(f"{indent_str}{key}: {value}")


def log_items(header: str, items: list[str | list[str]], prefix: str = "", indent_str: str = "    ") -> None:
    """Log a list of items with optional bundling."""
    log(f"  {header}")
    for i, item in enumerate(items):
        label = f"[{prefix}{i + 1}]" if prefix else f"[{i + 1}]"
        if isinstance(item, list):
            log(f"{indent_str}{label} BUNDLED ({len(item)} items):")
            for sub in item:
                log(f"{indent_str}  • {sub}")
        else:
            log(f"{indent_str}{label} {item}")


def log_wrapped(text: str, indent_str: str = "  ", width: int = 78, gap: int = 0) -> None:
    """Log text with word wrapping."""
    words = text.split()
    line = indent_str
    first = True
    for word in words:
        if len(line) + len(word) + 1 > width:
            log(line, gap=gap if first else 0)
            first = False
            line = indent_str + word
        else:
            line = line + " " + word if line != indent_str else indent_str + word
    if line.strip():
        log(line, gap=gap if first else 0)
