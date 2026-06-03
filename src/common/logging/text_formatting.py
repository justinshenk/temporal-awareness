"""Text formatting utilities for logging."""

from __future__ import annotations

import re

DEFAULT_WIDTH = 76


def center(text: str, width: int = DEFAULT_WIDTH, fill: str = " ") -> str:
    """Center text within a given width."""
    return text.center(width, fill)


def pad_left(text: str, width: int, fill: str = " ") -> str:
    """Right-align text (pad left)."""
    return text.rjust(width, fill)


def pad_right(text: str, width: int, fill: str = " ") -> str:
    """Left-align text (pad right)."""
    return text.ljust(width, fill)


def indent(text: str, spaces: int = 2) -> str:
    """Add indentation to text."""
    return " " * spaces + text


def fmt_prob(p: float, width: int = 10) -> str:
    """Format probability, using scientific notation for very small values."""
    if p < 0.0001:
        return f"{p:>{width}.1e}"
    return f"{p:>{width}.4f}"


def oneline(text: str) -> str:
    """Collapse whitespace to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def preview(text: str, max_len: int = 50) -> str:
    """Truncate text for preview display."""
    text = oneline(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
