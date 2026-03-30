"""Table formatting utilities for logging."""

from __future__ import annotations

from .log_primitives import log
from .section_headers import log_divider
from .text_formatting import pad_left, pad_right


def log_table_header(columns: list[tuple[str, int, str]], indent_str: str = "  ", divider_width: int = 62) -> None:
    """Log a table header row.

    Args:
        columns: List of (label, width, align) where align is '<', '>', or '^'
    """
    parts = []
    for label, width, align in columns:
        if align == "<":
            parts.append(pad_right(label, width))
        elif align == ">":
            parts.append(pad_left(label, width))
        else:
            parts.append(label.center(width))
    log(indent_str + "  ".join(parts))
    log_divider(divider_width, indent_str)


def log_table_row(cells: list[tuple[str, int, str]], indent_str: str = "  ") -> None:
    """Log a table row.

    Args:
        cells: List of (value, width, align) where align is '<', '>', or '^'
    """
    parts = []
    for value, width, align in cells:
        if align == "<":
            parts.append(pad_right(value, width))
        elif align == ">":
            parts.append(pad_left(value, width))
        else:
            parts.append(value.center(width))
    log(indent_str + "  ".join(parts))
