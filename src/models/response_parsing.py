"""Response parsing utilities for model outputs."""

from __future__ import annotations

import re


def parse_choice(
    response: str,
    short_label: str,
    long_label: str,
    choice_prefix: str,
) -> str:
    """
    Parse choice from model response.

    Looks for pattern: "<choice_prefix> <label>"
    Returns: "short_term", "long_term", or "unknown"
    """
    response_lower = response.lower().strip()
    prefix_lower = choice_prefix.lower()

    labels = [short_label, long_label]
    labels_stripped = [label.rstrip(".)") for label in labels]
    all_variants = set(label.lower() for label in labels + labels_stripped)
    labels_pattern = "|".join(
        re.escape(label) for label in sorted(all_variants, key=len, reverse=True)
    )

    pattern = rf"{re.escape(prefix_lower)}\s*({labels_pattern})"
    match = re.search(pattern, response_lower)

    if match:
        matched = match.group(1)
        if matched in (short_label.lower(), short_label.rstrip(".)").lower()):
            return "short_term"
        elif matched in (long_label.lower(), long_label.rstrip(".)").lower()):
            return "long_term"

    return "unknown"
