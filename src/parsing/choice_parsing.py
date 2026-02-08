"""Choice parsing utilities for model outputs."""

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


def extract_flip_tokens(labels: tuple[str, str]) -> tuple[str, str]:
    """Extract distinguishing tokens from two labels (e.g., 'a' and 'b' from 'a)' and 'b)')."""
    label1, label2 = labels
    min_len = min(len(label1), len(label2))

    # Find first differing position
    diff_start = None
    for i in range(min_len):
        if label1[i] != label2[i]:
            diff_start = i
            break

    if diff_start is None:
        if len(label1) != len(label2):
            if len(label1) > len(label2):
                return label1[min_len:], ""
            return "", label2[min_len:]
        return label1, label2

    # Find last differing position
    diff_end = None
    for i in range(1, min_len + 1):
        if label1[-i] != label2[-i]:
            diff_end = len(label1) - i + 1
            break

    if diff_end is None:
        diff_end = len(label1)

    if len(label1) != len(label2):
        len_diff = abs(len(label1) - len(label2))
        if len(label1) > len(label2):
            diff_end1, diff_end2 = diff_end, diff_end - len_diff
        else:
            diff_end1, diff_end2 = diff_end - len_diff, diff_end
        flip1 = (
            label1[diff_start:diff_end1]
            if diff_end1 > diff_start
            else label1[diff_start]
        )
        flip2 = (
            label2[diff_start:diff_end2]
            if diff_end2 > diff_start
            else label2[diff_start]
        )
    else:
        flip1 = label1[diff_start:diff_end]
        flip2 = label2[diff_start:diff_end]

    return flip1, flip2
