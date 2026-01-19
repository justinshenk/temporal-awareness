"""Formatting utilities."""

from __future__ import annotations

from typing import Optional


def determine_choice(
    chosen_label: Optional[str],
    short_term_label: str,
    long_term_label: str,
) -> str:
    """Determine if chosen label corresponds to short_term or long_term."""
    if chosen_label is None:
        return "unknown"
    chosen_lower = chosen_label.lower()
    if chosen_lower == short_term_label.lower():
        return "short_term"
    elif chosen_lower == long_term_label.lower():
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
        flip1 = label1[diff_start:diff_end1] if diff_end1 > diff_start else label1[diff_start]
        flip2 = label2[diff_start:diff_end2] if diff_end2 > diff_start else label2[diff_start]
    else:
        flip1 = label1[diff_start:diff_end]
        flip2 = label2[diff_start:diff_end]

    return flip1, flip2
