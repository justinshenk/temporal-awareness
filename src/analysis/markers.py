"""Section markers for prompt structure visualization.

Provides consistent section detection for activation patching and probe analysis.
Uses sample labels (not hardcoded) and format-aware positioning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import ModelRunner

# Colors for section markers in heatmaps
SECTION_COLORS = {
    "before_choices": "orange",
    "before_time_horizon": "cyan",
    "before_choice_output": "lime",
}


def find_section_markers(
    runner: "ModelRunner",
    text: str,
    short_term_label: str,
    long_term_label: str,
) -> dict[str, int]:
    """Find semantic section positions in tokenized prompt.

    Markers are placed RIGHT BEFORE each section:
    - before_choices: Right before options are presented (first label)
    - before_time_horizon: Right before CONSIDER section
    - before_choice_output: Right before model's choice (in response)

    Args:
        runner: ModelRunner instance
        text: Full prompt text (including response)
        short_term_label: Label for short-term option (from sample)
        long_term_label: Label for long-term option (from sample)

    Returns:
        Dict mapping section name -> token position index
    """
    markers = {}

    def text_pos_to_token_pos(text_pos: int) -> int:
        """Convert character position to token position."""
        prefix = text[:text_pos]
        prefix_formatted = runner._apply_chat_template(prefix)
        prefix_tokens = runner.tokenize(prefix_formatted)[0]
        return len(prefix_tokens)

    # Find first occurrence of each label (in options section)
    first_label_pos = None
    for label in [short_term_label, long_term_label]:
        pos = text.find(label)
        if pos >= 0:
            if first_label_pos is None or pos < first_label_pos:
                first_label_pos = pos

    if first_label_pos is not None:
        markers["before_choices"] = text_pos_to_token_pos(first_label_pos)

    # Find CONSIDER section (contains time_horizon)
    consider_pos = text.find("CONSIDER:")
    if consider_pos >= 0:
        markers["before_time_horizon"] = text_pos_to_token_pos(consider_pos)

    # Find choice output position - where model outputs its selection
    # Look for "I select:" (choice_prefix from format)
    choice_prefix_pos = text.find("I select:")
    if choice_prefix_pos >= 0:
        # Find the LAST occurrence (in response, not format instructions)
        last_pos = choice_prefix_pos
        while True:
            next_pos = text.find("I select:", last_pos + 1)
            if next_pos < 0:
                break
            last_pos = next_pos
        markers["before_choice_output"] = text_pos_to_token_pos(last_pos)

    return markers


def get_token_labels(runner: "ModelRunner", text: str) -> list[str]:
    """Get clean token labels for x-axis (words only, no position numbers).

    Args:
        runner: ModelRunner instance
        text: Input text

    Returns:
        List of token strings (cleaned for display)
    """
    formatted = runner._apply_chat_template(text)
    tokens = runner.tokenize(formatted)[0]
    labels = []
    for t in tokens:
        word = runner.tokenizer.decode([t]).strip()
        # Clean up for display
        word = word.replace("\n", "\\n")
        if len(word) > 10:
            word = word[:8] + ".."
        labels.append(word)
    return labels
