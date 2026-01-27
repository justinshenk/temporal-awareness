"""Token position resolution for activation analysis.

This module provides utilities for:
- Specifying token positions via keywords, text search, or relative offsets
- Resolving position specs to actual sequence indices
- Building display labels that show both the spec and resolved token

Display label convention:
- Labels combine the position spec with the actual token found
- Format: "{spec_label}\\n{token}" e.g. "[situation]\\nThe" or "end-1\\nselect"
- This helps visualization show what we're looking for AND what was found
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class ResolvedPositionInfo:
    """Resolved position info for a set of token positions.

    Stores the mapping from position index to resolved token information,
    used to generate informative axis labels in visualizations.
    """
    tokens: dict[int, str] = field(default_factory=dict)  # pos_idx -> token word
    indices: dict[int, int] = field(default_factory=dict)  # pos_idx -> sequence index


# Standard prompt format keywords for position references
PROMPT_KEYWORDS = {
    "situation": "SITUATION:",
    "task": "TASK:",
    "option_one": "OPTION_ONE:",
    "option_two": "OPTION_TWO:",
    "consider": "CONSIDER:",
    "action": "ACTION:",
    "format": "FORMAT:",
    "choice_prefix": "I select:",
    "reasoning_prefix": "My reasoning:",
}



@dataclass
class TokenPositionSpec:
    """Token position specification."""

    spec: Union[dict, int, str]

    @classmethod
    def from_dict(cls, data: Union[dict, int, str]) -> "TokenPositionSpec":
        return cls(spec=data)


@dataclass
class ResolvedPosition:
    """Resolved token position with metadata."""

    index: int
    label: str
    found: bool = True


def resolve_position(
    spec: Union[TokenPositionSpec, dict, int, str],
    tokens: list[str],
    prompt_len: Optional[int] = None,
) -> ResolvedPosition:
    """
    Resolve token position spec to absolute index.

    Args:
        spec: Position specification (TokenPositionSpec, dict, int, or str)
        tokens: List of token strings
        prompt_len: Length of prompt portion (for prompt_end relative)

    Returns:
        ResolvedPosition with index, label, and found status

    Spec formats:
        - int: Absolute position
        - str: Keyword to search for (or PROMPT_KEYWORDS key)
        - {"text": "..."}: Search for text in tokens
        - {"relative_to": "end", "offset": -1}: Relative to end
        - {"relative_to": "prompt_end", "offset": 0}: Relative to prompt end
        - {"keyword": "consider"}: Use PROMPT_KEYWORDS mapping
    """
    if isinstance(spec, TokenPositionSpec):
        spec = spec.spec

    seq_len = len(tokens)
    if prompt_len is None:
        prompt_len = seq_len

    # Absolute position
    if isinstance(spec, int):
        if 0 <= spec < seq_len:
            return ResolvedPosition(index=spec, label=f"pos_{spec}")
        return ResolvedPosition(index=-1, label=f"pos_{spec}", found=False)

    # String: keyword or text search
    if isinstance(spec, str):
        # Check if it's a keyword reference
        if spec.lower() in PROMPT_KEYWORDS:
            search_text = PROMPT_KEYWORDS[spec.lower()]
        else:
            search_text = spec
        return _search_text(tokens, search_text)

    # Dict spec
    if isinstance(spec, dict):
        # Keyword reference
        if "keyword" in spec:
            keyword = spec["keyword"].lower()
            if keyword in PROMPT_KEYWORDS:
                return _search_text(tokens, PROMPT_KEYWORDS[keyword])
            return ResolvedPosition(index=-1, label=f'"{keyword}"', found=False)

        # Text search
        if "text" in spec:
            return _search_text(tokens, spec["text"])

        # Relative position
        if "relative_to" in spec:
            offset = spec.get("offset", 0)
            rel = spec["relative_to"]

            if rel == "end":
                idx = seq_len + offset
            elif rel == "prompt_end":
                idx = prompt_len + offset
            elif rel == "start":
                idx = offset
            else:
                return ResolvedPosition(index=-1, label=f"{rel}{offset:+d}", found=False)

            label = f"{rel}{offset:+d}"
            if 0 <= idx < seq_len:
                return ResolvedPosition(index=idx, label=label)
            return ResolvedPosition(index=-1, label=label, found=False)

    return ResolvedPosition(index=-1, label=str(spec), found=False)


def _search_text(tokens: list[str], text: str) -> ResolvedPosition:
    """Search for text in token list."""
    text_lower = text.lower().strip()
    # Remove trailing punctuation for flexible matching
    text_base = text_lower.rstrip(":,.")
    label = f'"{text[:15]}..."' if len(text) > 15 else f'"{text}"'

    # Exact match first (with and without punctuation)
    for i, tok in enumerate(tokens):
        tok_clean = tok.lower().strip()
        if text_lower == tok_clean or text_base == tok_clean.rstrip(":."):
            return ResolvedPosition(index=i, label=label)

    # Substring match (base text without punctuation)
    # Require minimum length to avoid matching punctuation-only tokens
    for i, tok in enumerate(tokens):
        tok_clean = tok.lower().strip()
        tok_base = tok_clean.rstrip(":.")
        if len(tok_base) >= 2 and text_base in tok_clean:
            return ResolvedPosition(index=i, label=label)
        if len(tok_base) >= 2 and tok_base in text_base:
            return ResolvedPosition(index=i, label=label)

    return ResolvedPosition(index=-1, label=label, found=False)


def resolve_positions(
    specs: list[Union[TokenPositionSpec, dict, int, str]],
    tokens: list[str],
    prompt_len: Optional[int] = None,
) -> list[ResolvedPosition]:
    """Resolve multiple position specs."""
    return [resolve_position(spec, tokens, prompt_len) for spec in specs]


def get_position_label(spec: Union[TokenPositionSpec, dict, int, str]) -> str:
    """Get display label for position spec without resolving."""
    if isinstance(spec, TokenPositionSpec):
        spec = spec.spec

    if isinstance(spec, int):
        return f"pos {spec}"

    if isinstance(spec, str):
        # Clean up keyword names for display
        name = spec.replace("_", " ").title()
        return name[:12] if len(name) > 12 else name

    if isinstance(spec, dict):
        if "keyword" in spec:
            name = spec["keyword"].replace("_", " ").title()
            return name[:12] if len(name) > 12 else name
        if "text" in spec:
            text = spec["text"]
            return text[:10] + ".." if len(text) > 10 else text
        if "relative_to" in spec:
            rel = spec["relative_to"]
            offset = spec.get("offset", 0)
            return f"{rel}{offset:+d}"

    return str(spec)


def build_position_labels(
    specs: list[Union[TokenPositionSpec, dict, int, str]],
    position_info: ResolvedPositionInfo,
) -> list[str]:
    """Build display labels combining spec label with resolved token word.

    Format: "{spec_label}\\n{token}" e.g. "[situation]\\nThe" or "end-1\\nselect"
    This shows both what we're looking for and what token was actually found.

    Args:
        specs: Position specifications
        position_info: Resolved position info with tokens

    Returns:
        List of display labels for visualization axes
    """
    labels = []
    for i, spec in enumerate(specs):
        spec_label = get_position_label(spec)
        token = position_info.tokens.get(i, "")
        if token:
            labels.append(f"{spec_label}\n{token}")
        else:
            labels.append(spec_label)
    return labels


def resolve_positions_with_info(
    specs: list[Union[TokenPositionSpec, dict, int, str]],
    tokens: list[str],
    prompt_len: Optional[int] = None,
) -> tuple[list[ResolvedPosition], ResolvedPositionInfo]:
    """Resolve position specs and collect info for labels.

    Args:
        specs: Position specifications
        tokens: List of token strings
        prompt_len: Length of prompt portion (for prompt_end relative)

    Returns:
        Tuple of (resolved positions, position info for labels)
    """
    resolved = resolve_positions(specs, tokens, prompt_len)
    info = ResolvedPositionInfo()

    for i, pos in enumerate(resolved):
        info.indices[i] = pos.index
        if pos.found and 0 <= pos.index < len(tokens):
            # Clean up token for display
            tok = tokens[pos.index].strip()
            if len(tok) > 12:
                tok = tok[:10] + ".."
            info.tokens[i] = tok

    return resolved, info


def find_label_positions(
    tokens: list[str],
    labels: list[str],
) -> dict[str, int]:
    """Find positions of labels in tokenized text.

    Args:
        tokens: List of token strings
        labels: List of label strings to search for (e.g., from sample.short_term_label)

    Returns:
        Dict mapping label -> token position index (first occurrence)
    """
    positions = {}
    for label in labels:
        resolved = _search_text(tokens, label)
        if resolved.found:
            positions[label] = resolved.index
    return positions
