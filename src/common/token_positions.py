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
from functools import lru_cache
from typing import Optional, Union


@lru_cache(maxsize=1)
def _get_prompt_keywords() -> tuple[dict[str, str], frozenset[str]]:
    """Get keyword mappings from DefaultPromptFormat (cached).

    Returns tuple of (keyword_map, last_occurrence_keywords).
    Uses lru_cache to avoid re-computing on every call while avoiding
    mutable global state.
    """
    from ..formatting.configs import DefaultPromptFormat

    fmt = DefaultPromptFormat()
    return fmt.get_keyword_map(), frozenset(fmt.get_last_occurrence_keyword_names())


@dataclass
class ResolvedPositionInfo:
    """Resolved position info for a set of token positions.

    Stores the mapping from position index to resolved token information,
    used to generate informative axis labels in visualizations.
    """
    tokens: dict[int, str] = field(default_factory=dict)  # pos_idx -> token word
    indices: dict[int, int] = field(default_factory=dict)  # pos_idx -> sequence index



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
        - str: Keyword to search for (or keyword_map key)
        - {"text": "..."}: Search for text in tokens (first occurrence)
        - {"text": "...", "last": True}: Search for text (last occurrence)
        - {"relative_to": "end", "offset": -1}: Relative to end
        - {"relative_to": "prompt_end", "offset": 0}: Relative to prompt end
        - {"keyword": "consider"}: Use keyword_map mapping
    """
    keyword_map, last_occurrence_keywords = _get_prompt_keywords()

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
        keyword = spec.lower()
        if keyword in keyword_map:
            search_text = keyword_map[keyword]
            # Use last occurrence for keywords like choice_prefix (in response, not FORMAT)
            use_last = keyword in last_occurrence_keywords
        else:
            search_text = spec
            use_last = False
        return _search_text(tokens, search_text, last=use_last)

    # Dict spec
    if isinstance(spec, dict):
        # Keyword reference
        if "keyword" in spec:
            keyword = spec["keyword"].lower()
            if keyword in keyword_map:
                use_last = keyword in last_occurrence_keywords
                return _search_text(tokens, keyword_map[keyword], last=use_last)
            return ResolvedPosition(index=-1, label=f'"{keyword}"', found=False)

        # Text search (supports optional "last": True for last-occurrence matching)
        if "text" in spec:
            return _search_text(tokens, spec["text"], last=spec.get("last", False))

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


def _search_text(tokens: list[str], text: str, last: bool = False) -> ResolvedPosition:
    """Search for text in token list.

    Args:
        tokens: List of token strings
        text: Text to search for
        last: If True, return LAST occurrence; if False, return first

    Returns:
        ResolvedPosition with index of found token
    """
    text_lower = text.lower().strip()
    # Remove trailing punctuation for flexible matching
    text_base = text_lower.rstrip(":,.")
    label = f'"{text[:15]}..."' if len(text) > 15 else f'"{text}"'

    matches = []

    # Exact match first (with and without punctuation)
    for i, tok in enumerate(tokens):
        tok_clean = tok.lower().strip()
        if text_lower == tok_clean or text_base == tok_clean.rstrip(":."):
            matches.append(i)

    # Substring match (base text without punctuation)
    # Require minimum length to avoid matching punctuation-only tokens
    if not matches:
        for i, tok in enumerate(tokens):
            tok_clean = tok.lower().strip()
            tok_base = tok_clean.rstrip(":.")
            if len(tok_base) >= 2 and (text_base in tok_clean or tok_base in text_base):
                matches.append(i)

    if matches:
        idx = matches[-1] if last else matches[0]
        return ResolvedPosition(index=idx, label=label)

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


# =============================================================================
# Positions Schema (for cross-script interoperability)
# =============================================================================

import json
from pathlib import Path
from dataclasses import asdict


@dataclass
class PositionSpec:
    """A single position specification."""

    position: int  # Token position index
    token: str  # Token string at this position
    score: float  # Importance score (method-dependent)
    layer: Optional[int] = None  # Layer index (None = all layers)
    section: Optional[str] = None  # Section name (e.g., "choices", "time_horizon")

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove None values for cleaner JSON
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class PositionsFile:
    """Standardized positions file format."""

    model: str  # Model name/path
    method: str  # Method that generated this (e.g., "activation_patching", "attribution")
    positions: list[PositionSpec]

    # Optional metadata
    dataset_id: Optional[str] = None
    threshold: Optional[float] = None
    component: Optional[str] = None  # e.g., "resid_post", "attn_out"

    def to_dict(self) -> dict:
        d = {
            "model": self.model,
            "method": self.method,
            "positions": [p.to_dict() for p in self.positions],
        }
        if self.dataset_id:
            d["dataset_id"] = self.dataset_id
        if self.threshold is not None:
            d["threshold"] = self.threshold
        if self.component:
            d["component"] = self.component
        return d

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved positions: {path}")

    @classmethod
    def load(cls, path: Path) -> "PositionsFile":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)

        positions = [
            PositionSpec(
                position=p["position"],
                token=p["token"],
                score=p.get("score", p.get("recovery", 0.0)),  # Handle both keys
                layer=p.get("layer"),
                section=p.get("section"),
            )
            for p in data["positions"]
        ]

        return cls(
            model=data.get("model", "unknown"),
            method=data.get("method", "unknown"),
            positions=positions,
            dataset_id=data.get("dataset_id"),
            threshold=data.get("threshold"),
            component=data.get("component"),
        )

    def get_top_n(self, n: int) -> list[PositionSpec]:
        """Get top N positions by score."""
        sorted_pos = sorted(self.positions, key=lambda p: p.score, reverse=True)
        return sorted_pos[:n]
