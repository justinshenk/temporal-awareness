"""Token position utilities for matching and mapping between sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np

from .token_trajectory import TokenTrajectory


@dataclass
class PositionMapping:
    """Mapping between positions in two token sequences.

    Maps source positions to destination positions, with metadata.
    """

    mapping: dict[int, int] = field(default_factory=dict)
    src_len: int = 0
    dst_len: int = 0
    anchors: list[tuple[int, int]] = field(default_factory=list)
    anchor_texts: list[str] = field(default_factory=list)
    first_interesting_marker: str | None = None

    @classmethod
    def from_lengths(cls, src_len: int, dst_len: int) -> "PositionMapping":
        """Create linear mapping from sequence lengths (no anchors)."""
        mapping = interpolate_positions([], src_len, dst_len)
        return cls(mapping=mapping, src_len=src_len, dst_len=dst_len, anchors=[])

    def src_to_dst(self, src_pos: int, default: int | None = None) -> int | None:
        """Get destination position for a source position."""
        if default is None:
            default = src_pos
        return self.mapping.get(src_pos, default)

    def dst_to_src(self, dst_pos: int) -> int | None:
        """Get source position for a destination position (reverse lookup)."""
        for src, dst in self.mapping.items():
            if dst == dst_pos:
                return src
        return None

    def __getitem__(self, src_pos: int) -> int:
        """Get destination position for source position."""
        return self.mapping.get(src_pos, src_pos)

    def get(self, src_pos: int, default: int | None = None) -> int | None:
        """Alias for src_to_dst."""
        return self.src_to_dst(src_pos, default)

    def __iter__(self):
        """Iterate over (src_pos, dst_pos) pairs."""
        return iter(self.mapping.items())

    def items(self):
        """Return (src_pos, dst_pos) pairs like dict.items()."""
        return self.mapping.items()

    def __len__(self) -> int:
        return len(self.mapping)

    def __contains__(self, src_pos: int) -> bool:
        return src_pos in self.mapping

    @property
    def max_len(self) -> int:
        """Length of the longer sequence."""
        return max(self.src_len, self.dst_len)

    @property
    def min_len(self) -> int:
        """Length of the longer sequence."""
        return min(self.src_len, self.dst_len)

    @property
    def first_interesting_pos(self) -> int:
        if self.first_interesting_marker is None:
            return 0
        if self.first_interesting_marker not in self.anchor_texts:
            return 0
        idx = self.anchor_texts.index(self.first_interesting_marker)
        src_pos, dst_pos = self.anchors[idx]
        return min(src_pos, dst_pos)

    def inv(self) -> dict[int, int]:
        """Return inverse mapping (dst -> src)."""
        return {dst: src for src, dst in self.mapping.items()}


@dataclass
class ResolvedPositionInfo:
    """Resolved position info for visualization labels."""

    tokens: dict[int, str] = field(default_factory=dict)  # pos_idx -> token word
    indices: dict[int, int] = field(default_factory=dict)  # pos_idx -> sequence index


@dataclass
class ResolvedPosition:
    """Resolved token position with metadata."""

    index: int
    label: str
    found: bool = True


def search_text(tokens: list[str], text: str, last: bool = False) -> ResolvedPosition:
    """Search for text in token list.

    Args:
        tokens: List of token strings
        text: Text to search for
        last: If True, return LAST occurrence; if False, return first

    Returns:
        ResolvedPosition with index of found token
    """
    text_lower = text.lower().strip()
    text_base = text_lower.rstrip(":,.")
    label = f'"{text}"'

    matches = []

    # Exact match first (with and without punctuation)
    for i, tok in enumerate(tokens):
        tok_clean = tok.lower().strip()
        if text_lower == tok_clean or text_base == tok_clean.rstrip(":."):
            matches.append(i)

    # Substring match (base text without punctuation)
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


def find_label_positions(tokens: list[str], labels: list[str]) -> dict[str, int]:
    """Find positions of labels in tokenized text.

    Args:
        tokens: List of token strings
        labels: List of label strings to search for

    Returns:
        Dict mapping label -> token position index (first occurrence)
    """
    positions = {}
    for label in labels:
        resolved = search_text(tokens, label)
        if resolved.found:
            positions[label] = resolved.index
    return positions


def find_anchor_points(
    src_tokens: list[str],
    dst_tokens: list[str],
    anchor_texts: list[str] | None = None,
) -> list[tuple[int, int]]:
    """Find anchor points between two token sequences.

    Anchors are positions where we know the correspondence between sequences,
    based on matching text markers found in both.

    Args:
        src_tokens: Token strings from source sequence
        dst_tokens: Token strings from destination sequence
        anchor_texts: Text markers to find in both sequences (e.g., ["a)", "b)"])

    Returns:
        List of (src_pos, dst_pos) anchor tuples, sorted by src_pos
    """
    if not anchor_texts:
        return []

    # Dedupe while preserving order
    seen = set()
    unique_texts = [t for t in anchor_texts if not (t in seen or seen.add(t))]

    src_positions = find_label_positions(src_tokens, unique_texts)
    dst_positions = find_label_positions(dst_tokens, unique_texts)

    result = []
    result_texts = []
    for text in unique_texts:
        if text in src_positions and text in dst_positions:
            result.append((src_positions[text], dst_positions[text]))
            result_texts.append(text)

    combined = sorted(zip(result, result_texts), key=lambda x: x[0][0])
    result = [r for r, t in combined]
    result_texts = [t for r, t in combined]
    return result, result_texts


def interpolate_positions(
    anchors: list[tuple[int, int]],
    src_len: int,
    dst_len: int,
) -> dict[int, int]:
    """Interpolate position mapping between anchor points.

    Args:
        anchors: List of (src_pos, dst_pos) anchor tuples
        src_len: Length of source sequence
        dst_len: Length of destination sequence

    Returns:
        Dict mapping source position -> destination position
    """
    full_anchors = [(0, 0)] + anchors + [(src_len - 1, dst_len - 1)]

    mapping = {}
    for i in range(len(full_anchors) - 1):
        src_start, dst_start = full_anchors[i]
        src_end, dst_end = full_anchors[i + 1]
        src_range = src_end - src_start
        dst_range = dst_end - dst_start

        if src_range == 0:
            continue

        for src_pos in range(src_start, src_end + 1):
            t = (src_pos - src_start) / src_range if src_range > 0 else 0
            dst_pos = int(dst_start + t * dst_range)
            mapping[src_pos] = max(0, min(dst_pos, dst_len - 1))

    return mapping


def resolve_position(
    spec: Union[dict, int, str],
    tokens: list[str],
    prompt_len: int | None = None,
) -> ResolvedPosition:
    """Resolve token position spec to absolute index.

    Args:
        spec: Position specification (dict, int, or str)
        tokens: List of token strings
        prompt_len: Length of prompt portion (for prompt_end relative)

    Returns:
        ResolvedPosition with index, label, and found status

    Spec formats:
        - int: Absolute position
        - str: Text to search for in tokens
        - {"text": "..."}: Search for text (first occurrence)
        - {"text": "...", "last": True}: Search for text (last occurrence)
        - {"relative_to": "end", "offset": -1}: Relative to end
        - {"relative_to": "prompt_end", "offset": 0}: Relative to prompt end
    """
    seq_len = len(tokens)
    if prompt_len is None:
        prompt_len = seq_len

    # Absolute position
    if isinstance(spec, int):
        if 0 <= spec < seq_len:
            return ResolvedPosition(index=spec, label=f"pos_{spec}")
        return ResolvedPosition(index=-1, label=f"pos_{spec}", found=False)

    # String: text search
    if isinstance(spec, str):
        return search_text(tokens, spec)

    # Dict spec
    if isinstance(spec, dict):
        # Text search
        if "text" in spec:
            return search_text(tokens, spec["text"], last=spec.get("last", False))

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
                return ResolvedPosition(
                    index=-1, label=f"{rel}{offset:+d}", found=False
                )

            label = f"{rel}{offset:+d}"
            if 0 <= idx < seq_len:
                return ResolvedPosition(index=idx, label=label)
            return ResolvedPosition(index=-1, label=label, found=False)

    return ResolvedPosition(index=-1, label=str(spec), found=False)


def resolve_positions(
    specs: list[Union[dict, int, str]],
    tokens: list[str],
    prompt_len: int | None = None,
) -> list[ResolvedPosition]:
    """Resolve multiple position specs."""
    return [resolve_position(spec, tokens, prompt_len) for spec in specs]


def resolve_positions_with_info(
    specs: list[Union[dict, int, str]],
    tokens: list[str],
    prompt_len: int | None = None,
) -> tuple[list[ResolvedPosition], ResolvedPositionInfo]:
    """Resolve position specs and collect info for labels.

    Args:
        specs: Position specifications
        tokens: List of token strings
        prompt_len: Length of prompt portion

    Returns:
        Tuple of (resolved positions, position info for labels)
    """
    resolved = resolve_positions(specs, tokens, prompt_len)
    info = ResolvedPositionInfo()

    for i, pos in enumerate(resolved):
        info.indices[i] = pos.index
        if pos.found and 0 <= pos.index < len(tokens):
            tok = tokens[pos.index].strip()
            if len(tok) > 12:
                tok = tok[:10] + ".."
            info.tokens[i] = tok

    return resolved, info


def decode_token_ids(tokenizer, token_ids: list[int]) -> list[str]:
    """Decode token IDs to individual token strings.

    Args:
        tokenizer: Tokenizer with decode method
        token_ids: List of token IDs

    Returns:
        List of decoded token strings
    """
    return [tokenizer.decode([t]) for t in token_ids]


def build_position_mapping(
    tokenizer,
    src_traj: TokenTrajectory,
    dst_traj: TokenTrajectory,
    anchor_texts: list[str] | None = None,
) -> PositionMapping:
    """Build mapping from source positions to destination positions.

    Uses semantic matching via anchor texts, then interpolation for unmatched.

    Args:
        tokenizer: Tokenizer for decoding token IDs
        src_traj: Source trajectory with token_ids
        dst_traj: Destination trajectory with token_ids
        anchor_texts: Text markers to find in both sequences for alignment

    Returns:
        PositionMapping with mapping dict and metadata
    """
    src_tokens = decode_token_ids(tokenizer, src_traj.token_ids)
    dst_tokens = decode_token_ids(tokenizer, dst_traj.token_ids)

    anchor_points, anchor_markers = find_anchor_points(
        src_tokens, dst_tokens, anchor_texts
    )
    mapping = interpolate_positions(
        anchor_points, src_traj.n_sequence, dst_traj.n_sequence
    )

    return PositionMapping(
        mapping=mapping,
        src_len=src_traj.n_sequence,
        dst_len=dst_traj.n_sequence,
        anchors=anchor_points,
        anchor_texts=anchor_markers,
    )


def build_position_arrays(
    pos_mapping: dict[int, int], src_len: int, dst_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build position mapping arrays for vectorized indexing.

    Args:
        pos_mapping: Maps source positions to destination positions
        src_len: Length of source sequence
        dst_len: Length of destination sequence

    Returns:
        Tuple of (src_pos, dst_pos, valid_mask)
    """
    src_pos = np.arange(src_len)
    dst_pos = np.array([pos_mapping.get(p, p) for p in range(src_len)])
    valid = dst_pos < dst_len
    return src_pos, dst_pos, valid
