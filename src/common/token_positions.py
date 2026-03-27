"""Token position utilities for matching and mapping between sequences."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base_schema import BaseSchema
from .position_mapping_base import SamplePositionMappingBase
from .token_trajectory import TokenTrajectory


@dataclass
class PairPositionMapping(BaseSchema):
    """Mapping between positions in two token sequences.

    Maps source positions to destination positions, with metadata.

    Attributes:
        mapping: Dict mapping source position indices to destination position indices.
        src_len: Length of the source sequence.
        dst_len: Length of the destination sequence.
        anchors: List of (src_pos, dst_pos) tuples marking known correspondences.
            Parallel array with anchor_texts - anchors[i] corresponds to anchor_texts[i].
        anchor_texts: List of text markers used to find anchor positions.
            Parallel array with anchors - anchor_texts[i] is the text that was matched
            to find anchors[i].
        first_interesting_marker: Optional marker text indicating the first position
            of interest (e.g., where answer choices begin).
        src_tokens: Optional list of decoded tokens for source sequence.
        dst_tokens: Optional list of decoded tokens for destination sequence.
    """

    mapping: dict[int, int] = field(default_factory=dict)
    src_len: int = 0
    dst_len: int = 0
    anchors: list[tuple[int, int]] = field(default_factory=list)
    anchor_texts: list[str] = field(default_factory=list)
    first_interesting_marker: str | None = None
    src_tokens: list[str] = field(default_factory=list)
    dst_tokens: list[str] = field(default_factory=list)
    # Reverse mapping: dst -> src, ensures ALL dst positions have a mapping
    # (multiple dst positions may map to the same src when dst is longer)
    reverse_mapping: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dict with computed alignment info."""
        d = super().to_dict()
        # Add alignment info if tokens are available
        if self.src_tokens or self.dst_tokens:
            anchor_set = set(tuple(a) for a in self.anchors)
            alignment = []
            for src_pos in range(self.src_len):
                dst_pos = self.mapping.get(src_pos, src_pos)
                is_anchor = (src_pos, dst_pos) in anchor_set
                src_token = self.src_tokens[src_pos] if src_pos < len(self.src_tokens) else ""
                dst_token = self.dst_tokens[dst_pos] if dst_pos < len(self.dst_tokens) else ""
                alignment.append({
                    "src_pos": src_pos,
                    "dst_pos": dst_pos,
                    "src_token": src_token,
                    "dst_token": dst_token,
                    "is_anchor": is_anchor,
                    "is_interpolated": not is_anchor,
                })
            d["alignment"] = alignment
        return d

    @classmethod
    def from_lengths(cls, src_len: int, dst_len: int) -> "PairPositionMapping":
        """Create linear mapping from sequence lengths (no anchors)."""
        mapping = interpolate_positions([], src_len, dst_len)
        return cls(mapping=mapping, src_len=src_len, dst_len=dst_len, anchors=[])

    def src_to_dst(self, src_pos: int, default: int | None = None) -> int | None:
        """Get destination position for a source position.

        Args:
            src_pos: Source position index to look up.
            default: Value to return if src_pos is not in the mapping.
                If None (the default), returns src_pos itself (identity mapping).

        Returns:
            The destination position corresponding to src_pos, or the default
            value if src_pos is not found in the mapping.
        """
        if default is None:
            default = src_pos
        return self.mapping.get(src_pos, default)

    @property
    def inv_mapping_complete(self) -> dict[int, int]:
        """Get complete inverse mapping (dst -> src) with all dst positions covered.

        Uses reverse_mapping if available, otherwise builds from forward mapping.
        """
        if self.reverse_mapping:
            return dict(self.reverse_mapping)

        # Build from forward mapping
        inv = {}
        for src, dst in self.mapping.items():
            inv[dst] = src
        return inv

    def dst_to_src(self, dst_pos: int, default: int | None = None) -> int | None:
        """Get source position for a destination position (reverse lookup).

        Always returns a valid source position - never returns None for valid
        dst positions. If multiple dst positions map to the same src (when dst
        is longer than src), this returns that shared src position.

        Args:
            dst_pos: Destination position index to look up.
            default: Value to return if dst_pos is not found. If None, uses
                interpolation to find the nearest src position.

        Returns:
            The source position corresponding to dst_pos.
        """
        # Use inv_mapping_complete for lookup
        inv = self.inv_mapping_complete
        if dst_pos in inv:
            return inv[dst_pos]

        # Fall back to interpolation if not found
        if default is not None:
            return default

        # Interpolate based on position ratio
        if self.dst_len == 0:
            return 0
        ratio = dst_pos / self.dst_len
        return min(int(ratio * self.src_len), self.src_len - 1)

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
        """Length of the shorter sequence."""
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

    def dst_to_src_interpolated(self, dst_pos: int) -> int:
        """Get source position for a destination position, with interpolation.

        Unlike dst_to_src which returns None for unmapped positions, this method
        uses linear interpolation based on src_len and dst_len to estimate the
        corresponding source position.
        """
        # First try direct lookup
        for src, dst in self.mapping.items():
            if dst == dst_pos:
                return src

        # Fall back to linear interpolation
        if self.dst_len == 0:
            return 0
        ratio = dst_pos / self.dst_len
        return int(ratio * self.src_len)

    def switch(self) -> "PairPositionMapping":
        """Return a new PairPositionMapping with src and dst swapped.

        Inverts the mapping direction: if original maps clean→corrupted,
        switched maps corrupted→clean.

        The switched mapping uses:
        - Original reverse_mapping as the new forward mapping
        - Original forward mapping as the new reverse_mapping

        This ensures both directions always have complete coverage.

        Returns:
            A new PairPositionMapping with source and destination swapped.
        """
        switched_anchors = [(dst, src) for src, dst in self.anchors]

        # Use reverse_mapping as new forward mapping (ensures all positions covered)
        # If no reverse_mapping, build one from forward mapping
        if self.reverse_mapping:
            new_mapping = dict(self.reverse_mapping)
        else:
            new_mapping = self.inv()

        # Original forward mapping becomes new reverse_mapping
        new_reverse = dict(self.mapping)

        return PairPositionMapping(
            mapping=new_mapping,
            src_len=self.dst_len,
            dst_len=self.src_len,
            anchors=switched_anchors,
            anchor_texts=self.anchor_texts,
            first_interesting_marker=self.first_interesting_marker,
            src_tokens=self.dst_tokens,
            dst_tokens=self.src_tokens,
            reverse_mapping=new_reverse,
        )


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
) -> tuple[list[tuple[int, int]], list[str]]:
    """Find anchor points between two token sequences.

    Anchors are positions where we know the correspondence between sequences,
    based on matching text markers found in both.

    Args:
        src_tokens: Token strings from source sequence
        dst_tokens: Token strings from destination sequence
        anchor_texts: Text markers to find in both sequences (e.g., ["a)", "b)"])

    Returns:
        Tuple of (anchor_points, anchor_markers) where:
            - anchor_points: List of (src_pos, dst_pos) tuples, sorted by src_pos
            - anchor_markers: List of text markers that were matched, in same order
    """
    if not anchor_texts:
        return [], []

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
    spec: dict | int | str,
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
    specs: list[dict | int | str],
    tokens: list[str],
    prompt_len: int | None = None,
) -> list[ResolvedPosition]:
    """Resolve multiple position specs."""
    return [resolve_position(spec, tokens, prompt_len) for spec in specs]


def resolve_positions_with_info(
    specs: list[dict | int | str],
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
) -> PairPositionMapping:
    """Build mapping from source positions to destination positions.

    Uses semantic matching via anchor texts, then interpolation for unmatched.

    Args:
        tokenizer: Tokenizer for decoding token IDs
        src_traj: Source trajectory with token_ids
        dst_traj: Destination trajectory with token_ids
        anchor_texts: Text markers to find in both sequences for alignment

    Returns:
        PairPositionMapping with mapping dict and metadata
    """
    src_tokens = decode_token_ids(tokenizer, src_traj.token_ids)
    dst_tokens = decode_token_ids(tokenizer, dst_traj.token_ids)

    anchor_points, anchor_markers = find_anchor_points(
        src_tokens, dst_tokens, anchor_texts
    )
    mapping = interpolate_positions(
        anchor_points, src_traj.n_sequence, dst_traj.n_sequence
    )

    return PairPositionMapping(
        mapping=mapping,
        src_len=src_traj.n_sequence,
        dst_len=dst_traj.n_sequence,
        anchors=anchor_points,
        anchor_texts=anchor_markers,
        src_tokens=src_tokens,
        dst_tokens=dst_tokens,
    )


def build_position_mapping_from_sample_mappings(
    src_mapping: SamplePositionMappingBase,
    dst_mapping: SamplePositionMappingBase,
    src_tokens: list[str] | None = None,
    dst_tokens: list[str] | None = None,
) -> PairPositionMapping:
    """Build PairPositionMapping by aligning format_pos group boundaries.

    Instead of using text anchors, this aligns positions based on their
    semantic format_pos labels. Both the FIRST and LAST positions of each
    format_pos group become anchors, ensuring that regions don't bleed into
    adjacent sections that only exist in one sample.

    Algorithm:
    1. Group src and dst positions by format_pos name
    2. For each format_pos that exists in both:
       - Use the FIRST position as an anchor (start of region)
       - Use the LAST position as an anchor (end of region)
    3. Use interpolate_positions to fill gaps between anchors
    4. Build reverse mapping (dst->src) ensuring ALL dst positions are covered
       by mapping extra dst positions to the nearest src position

    This ensures that semantic regions are properly bounded. For example,
    if src has consider_tail[65-76] and dst has consider_tail[65-76] followed
    by constraint[77-94], anchoring on BOTH first and last prevents
    consider_tail positions from bleeding into constraint positions.

    Args:
        src_mapping: SamplePositionMapping for source sequence
        dst_mapping: SamplePositionMapping for destination sequence
        src_tokens: Optional decoded tokens for source (for visualization)
        dst_tokens: Optional decoded tokens for destination (for visualization)

    Returns:
        PairPositionMapping with format_pos-based alignment
    """
    src_len = src_mapping.full_len
    dst_len = dst_mapping.full_len

    # Get named_positions from both mappings
    src_named = src_mapping.named_positions
    dst_named = dst_mapping.named_positions

    # Find common format_pos names
    common_format_pos = set(src_named.keys()) & set(dst_named.keys())

    # Build anchors from BOTH first and last positions of each format_pos group
    anchors: list[tuple[int, int]] = []
    anchor_texts: list[str] = []

    for format_pos in common_format_pos:
        src_positions = sorted(src_named[format_pos])
        dst_positions = sorted(dst_named[format_pos])

        if not src_positions or not dst_positions:
            continue

        # Anchor on FIRST position (start of region)
        anchors.append((src_positions[0], dst_positions[0]))
        anchor_texts.append(format_pos)

        # Anchor on LAST position (end of region) if different from first
        if len(src_positions) > 1 or len(dst_positions) > 1:
            anchors.append((src_positions[-1], dst_positions[-1]))
            anchor_texts.append(f"{format_pos}_end")

    # Sort anchors by src position, deduplicate by src position
    combined = sorted(zip(anchors, anchor_texts), key=lambda x: x[0][0])

    # Deduplicate: if multiple anchors have the same src position, keep first
    seen_src_pos = set()
    deduped_anchors = []
    deduped_texts = []
    for anchor, text in combined:
        if anchor[0] not in seen_src_pos:
            seen_src_pos.add(anchor[0])
            deduped_anchors.append(anchor)
            deduped_texts.append(text)

    anchors = deduped_anchors
    anchor_texts = deduped_texts

    # Use interpolate_positions to fill gaps between anchors (src -> dst)
    mapping = interpolate_positions(anchors, src_len, dst_len)

    # Build reverse mapping (dst -> src) ensuring ALL dst positions are covered
    # For dst positions that don't have a direct src mapping, map to nearest src
    reverse_mapping = _build_complete_reverse_mapping(mapping, src_len, dst_len, anchors)

    return PairPositionMapping(
        mapping=mapping,
        src_len=src_len,
        dst_len=dst_len,
        anchors=anchors,
        anchor_texts=anchor_texts,
        src_tokens=src_tokens or [],
        dst_tokens=dst_tokens or [],
        reverse_mapping=reverse_mapping,
    )


def _build_complete_reverse_mapping(
    forward_mapping: dict[int, int],
    src_len: int,
    dst_len: int,
    anchors: list[tuple[int, int]],
) -> dict[int, int]:
    """Build reverse mapping (dst->src) ensuring ALL dst positions are covered.

    When dst has more positions than src (e.g., constraint section in dst only),
    multiple dst positions map to the same src position. We use the NEXT anchor's
    src position (mapping forward to the next semantic region).

    Args:
        forward_mapping: src -> dst mapping
        src_len: Length of source sequence
        dst_len: Length of destination sequence
        anchors: List of (src, dst) anchor pairs

    Returns:
        Dict mapping every dst position to a src position
    """
    # First, build direct reverse from forward mapping
    reverse = {}
    for src, dst in forward_mapping.items():
        # If multiple src map to same dst, keep the one with smallest src
        if dst not in reverse or src < reverse[dst]:
            reverse[dst] = src

    # Find dst positions that are not covered
    covered_dst = set(reverse.keys())
    all_dst = set(range(dst_len))
    uncovered_dst = all_dst - covered_dst

    if not uncovered_dst:
        return reverse

    # Sort anchors by dst position for boundary lookup
    sorted_anchors = sorted(anchors, key=lambda x: x[1])

    # For each uncovered dst position, find the PREVIOUS anchor and use its src
    for dst_pos in uncovered_dst:
        # Find the anchor just BEFORE this dst position
        prev_anchor_src = 0  # Default to first position
        for src, dst in sorted_anchors:
            if dst <= dst_pos:
                prev_anchor_src = src
            else:
                break
        reverse[dst_pos] = prev_anchor_src

    return reverse


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
