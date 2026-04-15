"""Result dataclasses for attention pattern analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ....common.base_schema import BaseSchema
from ....common.file_io import save_json
from .attn_head_attribution import HeadAttributionResults, HeadPositionPatchingResult, HeadSweepResults
from ..fine.fine_results import LayerPositionResult


@dataclass
class HeadAttnInfo(BaseSchema):
    """Information about a single attention head."""

    head_idx: int

    # Attention pattern metrics
    attn_to_source: float = 0.0  # Total attention to source positions (time_horizon)
    attn_to_dest: float = 0.0  # Self-attention to destination position
    attn_entropy: float = 0.0  # Entropy of attention distribution

    # Top attended positions
    top_attended_positions: list[int] = field(default_factory=list)
    top_attended_weights: list[float] = field(default_factory=list)
    top_attended_labels: list[str] = field(default_factory=list)  # format_pos labels

    # Output direction analysis
    logit_contribution: float = 0.0  # head_out @ logit_direction
    output_norm: float = 0.0  # ||head_out||

    # Clean vs corrupted comparison
    attn_pattern_diff: float = 0.0  # L2 norm of pattern difference
    attn_pattern_diff_l1: float = 0.0  # L1 norm of pattern difference
    attn_pattern_cosine: float = 0.0  # Cosine similarity (1.0 = identical)
    is_dynamic: bool = False  # Pattern changes between clean/corrupted


@dataclass
class AttnLayerResult(BaseSchema):
    """Results for a single attention layer."""

    layer: int
    n_heads: int = 0

    # Per-head results
    head_results: list[HeadAttnInfo] = field(default_factory=list)

    # Layer-level aggregates
    total_attn_to_source: float = 0.0  # Sum across heads
    mean_attn_to_source: float = 0.0
    n_source_attending_heads: int = 0  # Heads with attn_to_source > threshold

    def get_head_result(self, head_idx: int) -> HeadAttnInfo | None:
        """Get results for a specific head."""
        for h in self.head_results:
            if h.head_idx == head_idx:
                return h
        return None

    def get_top_heads_by_source_attention(self, n: int = 5) -> list[HeadAttnInfo]:
        """Get heads with highest attention to source positions."""
        sorted_heads = sorted(self.head_results, key=lambda h: h.attn_to_source, reverse=True)
        return sorted_heads[:n]

    def get_dynamic_heads(self) -> list[HeadAttnInfo]:
        """Get heads whose attention patterns change between conditions."""
        return [h for h in self.head_results if h.is_dynamic]

    @classmethod
    def from_dict(cls, d: dict) -> "AttnLayerResult":
        """Custom deserialization to handle nested HeadAttnInfo."""
        result = cls(
            layer=d.get("layer", 0),
            n_heads=d.get("n_heads", 0),
            total_attn_to_source=d.get("total_attn_to_source", 0.0),
            mean_attn_to_source=d.get("mean_attn_to_source", 0.0),
            n_source_attending_heads=d.get("n_source_attending_heads", 0),
        )
        result.head_results = [
            HeadAttnInfo.from_dict(hr) for hr in d.get("head_results", [])
        ]
        return result


@dataclass
class DstGroupAttention(BaseSchema):
    """Attention from one destination format_pos group, label-aligned across frames.

    All canonical labels (the union of named positions in clean+corrupted frames)
    form the columns. ``clean[layer]`` and ``corrupted[layer]`` are
    ``[n_heads, n_labels]`` arrays sliced from the corresponding frame at the
    canonical-label index — values are 0 where a label has no position in
    that frame. Both sides are MEAN attention across rel_pos within the
    destination group (so e.g. dst="format_content" averages all
    "format_content:0..N" positions).
    """

    dst_label: str = ""
    canonical_labels: list[str] = field(default_factory=list)
    dst_position_indices: list[int] = field(default_factory=list)
    clean: dict[int, list[list[float]]] = field(default_factory=dict)
    corrupted: dict[int, list[list[float]]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "DstGroupAttention":
        return cls(
            dst_label=d.get("dst_label", ""),
            canonical_labels=list(d.get("canonical_labels", [])),
            dst_position_indices=list(d.get("dst_position_indices", [])),
            clean={int(k): v for k, v in d.get("clean", {}).items()},
            corrupted={int(k): v for k, v in d.get("corrupted", {}).items()},
        )


@dataclass
class AttnPairResult(BaseSchema):
    """Attention analysis results for a single pair."""

    pair_idx: int = 0

    layer_results: list[AttnLayerResult] = field(default_factory=list)

    # Per-dst-format_pos attention. Built for EVERY format_pos that appears
    # in the position mapping (clean ∪ corrupted) — both sides treat all
    # format_pos as candidate destinations and as candidate sources.
    dst_group_attention: dict[str, DstGroupAttention] = field(default_factory=dict)

    # Head attribution results (causal importance of each head)
    head_attribution: "HeadAttributionResults | None" = None

    # Head redundancy results (denoising vs noising gap per head)
    head_redundancy: "HeadSweepResults | None" = None

    # Position patching results for top heads
    head_position_patching: list["HeadPositionPatchingResult"] = field(default_factory=list)

    # Layer-position patching result for attn_out
    layer_position: "LayerPositionResult | None" = None

    def get_layer_result(self, layer: int) -> AttnLayerResult | None:
        """Get results for a specific layer."""
        for r in self.layer_results:
            if r.layer == layer:
                return r
        return None

    def get_all_source_attending_heads(
        self, threshold: float = 0.1
    ) -> list[tuple[int, int, float]]:
        """Get all heads that attend to source positions above threshold."""
        heads = []
        for lr in self.layer_results:
            for hi in lr.head_results:
                if hi.attn_to_source >= threshold:
                    heads.append((lr.layer, hi.head_idx, hi.attn_to_source))
        return sorted(heads, key=lambda x: x[2], reverse=True)

    def pop_heavy(self) -> None:
        """Remove heavy attention pattern data to save memory."""
        self.dst_group_attention = {}

    @classmethod
    def from_dict(cls, d: dict) -> "AttnPairResult":
        """Custom deserialization to handle nested dataclasses."""
        result = cls(pair_idx=d.get("pair_idx", 0))
        result.dst_group_attention = {
            k: DstGroupAttention.from_dict(v)
            for k, v in (d.get("dst_group_attention", {}) or {}).items()
        }

        # Deserialize layer_results
        result.layer_results = [
            AttnLayerResult.from_dict(lr) for lr in d.get("layer_results", [])
        ]

        # Deserialize head_attribution
        if "head_attribution" in d and d["head_attribution"] is not None:
            result.head_attribution = HeadAttributionResults.from_dict(d["head_attribution"])

        # Deserialize head_redundancy
        if "head_redundancy" in d and d["head_redundancy"] is not None:
            result.head_redundancy = HeadSweepResults.from_dict(d["head_redundancy"])

        # Deserialize head_position_patching
        result.head_position_patching = [
            HeadPositionPatchingResult.from_dict(hpp)
            for hpp in d.get("head_position_patching", [])
        ]

        # Deserialize layer_position
        if "layer_position" in d and d["layer_position"] is not None:
            result.layer_position = LayerPositionResult.from_dict(d["layer_position"])

        return result


@dataclass
class AttnAggregatedResults(BaseSchema):
    """Aggregated attention analysis across all pairs."""

    pair_results: list[AttnPairResult] = field(default_factory=list)

    # Layers that the head_attribution / position_patching steps were configured to use.
    # The per-pair analysis itself sweeps ALL model layers regardless.
    layers_analyzed: list[int] = field(default_factory=list)

    @property
    def n_pairs(self) -> int:
        return len(self.pair_results)

    def add(self, result: AttnPairResult) -> None:
        """Add a pair result."""
        self.pair_results.append(result)

    def get_mean_source_attention(self, layer: int, head_idx: int) -> float:
        """Get mean attention to source for a specific head across pairs."""
        values = []
        for pr in self.pair_results:
            lr = pr.get_layer_result(layer)
            if lr:
                hi = lr.get_head_result(head_idx)
                if hi:
                    values.append(hi.attn_to_source)
        return float(np.mean(values)) if values else 0.0

    def get_consistent_source_heads(
        self, layer: int, min_attn: float = 0.1, min_pairs: int = 5
    ) -> list[tuple[int, float, int]]:
        """Find heads that consistently attend to source across pairs.

        Args:
            layer: Layer to analyze
            min_attn: Minimum attention to source to count
            min_pairs: Minimum number of pairs where head must attend

        Returns:
            List of (head_idx, mean_attn, n_pairs_attending)
        """
        head_attns: dict[int, list[float]] = {}

        for pr in self.pair_results:
            lr = pr.get_layer_result(layer)
            if not lr:
                continue
            for hi in lr.head_results:
                if hi.attn_to_source >= min_attn:
                    if hi.head_idx not in head_attns:
                        head_attns[hi.head_idx] = []
                    head_attns[hi.head_idx].append(hi.attn_to_source)

        results = []
        for head_idx, attns in head_attns.items():
            if len(attns) >= min_pairs:
                results.append((head_idx, float(np.mean(attns)), len(attns)))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_dynamic_heads_across_pairs(
        self, layer: int, min_pairs: int = 3
    ) -> list[tuple[int, int]]:
        """Find heads that are dynamic (attention changes) across multiple pairs.

        Returns:
            List of (head_idx, n_pairs_dynamic)
        """
        head_counts: dict[int, int] = {}

        for pr in self.pair_results:
            lr = pr.get_layer_result(layer)
            if not lr:
                continue
            for hi in lr.head_results:
                if hi.is_dynamic:
                    head_counts[hi.head_idx] = head_counts.get(hi.head_idx, 0) + 1

        return [
            (head_idx, count)
            for head_idx, count in sorted(head_counts.items(), key=lambda x: -x[1])
            if count >= min_pairs
        ]

    def save(self, output_dir: Path) -> None:
        """Save results to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.to_dict(), output_dir / "attn_analysis_agg.json")

    @classmethod
    def load(cls, output_dir: Path) -> "AttnAggregatedResults | None":
        """Load from JSON."""
        path = Path(output_dir) / "attn_analysis_agg.json"
        if path.exists():
            return cls.from_json(path)
        return None

    def filter_by_indices(self, indices: list[int]) -> "AttnAggregatedResults":
        """Create a filtered copy containing only the specified pair indices."""
        filtered = AttnAggregatedResults(layers_analyzed=self.layers_analyzed)
        for idx in indices:
            if idx < len(self.pair_results):
                filtered.pair_results.append(self.pair_results[idx])
        return filtered

    def print_summary(self) -> None:
        """Print summary of attention analysis."""
        print(f"[attn] Attention Analysis: {self.n_pairs} pairs, layers {self.layers_analyzed}")

        for layer in self.layers_analyzed:
            consistent = self.get_consistent_source_heads(layer, min_attn=0.1, min_pairs=3)
            dynamic = self.get_dynamic_heads_across_pairs(layer, min_pairs=3)
            print(f"  L{layer}: {len(consistent)} consistent source-attending heads, {len(dynamic)} dynamic heads")

            # Show top 3 source-attending heads
            for head_idx, mean_attn, n_pairs in consistent[:3]:
                print(f"    H{head_idx}: mean attn to source={mean_attn:.3f} ({n_pairs} pairs)")
