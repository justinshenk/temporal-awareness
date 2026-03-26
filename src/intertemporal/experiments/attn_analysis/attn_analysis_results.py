"""Result dataclasses for attention pattern analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ....common.base_schema import BaseSchema
from ....common.file_io import save_json

__all__ = [
    "HeadAttnInfo",
    "AttnLayerResult",
    "AttnPairResult",
    "AttnAggregatedResults",
]


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


@dataclass
class AttnPairResult(BaseSchema):
    """Attention analysis results for a single pair."""

    pair_idx: int = 0
    dest_position: int = 0  # Primary destination position analyzed (corrupted frame)
    source_positions: list[int] = field(default_factory=list)  # Source positions in CORRUPTED frame
    source_positions_clean: list[int] = field(default_factory=list)  # Source positions in CLEAN frame
    source_position_names: list[str] = field(default_factory=list)  # Semantic names (e.g., ["time_horizon"])
    dest_position_names: list[str] = field(default_factory=list)  # Semantic names (e.g., ["response_choice"])

    layer_results: list[AttnLayerResult] = field(default_factory=list)

    # Full attention patterns for visualization (optional, can be heavy)
    # Shape per layer: [n_heads, seq_len] - attention from P_dest to all positions
    attention_patterns: dict[int, list[list[float]]] = field(default_factory=dict)

    # Corrupted attention patterns for side-by-side comparison (optional)
    # Same shape as attention_patterns: [n_heads, seq_len]
    corrupted_attention_patterns: dict[int, list[list[float]]] = field(default_factory=dict)

    def get_layer_result(self, layer: int) -> AttnLayerResult | None:
        """Get results for a specific layer."""
        for r in self.layer_results:
            if r.layer == layer:
                return r
        return None

    def get_all_source_attending_heads(
        self, threshold: float = 0.1
    ) -> list[tuple[int, int, float]]:
        """Get all heads that attend to source positions above threshold.

        Returns:
            List of (layer, head_idx, attn_to_source) tuples
        """
        heads = []
        for lr in self.layer_results:
            for hi in lr.head_results:
                if hi.attn_to_source >= threshold:
                    heads.append((lr.layer, hi.head_idx, hi.attn_to_source))
        return sorted(heads, key=lambda x: x[2], reverse=True)

    def pop_heavy(self) -> None:
        """Remove heavy data (attention patterns) to save memory."""
        self.attention_patterns = {}
        self.corrupted_attention_patterns = {}


@dataclass
class AttnAggregatedResults(BaseSchema):
    """Aggregated attention analysis across all pairs."""

    pair_results: list[AttnPairResult] = field(default_factory=list)

    # Configuration used
    layers_analyzed: list[int] = field(default_factory=list)
    source_positions: list[int] = field(default_factory=list)

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
