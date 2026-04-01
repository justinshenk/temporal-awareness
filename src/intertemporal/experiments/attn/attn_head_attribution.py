"""Head attribution and position patching analysis.

Measures per-head causal importance using hook_z + W_O projection.
This is different from attention pattern analysis (where heads look)
- this measures how much each head contributes to the final answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from ....common.base_schema import BaseSchema
from ....common.device_utils import clear_gpu_memory
from ....common.logging import log
from ....common.profiler import profile
from ....inference.interventions import InterventionTarget
from ....activation_patching.patch_choice import patch_for_choice

if TYPE_CHECKING:
    from ....binary_choice import BinaryChoiceRunner
    from ....common.contrastive_pair import ContrastivePair


# =============================================================================
# Result Classes
# =============================================================================


@dataclass
class HeadAttributionResult(BaseSchema):
    """Result for a single attention head's causal contribution.

    Attributes:
        layer: Layer index
        head: Head index
        attribution_score: Raw attribution score (can be positive or negative)
        abs_score: Absolute attribution score (importance magnitude)
    """

    layer: int
    head: int
    attribution_score: float = 0.0
    abs_score: float = 0.0

    @property
    def label(self) -> str:
        """Human-readable label for this head."""
        return f"L{self.layer}.H{self.head}"


@dataclass
class HeadAttributionResults(BaseSchema):
    """Results from head attribution sweep.

    Attributes:
        n_layers: Number of layers in model
        n_heads: Number of heads per layer
        layers_analyzed: Which layers were analyzed
        results: List of HeadAttributionResult for each (layer, head)
        attribution_matrix: [n_layers, n_heads] array of attribution scores
    """

    n_layers: int = 0
    n_heads: int = 0
    layers_analyzed: list[int] = field(default_factory=list)
    results: list[HeadAttributionResult] = field(default_factory=list)
    attribution_matrix: np.ndarray | None = None

    def get_top_heads(self, n: int = 20) -> list[HeadAttributionResult]:
        """Get top N heads by absolute attribution score."""
        return sorted(self.results, key=lambda h: h.abs_score, reverse=True)[:n]

    def build_matrix(self) -> None:
        """Build attribution_matrix from results."""
        if not self.results or not self.layers_analyzed:
            return

        layer_to_idx = {layer: i for i, layer in enumerate(self.layers_analyzed)}
        matrix = np.zeros((len(self.layers_analyzed), self.n_heads))

        for r in self.results:
            if r.layer in layer_to_idx:
                matrix[layer_to_idx[r.layer], r.head] = r.attribution_score

        self.attribution_matrix = matrix


@dataclass
class HeadPositionPatchingResult(BaseSchema):
    """Position patching results for a single head.

    Attributes:
        layer: Layer index
        head: Head index
        positions: List of positions that were patched
        position_names: Semantic names for positions (if available)
        effects: Effect at each position (recovery score)
    """

    layer: int
    head: int
    positions: list[int] = field(default_factory=list)
    position_names: list[str] = field(default_factory=list)
    effects: list[float] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"L{self.layer}.H{self.head}"

    def get_top_positions(self, n: int = 3) -> list[tuple[int, float, str]]:
        """Get top N positions by effect magnitude."""
        indexed = []
        for i, (pos, effect) in enumerate(zip(self.positions, self.effects)):
            name = self.position_names[i] if i < len(self.position_names) else f"P{pos}"
            indexed.append((pos, effect, name))
        return sorted(indexed, key=lambda x: abs(x[1]), reverse=True)[:n]


# =============================================================================
# Analysis Functions
# =============================================================================


def _get_attention_hook_filter(layers: list[int]) -> callable:
    """Create filter for attention-related hooks at specific layers."""
    hooks = set()
    for layer in layers:
        hooks.add(f"blocks.{layer}.attn.hook_z")
    return lambda name: name in hooks


@profile
def run_head_attribution(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
    layers: list[int] | None = None,
) -> HeadAttributionResults:
    """Run per-head attribution using hook_z + W_O projection.

    Computes per-head importance by:
    1. Getting hook_z (before O projection) with shape [batch, seq, n_heads, d_head]
    2. Computing per-head contributions via z @ W_O
    3. Projecting onto logit direction

    Args:
        runner: Model runner
        pair: Contrastive pair
        layers: Layers to analyze (default: second half of network)

    Returns:
        HeadAttributionResults with per-head attribution scores
    """
    n_heads = runner._backend.get_n_heads()
    n_layers = runner.n_layers

    if layers is None:
        layers = list(range(n_layers // 2, n_layers))

    names_filter = _get_attention_hook_filter(layers)

    # Run clean trajectory with cache
    clean_choice = runner.choose(
        pair.clean_prompt,
        pair.choice_prefix,
        pair.clean_labels,
        with_cache=True,
        names_filter=names_filter,
    )
    clean_cache = clean_choice.cache

    # Run corrupted trajectory with cache
    corrupted_choice = runner.choose(
        pair.corrupted_prompt,
        pair.choice_prefix,
        pair.corrupted_labels,
        with_cache=True,
        names_filter=names_filter,
    )
    corrupted_cache = corrupted_choice.cache

    # Get divergent position for metric computation
    clean_div_pos, corrupted_div_pos = pair.choice_divergent_positions

    # Get logit direction (difference between choice A and B logits)
    W_U = runner.W_U
    labels = pair.clean_labels
    label_a_id = runner.encode_ids(labels[0], add_special_tokens=False)[0]
    label_b_id = runner.encode_ids(labels[1], add_special_tokens=False)[0]
    logit_direction = W_U[:, label_a_id] - W_U[:, label_b_id]
    logit_direction = logit_direction / torch.norm(logit_direction)

    results = HeadAttributionResults(
        n_layers=n_layers,
        n_heads=n_heads,
        layers_analyzed=layers,
    )

    total = len(layers) * n_heads
    count = 0

    with torch.no_grad():
        for layer in layers:
            hook_z_name = f"blocks.{layer}.attn.hook_z"

            clean_z = clean_cache.get(hook_z_name)
            corrupted_z = corrupted_cache.get(hook_z_name)

            if clean_z is None or corrupted_z is None:
                log(f"[attn][head_attrib] Warning: hook_z not available for layer {layer}")
                for head in range(n_heads):
                    count += 1
                    results.results.append(HeadAttributionResult(
                        layer=layer,
                        head=head,
                        attribution_score=0.0,
                        abs_score=0.0,
                    ))
                continue

            if count == 0:
                log(f"[attn][head_attrib] hook_z shape: {clean_z.shape}")

            # Get W_O: [n_heads, d_head, d_model]
            W_O = runner._backend.get_W_O(layer)

            # Get z at metric positions
            clean_seq_len = clean_z.shape[1]
            corrupted_seq_len = corrupted_z.shape[1]
            clean_metric_pos = min(clean_div_pos - 1, clean_seq_len - 1)
            corrupted_metric_pos = min(corrupted_div_pos - 1, corrupted_seq_len - 1)

            clean_z_at_pos = clean_z[0, clean_metric_pos, :, :].clone()
            corrupted_z_at_pos = corrupted_z[0, corrupted_metric_pos, :, :].clone()

            # Compute per-head contributions
            clean_contrib = torch.einsum("hd,hdm->hm", clean_z_at_pos, W_O)
            corrupted_contrib = torch.einsum("hd,hdm->hm", corrupted_z_at_pos, W_O)

            for head in range(n_heads):
                count += 1
                if count % 32 == 0 or count == total:
                    log(f"[attn][head_attrib] {count}/{total}")

                diff = clean_contrib[head] - corrupted_contrib[head]
                score = torch.dot(diff, logit_direction).item()

                results.results.append(HeadAttributionResult(
                    layer=layer,
                    head=head,
                    attribution_score=score,
                    abs_score=abs(score),
                ))

    results.build_matrix()

    # Cleanup
    clean_choice.pop_heavy()
    corrupted_choice.pop_heavy()
    clear_gpu_memory()

    return results


@dataclass
class HeadSweepResult(BaseSchema):
    """Per-head patching result comparing denoising vs noising.

    Attributes:
        layer: Layer index
        head: Head index
        denoising_recovery: Recovery when patching clean → corrupted
        noising_disruption: Disruption when patching corrupted → clean
    """

    layer: int
    head: int
    denoising_recovery: float = 0.0
    noising_disruption: float = 0.0

    @property
    def gap(self) -> float:
        """Redundancy gap: denoising - noising."""
        return self.denoising_recovery - self.noising_disruption

    @property
    def combined_score(self) -> float:
        """Combined importance score (average of both modes)."""
        return (self.denoising_recovery + self.noising_disruption) / 2

    @property
    def label(self) -> str:
        return f"L{self.layer}.H{self.head}"

    @property
    def is_bottleneck(self) -> bool:
        """True if head carries unique information (small gap)."""
        return abs(self.gap) < 0.5

    @property
    def is_redundant(self) -> bool:
        """True if head has backup pathways (large negative gap)."""
        return self.gap < -0.5


@dataclass
class HeadSweepResults(BaseSchema):
    """Aggregated head redundancy analysis results."""

    n_layers: int = 0
    n_heads: int = 0
    layers_analyzed: list[int] = field(default_factory=list)
    results: list[HeadSweepResult] = field(default_factory=list)

    def get_sorted_by_gap(self, descending: bool = True) -> list[HeadSweepResult]:
        """Get heads sorted by absolute gap magnitude."""
        return sorted(self.results, key=lambda h: abs(h.gap), reverse=descending)

    def get_bottleneck_heads(self, threshold: float = 0.5) -> list[HeadSweepResult]:
        """Get heads that are bottlenecks (small gap = critical)."""
        return [h for h in self.results if abs(h.gap) < threshold]

    def get_redundant_heads(self, threshold: float = -0.5) -> list[HeadSweepResult]:
        """Get heads that are redundant (large negative gap)."""
        return [h for h in self.results if h.gap < threshold]


@profile
def run_head_patching_sweep(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
    layers: list[int] | None = None,
    top_n: int | None = None,
) -> HeadSweepResults:
    """Run per-head redundancy analysis comparing denoising vs noising.

    For each head, runs activation patching in both modes:
    - Denoising: patch clean values into corrupted run → recovery
    - Noising: patch corrupted values into clean run → disruption

    Gap = denoising - noising:
    - Small gap (close to 0) = bottleneck, head carries unique critical info
    - Large positive gap = head has unique info but is not critical
    - Large negative gap = redundant, other pathways compensate

    Args:
        runner: Model runner
        pair: Contrastive pair
        layers: Layers to analyze (default: second half of network)
        top_n: If set, only analyze top N heads by attribution

    Returns:
        HeadSweepResults with per-head denoising/noising scores and gaps
    """
    n_heads = runner._backend.get_n_heads()
    n_layers = runner.n_layers

    if layers is None:
        layers = list(range(n_layers // 2, n_layers))

    results = HeadSweepResults(
        n_layers=n_layers,
        n_heads=n_heads,
        layers_analyzed=layers,
    )

    # If top_n specified, first run attribution to find important heads
    heads_to_analyze: list[tuple[int, int]] = []
    if top_n is not None:
        attrib = run_head_attribution(runner, pair, layers)
        top_heads = attrib.get_top_heads(top_n)
        heads_to_analyze = [(h.layer, h.head) for h in top_heads]
    else:
        for layer in layers:
            for head in range(n_heads):
                heads_to_analyze.append((layer, head))

    total = len(heads_to_analyze)
    log(f"[attn][redundancy] Analyzing {total} heads...")

    for count, (layer, head) in enumerate(heads_to_analyze):
        if (count + 1) % 20 == 0 or count + 1 == total:
            log(f"[attn][redundancy] {count + 1}/{total}")

        target = InterventionTarget.at_head(layer=layer, head=head)

        # Run both modes
        dn = patch_for_choice(runner, pair, target, "denoising", clear_memory=True)
        ns = patch_for_choice(runner, pair, target, "noising", clear_memory=True)

        results.results.append(HeadSweepResult(
            layer=layer,
            head=head,
            denoising_recovery=dn.recovery,
            noising_disruption=ns.disruption,
        ))

    clear_gpu_memory()
    return results


@profile
def run_head_position_patching(
    runner: "BinaryChoiceRunner",
    pair: "ContrastivePair",
    top_heads: list[HeadAttributionResult],
    positions: list[int],
    position_names: list[str] | None = None,
    n_heads: int = 10,
) -> list[HeadPositionPatchingResult]:
    """Run position-level patching for top heads.

    For each top head, patches at each position independently
    to find which positions are most important for that head's effect.

    Args:
        runner: Model runner
        pair: Contrastive pair
        top_heads: List of top heads from attribution analysis
        positions: Absolute positions to patch
        position_names: Semantic names for positions (optional)
        n_heads: Number of top heads to analyze

    Returns:
        List of HeadPositionPatchingResult, one per head
    """
    results = []

    if not positions:
        log("[attn][pos_patch] No positions to patch")
        return results

    if position_names is None:
        position_names = [f"P{p}" for p in positions]

    for head_result in top_heads[:n_heads]:
        layer = head_result.layer
        head = head_result.head

        log(f"[attn][pos_patch] {head_result.label}")

        pos_result = HeadPositionPatchingResult(
            layer=layer,
            head=head,
            positions=positions,
            position_names=position_names,
        )

        for pos in positions:
            target = InterventionTarget.at_head(
                layer=layer,
                head=head,
                positions=[pos],
            )

            dn = patch_for_choice(runner, pair, target, "denoising", clear_memory=True)
            pos_result.effects.append(dn.recovery)

        results.append(pos_result)

    clear_gpu_memory()
    return results
