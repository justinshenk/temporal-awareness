"""Method agreement analysis for attribution patching.

Computes overlap (Jaccard similarity) between top attribution cells across methods.
If overlap is >70%, methods agree on what matters.
If overlap is <50%, methods tell genuinely different stories.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ....common.base_schema import BaseSchema
from ....attribution_patching import AttrPatchAggregatedResults, AttributionSummary
from ....attribution_patching.attribution_key import AttributionKey


@dataclass
class MethodPairAgreement(BaseSchema):
    """Agreement between two attribution methods.

    Attributes:
        method_a: First method name (e.g., "standard/resid_post")
        method_b: Second method name
        jaccard: Jaccard similarity (intersection / union)
        overlap_count: Number of shared top cells
        total_union: Size of union of top cells
        top_k: Number of top cells used for comparison
    """

    method_a: str
    method_b: str
    jaccard: float
    overlap_count: int
    total_union: int
    top_k: int

    @property
    def agreement_level(self) -> str:
        """Human-readable agreement level."""
        if self.jaccard >= 0.7:
            return "high"
        if self.jaccard >= 0.5:
            return "moderate"
        return "low"


@dataclass
class MethodAgreementResults(BaseSchema):
    """Complete method agreement analysis results.

    Attributes:
        pair_agreements: Pairwise agreement between methods
        mean_jaccard: Mean Jaccard similarity across all pairs
        methods_analyzed: List of method keys analyzed
        top_k: Number of top cells used
        mode: "denoising" or "noising"
    """

    pair_agreements: list[MethodPairAgreement] = field(default_factory=list)
    mean_jaccard: float = 0.0
    methods_analyzed: list[str] = field(default_factory=list)
    top_k: int = 20
    mode: str = ""

    @property
    def overall_agreement(self) -> str:
        """Overall agreement level across all method pairs."""
        if self.mean_jaccard >= 0.7:
            return "high"
        if self.mean_jaccard >= 0.5:
            return "moderate"
        return "low"

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print(f"Method Agreement ({self.mode}): {self.overall_agreement} (mean Jaccard={self.mean_jaccard:.3f})")
        print(f"  Methods compared: {len(self.methods_analyzed)}")
        if self.pair_agreements:
            top_agreement = max(self.pair_agreements, key=lambda x: x.jaccard)
            low_agreement = min(self.pair_agreements, key=lambda x: x.jaccard)
            print(f"  Best pair: {top_agreement.method_a} vs {top_agreement.method_b} ({top_agreement.jaccard:.3f})")
            print(f"  Worst pair: {low_agreement.method_a} vs {low_agreement.method_b} ({low_agreement.jaccard:.3f})")


def _get_top_k_cells(scores: np.ndarray, k: int) -> set[tuple[int, int]]:
    """Get indices of top-k cells by absolute value.

    Args:
        scores: Attribution scores [n_layers, n_positions]
        k: Number of top cells to return

    Returns:
        Set of (layer, position) tuples
    """
    flat_scores = np.abs(scores).flatten()
    top_indices = np.argsort(flat_scores)[-k:]

    n_cols = scores.shape[1]
    cells = set()
    for idx in top_indices:
        layer = idx // n_cols
        pos = idx % n_cols
        cells.add((layer, pos))

    return cells


def _compute_jaccard(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_method_agreement(
    summary: AttributionSummary,
    top_k: int = 20,
    mode: str = "",
) -> MethodAgreementResults:
    """Compute agreement between attribution methods.

    Args:
        summary: Attribution summary with results from multiple methods
        top_k: Number of top cells to compare
        mode: "denoising" or "noising" for labeling

    Returns:
        MethodAgreementResults with pairwise comparisons
    """
    if not summary or not summary.results:
        return MethodAgreementResults(mode=mode, top_k=top_k)

    # Extract top-k cells for each method
    method_cells: dict[str, set[tuple[int, int]]] = {}

    for key_str, result in summary.results.items():
        if result.scores.size == 0:
            continue
        cells = _get_top_k_cells(result.scores, top_k)
        method_cells[key_str] = cells

    methods = list(method_cells.keys())

    if len(methods) < 2:
        return MethodAgreementResults(
            methods_analyzed=methods,
            top_k=top_k,
            mode=mode,
            mean_jaccard=1.0 if len(methods) == 1 else 0.0,
        )

    # Compute pairwise agreements
    pair_agreements = []
    jaccard_values = []

    for i, method_a in enumerate(methods):
        for method_b in methods[i + 1:]:
            cells_a = method_cells[method_a]
            cells_b = method_cells[method_b]

            jaccard = _compute_jaccard(cells_a, cells_b)
            overlap = len(cells_a & cells_b)
            union = len(cells_a | cells_b)

            pair_agreements.append(MethodPairAgreement(
                method_a=method_a,
                method_b=method_b,
                jaccard=jaccard,
                overlap_count=overlap,
                total_union=union,
                top_k=top_k,
            ))
            jaccard_values.append(jaccard)

    mean_jaccard = float(np.mean(jaccard_values)) if jaccard_values else 0.0

    return MethodAgreementResults(
        pair_agreements=pair_agreements,
        mean_jaccard=mean_jaccard,
        methods_analyzed=methods,
        top_k=top_k,
        mode=mode,
    )


def analyze_attribution_agreement(
    agg: AttrPatchAggregatedResults,
    top_k: int = 20,
) -> dict[str, MethodAgreementResults]:
    """Analyze method agreement for aggregated attribution results.

    Args:
        agg: Aggregated attribution patching results
        top_k: Number of top cells to compare

    Returns:
        Dict with keys "denoising" and "noising" mapping to agreement results
    """
    results = {}

    if agg.denoising_agg:
        results["denoising"] = compute_method_agreement(
            agg.denoising_agg, top_k=top_k, mode="denoising"
        )

    if agg.noising_agg:
        results["noising"] = compute_method_agreement(
            agg.noising_agg, top_k=top_k, mode="noising"
        )

    return results
