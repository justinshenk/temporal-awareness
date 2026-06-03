"""Node metrics for branching point analysis.

Provides metrics for analyzing branching nodes in token trees.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...base_schema import BaseSchema
from ..analysis_base import DistributionalAnalysis


@dataclass
class NodeMetrics(DistributionalAnalysis):
    """Metrics at a branching node's vocab distribution."""

    next_token_logprobs: list[float]  # logprobs of candidate tokens at this node
    vocab_entropy: (
        float  # H of full vocab dist at divergent pos — lower = more decisive
    )
    vocab_diversity: float  # D₁ = e^H — effective vocab size at decision point
    vocab_simpson: float  # D₂ = 1/Σpᵢ² — Simpson diversity at decision point

    # Token Constraint Bound (requires hidden state + W_U to compute)
    # Default 0.0 means "not computed" (requires model access during analysis)
    vocab_tcb: float = 0.0


@dataclass
class NodeAnalysis(BaseSchema):
    """Analysis at a branching node."""

    node_idx: int
    metrics: NodeMetrics
