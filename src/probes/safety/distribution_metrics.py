"""Output-distribution metrics (entropy) for context-drift stress testing."""

from __future__ import annotations

import numpy as np


def softmax_entropy(logits: np.ndarray) -> float:
    """Shannon entropy (nats) of softmax(logits) for a 1-D logit vector."""
    x = np.asarray(logits, dtype=np.float64).ravel()
    x = x - x.max()
    p = np.exp(x)
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())
