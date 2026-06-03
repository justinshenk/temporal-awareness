"""Vectorized attribution computation."""

from __future__ import annotations

import numpy as np
import torch


def compute_attribution_vectorized(
    clean_act: torch.Tensor,
    corr_act: torch.Tensor,
    grad: torch.Tensor,
    clean_pos: np.ndarray,
    corr_pos: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Compute attribution scores using vectorized operations.

    Attribution = (clean - corrupted) * gradient

    Args:
        clean_act: Clean activations [batch, seq, hidden]
        corr_act: Corrupted activations [batch, seq, hidden]
        grad: Gradient of metric w.r.t. corrupted activations
        clean_pos: Clean position indices
        corr_pos: Corrupted position indices (mapped from clean)
        valid: Boolean mask for valid positions

    Returns:
        Attribution scores [clean_len]
    """
    clean_len = len(clean_pos)
    scores = np.zeros(clean_len)

    valid_clean = clean_pos[valid]
    valid_corr = corr_pos[valid]

    if len(valid_clean) == 0:
        return scores

    # Handle tensor shapes
    if clean_act.ndim == 3:
        clean_acts = clean_act[0, valid_clean, :]
    else:
        clean_acts = clean_act[valid_clean, :]

    if corr_act.ndim == 3:
        corr_acts = corr_act[0, valid_corr, :]
    else:
        corr_acts = corr_act[valid_corr, :]

    diff = clean_acts - corr_acts

    if grad.ndim == 3:
        grads = grad[0, valid_corr, :]
    else:
        grads = grad[valid_corr, :]

    attr = torch.sum(diff * grads, dim=1).detach().cpu().numpy()
    scores[valid] = attr

    return scores
