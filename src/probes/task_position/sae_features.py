"""Apply the Gemma-Scope L20 SAE to saved residual activations."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from sae_lens import SAE

GEMMA_SCOPE_RELEASE = "gemma-scope-9b-it-res"
GEMMA_SCOPE_L20_SAE_ID = "layer_20/width_131k/average_l0_81"
F90871_INDEX = 90871


def load_gemma_scope_l20_sae(device: str = "cuda") -> SAE:
    """Load the Gemma-Scope L20 IT SAE used by the existing context-fatigue work."""
    sae = SAE.from_pretrained(
        release=GEMMA_SCOPE_RELEASE, sae_id=GEMMA_SCOPE_L20_SAE_ID
    )
    if isinstance(sae, tuple):
        sae = sae[0]  # newer sae_lens versions return (sae, cfg, _)
    sae = sae.to(device).eval()
    return sae


@torch.no_grad()
def encode_features(
    sae: SAE,
    activations: torch.Tensor,
    feature_indices: Iterable[int],
    batch_size: int = 4096,
    device: str = "cuda",
) -> np.ndarray:
    """Encode activations through the SAE and return only the selected feature columns."""
    feature_indices = list(feature_indices)
    n = activations.shape[0]
    out = np.zeros((n, len(feature_indices)), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = activations[start:end].to(device).to(sae.dtype)
        feats = sae.encode(batch)
        out[start:end] = feats[:, feature_indices].float().cpu().numpy()
    return out
