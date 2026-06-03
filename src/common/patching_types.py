"""Type aliases for activation patching."""

from typing import Literal

# =============================================================================
# Component types
# =============================================================================

# All model components (for hooks/capturing)
# Order follows data flow: resid_pre -> attn -> resid_mid -> mlp -> resid_post
COMPONENTS = ("resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "attn_z")
Component = Literal["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "attn_z"]
"""All model components for activation capture.

resid_mid: Residual stream after attention but BEFORE MLP. Captures attention's contribution.
attn_z: Per-head attention output BEFORE O projection. Shape [batch, seq, n_heads, d_head].
        Use for head-level interventions.
"""

# Components used in patching/attribution
PATCHING_COMPONENTS = ("resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "attn_z")
PatchingComponent = Literal["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "attn_z"]
"""Components used for patching and attribution.

attn_z: Per-head attention output BEFORE O projection. Use with head parameter for head-level patching.
"""

# =============================================================================
# Mode types
# =============================================================================

PatchingMode = Literal["denoising", "noising", "zero_ablation", "mean_ablation", "gaussian_noise"]
"""Mode for activation patching:
- 'denoising': Run on corrupted, patch in clean activations (REMOVE noise)
- 'noising': Run on clean, patch in corrupted activations (ADD noise)
- 'zero_ablation': Set activations to zero (test circuit necessity)
- 'mean_ablation': Set activations to pre-computed mean (test circuit necessity)
- 'gaussian_noise': Add Gaussian noise to activations (robustness testing)
"""

TrajectoryType = Literal["clean", "corrupted"]
"""Which trajectory in a contrastive pair."""

GradTarget = Literal["clean", "corrupted"]
"""Where to compute gradients in attribution patching."""
