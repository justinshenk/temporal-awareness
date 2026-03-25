"""Type aliases for activation patching."""

from typing import Literal

# =============================================================================
# Component types
# =============================================================================

# All model components (for hooks/capturing)
COMPONENTS = ("resid_pre", "resid_post", "attn_out", "mlp_out")
Component = Literal["resid_pre", "resid_post", "attn_out", "mlp_out"]
"""All model components for activation capture."""

# Components used in patching/attribution (resid_pre excluded as redundant)
PATCHING_COMPONENTS = ("resid_post", "attn_out", "mlp_out")
PatchingComponent = Literal["resid_post", "attn_out", "mlp_out"]
"""Components used for patching and attribution."""

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
