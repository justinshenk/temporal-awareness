"""Type aliases for activation patching."""

from typing import Literal

# Patching mode types
PatchingMode = Literal["denoising", "noising"]
"""Mode for activation patching:
- 'denoising': Run on corrupted, patch in clean activations (REMOVE noise)
- 'noising': Run on clean, patch in corrupted activations (ADD noise)
"""

# Trajectory types
TrajectoryType = Literal["clean", "corrupted"]
"""Which trajectory in a contrastive pair."""

# Gradient computation target
GradTarget = Literal["clean", "corrupted"]
"""Where to compute gradients in attribution patching."""
