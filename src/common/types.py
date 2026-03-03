"""Common type aliases for the codebase."""

from typing import Literal, Union

# Patching mode types
PatchingMode = Literal["denoising", "noising"]
"""Mode for activation patching:
- 'denoising': Run on clean, patch corrupted activations
- 'noising': Run on corrupted, patch clean activations
"""

# Trajectory types
TrajectoryType = Literal["clean", "corrupted"]
"""Which trajectory in a contrastive pair."""

# Gradient computation target
GradTarget = Literal["clean", "corrupted"]
"""Where to compute gradients in attribution patching."""
