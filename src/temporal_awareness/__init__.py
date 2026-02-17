"""Temporal Awareness: Detecting and steering temporal preference in LLMs.

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.
"""

__version__ = "0.1.0"

from ..common.auto_export import auto_export

# Explicit imports from external package (required)
from latents import SteeringFramework
from latents.model_adapter import get_model_config

__all__ = auto_export(__file__, __name__, globals())
