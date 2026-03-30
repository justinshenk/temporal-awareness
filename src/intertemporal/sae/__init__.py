"""Temporal SAE training pipeline.

Trains Sparse Autoencoders on position-specific activations from LLM responses
to temporal preference questions. Key modules:

- sae_analysis: SAE model, training, and feature extraction
- sae_inference: Position-based activation extraction
- sae_positions: Token position resolution and component hooks
- sae_pipeline: Pipeline orchestration with crash recovery
- pipeline_state: State management and configuration

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.
"""

from src.common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
