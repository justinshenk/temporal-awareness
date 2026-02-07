"""Experiment orchestration for intertemporal preference analysis.

This package provides unified functions for running the full analysis pipeline:
- Data generation
- Activation patching
- Attribution patching
- Contrastive analysis (steering vectors)
- Steering evaluation

IMPORTANT DESIGN PRINCIPLES:
1. NEVER use TransformerLens/NNsight/Pyvene APIs directly - ALWAYS use ModelRunner
2. All experiment functions must work with any backend (configurable via ModelRunner)
3. No magic numbers - use config parameters or named constants
4. Check for existing code before adding new functions (avoid duplication)
"""

from ..common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
