"""Binary choice types for LLM response analysis.

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.

Provides abstract base classes and concrete implementations for
representing binary choices made by language models.

Usage:
    from src.common.choice import SimpleBinaryChoice
    from src.common import SimpleBinaryChoice  # also works (via re-export)
"""

from src.common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
