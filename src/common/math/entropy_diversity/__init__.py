"""Core entropy and diversity theory (unified framework).

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.
"""

from src.common.auto_export import auto_export

# auto_export imports all modules in this package and re-exports their public names
# Do NOT add `from .xxx import *` statements - let auto_export handle it
__all__ = auto_export(__file__, __name__, globals())

# Explicitly re-export _EPS (underscore-prefixed but needed by other modules)
from .core import _EPS

__all__ = __all__ + ["_EPS"]
