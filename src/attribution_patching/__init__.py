"""Attribution patching module for gradient-based importance scoring.

Attribution patching computes: (clean - corrupted) * gradient
to approximate causal effects without running actual interventions.

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.
"""

from ..common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
