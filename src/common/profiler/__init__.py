"""Simple profiling utilities.

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.

Usage:
    from src.common.profiler import P
    from src.common import P  # also works (via re-export)

    with P("section_name"):
        work()

    P.report()

    # Or use the @profile decorator
    from src.common.profiler import profile

    @profile  # uses function name
    def load_data():
        return load()

    @profile("custom_name")  # custom identifier
    def other_func():
        pass
"""

from src.common.auto_export import auto_export

# Explicit import needed: 'profile' is excluded by auto_export (stdlib collision)
from .decorators import profile

__all__ = auto_export(__file__, __name__, globals())
__all__.append("profile")  # Add back excluded name
