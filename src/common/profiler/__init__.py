"""Simple profiling utilities.

Usage:
    from src.common.profiler import P

    with P("section_name"):
        work()

    P.report()

    # Or use the profile_fn decorator
    from src.common.profiler import profile_fn

    @profile_fn("LOAD DATA")
    def load_data():
        return load()
"""

from ..auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
