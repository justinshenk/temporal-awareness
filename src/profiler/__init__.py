"""Simple profiling utilities.

Usage:
    from src.profiler import P

    with P("section_name"):
        work()

    P.report()
"""

from .timer import P, Profiler, TimingEntry

__all__ = ["P", "Profiler", "TimingEntry"]
