"""Common utilities for temporal-awareness experiments.

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.

This package provides core data structures and utilities:

Binary Choice Types:
    from src.common import SimpleBinaryChoice, GroupedBinaryChoice

Token Tree Types:
    from src.common import TokenTrajectory, TokenTree

Analysis:
    from src.common import TrajectoryAnalysis, analyze_token_tree

Math Utilities:
    from src.common import perplexity, aggregate, AggregationMethod

Profiling:
    from src.common import P, profile

Base Utilities:
    from src.common import BaseSchema, load_json, save_json

Subpackages can also be accessed directly:
    from src.common.choice import SimpleBinaryChoice
    from src.common.math import perplexity
    from src.common.analysis import TrajectoryAnalysis
"""

from src.common.auto_export import auto_export

# Re-export from subpackages for flat access (e.g., from src.common import SimpleBinaryChoice)
from .choice import *
from .analysis import *
from .math import *
from .profiler import *
from .time_value import TimeValue, TIME_UNITS, TIME_UNIT_TO_YEARS, DEFAULT_TIME_UNIT

__all__ = auto_export(__file__, __name__, globals())
