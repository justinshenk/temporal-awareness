"""Common utilities for temporal-awareness experiments.

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.
"""

from src.common.auto_export import auto_export

# Re-export from subpackages for flat access (e.g., from src.common import SimpleBinaryChoice)
# NOTE: Import order matters! analysis must come before choice (choice imports from analysis)
from .math import *
from .analysis import *
from .choice import *
from .profiler import *
from .time_value import TimeValue, TIME_UNITS, TIME_UNIT_TO_YEARS, DEFAULT_TIME_UNIT

__all__ = auto_export(__file__, __name__, globals())
