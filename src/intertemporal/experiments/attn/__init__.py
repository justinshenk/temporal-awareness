"""Attention pattern analysis for intertemporal experiments.

Analyzes attention patterns using semantic position names:
- Source positions: where information is read FROM (e.g., "time_horizon")
- Dest positions: where information flows TO (e.g., "response_choice")
"""

from src.common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
