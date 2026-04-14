"""Configuration for analysis slices in visualization.

Controls which analysis slices are generated.
"""

from __future__ import annotations

# When True, generates all analysis slices (all, same_labels, different_labels, etc.)
# When False, only generates core slices defined below
GENERATE_ALL_SLICES = False

# Core slices that are ALWAYS generated (even when GENERATE_ALL_SLICES=False)
# These are the most important analysis views
CORE_SLICES = ["all", "horizon", "no_horizon", "half_horizon"]
