"""Visualization configuration macros.

Controls which visualizations are generated to avoid combinatorial explosion.
"""

from __future__ import annotations

# =============================================================================
# Analysis Slice Configuration
# =============================================================================

# When True, generates all analysis slices (all, same_labels, different_labels, etc.)
# When False, only generates core slices defined below
GENERATE_ALL_SLICES = False

# Core slices that are ALWAYS generated (even when GENERATE_ALL_SLICES=False)
# These are the most important analysis views
CORE_SLICES = ["all", "horizon", "no_horizon"]

# =============================================================================
# Attribution Visualization Axes
# =============================================================================

# Which axes to generate subfolders for in attribution patching viz.
# Each enabled axis generates plots varying that axis while keeping others at default.
# Options: "method", "component", "grad_at", "quadrature"
ATT_VIZ_AXES = ["method", "component"]

# When True, generates comparison.png showing all methods side-by-side
ATT_VIZ_COMPARISON = True

# Default values for attribution patching visualization axes
# These are used when varying one axis while keeping others at default
ATT_DEFAULT_METHOD = "eap_ig"
ATT_DEFAULT_COMPONENT = "resid_post"
ATT_DEFAULT_GRAD_AT = "clean"
ATT_DEFAULT_QUADRATURE = None  # None means use first available
