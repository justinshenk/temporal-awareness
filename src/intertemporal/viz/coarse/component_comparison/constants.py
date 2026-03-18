"""Constants for component comparison visualizations."""

from __future__ import annotations

# Component order for consistent visualization
COMPONENTS = ["resid_pre", "attn_out", "mlp_out", "resid_post"]

COMPONENT_COLORS = {
    "resid_pre": "#1f77b4",   # blue
    "attn_out": "#ff7f0e",    # orange
    "mlp_out": "#2ca02c",     # green
    "resid_post": "#d62728",  # red
}

# Subdirectory names for organized output
SUBDIR_SANITY = "01_sanity_checks"
SUBDIR_OVERVIEW = "02_overview"
SUBDIR_DECOMP = "03_component_decomp"
SUBDIR_REDUNDANCY = "04_redundancy"
SUBDIR_SYNTHESIS = "05_circuit_synthesis"

SUBDIRS = [SUBDIR_SANITY, SUBDIR_OVERVIEW, SUBDIR_DECOMP, SUBDIR_REDUNDANCY, SUBDIR_SYNTHESIS]

# Colormap conventions for consistency across plots
# These are documented in the README for reader reference
CMAP_MAGNITUDE = "RdYlGn"      # Recovery/disruption magnitude (green=high, red=low)
CMAP_INTERACTION = "hot"       # 2D localization map (white=high, black=low)
CMAP_DIVERGING = "RdBu_r"      # Redundancy gap (red=necessity, blue=sufficiency)
CMAP_SEQUENTIAL = "viridis"    # Layer/position indexing (yellow=late, purple=early)
