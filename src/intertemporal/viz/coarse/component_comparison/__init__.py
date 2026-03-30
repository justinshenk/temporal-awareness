"""Component comparison visualizations package.

Visualizations that compare patching effects across all four components:
resid_pre, attn_out, mlp_out, resid_post.
"""

from __future__ import annotations

from .comp_constants import COMPONENTS, COMPONENT_COLORS, SUBDIRS
from .comp_main import plot_all_component_comparisons

__all__ = [
    "COMPONENTS",
    "COMPONENT_COLORS",
    "SUBDIRS",
    "plot_all_component_comparisons",
]
