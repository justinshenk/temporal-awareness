"""Coarse activation patching visualization module.

Provides visualization for layer and position sweep patching results.
"""

from .colors import (
    OKABE_ITO,
    METRIC_COLORS,
    LINE_STYLES,
    LINE_WIDTHS,
    MARKERS,
    MARKER_SIZES,
    SUBPLOT_TITLE_STYLE,
)

from .helpers import (
    get_tick_spacing,
    get_tick_color,
    color_xaxis_ticks,
    save_with_colored_ticks,
    save_with_colored_ticks_multi,
    add_token_type_legend,
    finalize_plot,
)

__all__ = [
    # Colors
    "OKABE_ITO",
    "METRIC_COLORS",
    "LINE_STYLES",
    "LINE_WIDTHS",
    "MARKERS",
    "MARKER_SIZES",
    "SUBPLOT_TITLE_STYLE",
    # Helpers
    "get_tick_spacing",
    "get_tick_color",
    "color_xaxis_ticks",
    "save_with_colored_ticks",
    "save_with_colored_ticks_multi",
    "add_token_type_legend",
    "finalize_plot",
]
