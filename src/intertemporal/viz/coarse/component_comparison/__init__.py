"""Component comparison visualizations package.

Visualizations that compare patching effects across all four components:
resid_pre, attn_out, mlp_out, resid_post.
"""

from __future__ import annotations

from .....common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
