"""Interactive geometric visualization app for temporal awareness analysis.

A FastAPI + React web application for exploring PCA, UMAP, and t-SNE
embeddings across layers, positions, and components.

Features:
- 3D scatter plot visualization with WebGL
- Layer/component/position selection
- Multiple dimensionality reduction methods (PCA, UMAP, t-SNE)
- Color by various metadata fields
- Sample detail inspection on click
"""

from ...common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
