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

from .data_loader import GeoVizDataLoader
from .server import create_app, run_app

__all__ = [
    "GeoVizDataLoader",
    "create_app",
    "run_app",
]
