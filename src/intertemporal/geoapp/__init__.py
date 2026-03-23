"""Interactive geometric visualization app for temporal awareness analysis.

Provides a modern Dash-based web UI with multiple views for exploring
PCA, UMAP, and t-SNE embeddings across layers, positions, and components.

Features:
- Component Explorer: Drill down into specific layer/component/position combinations
- Layer Explorer: Compare all 4 components side-by-side for a layer
- Trajectory View: Visualize how activations evolve across layers
- Position Slider: Explore different token positions dynamically
- No-horizon toggle: Highlight samples without time horizon specification
"""

from .app import create_app, run_app
from .data_loader import GeoVizDataLoader

__all__ = [
    "GeoVizDataLoader",
    "create_app",
    "run_app",
]
