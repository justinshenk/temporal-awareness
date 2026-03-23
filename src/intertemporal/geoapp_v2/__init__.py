"""FastAPI backend for GeoViz visualization app v2.

Provides JSON API endpoints for embedding data, metadata, and sample information.
Reuses the existing GeoVizDataLoader for data access.
"""

from .models import (
    ColorValues,
    ConfigResponse,
    EmbeddingResponse,
    ErrorResponse,
    MetricsResponse,
    PCAMetrics,
    Point3D,
    ProbeMetrics,
    SampleResponse,
)
from .routes import create_router
from .server import create_app, run_app

__all__ = [
    # Models
    "ColorValues",
    "ConfigResponse",
    "EmbeddingResponse",
    "ErrorResponse",
    "MetricsResponse",
    "PCAMetrics",
    "Point3D",
    "ProbeMetrics",
    "SampleResponse",
    # Server
    "create_app",
    "create_router",
    "run_app",
]
