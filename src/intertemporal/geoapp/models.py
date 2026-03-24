"""Pydantic models for GeoViz API responses."""

from typing import Any

from pydantic import BaseModel, Field


class ConfigResponse(BaseModel):
    """Response for /api/config endpoint."""

    layers: list[int] = Field(description="Available transformer layers")
    components: list[str] = Field(description="Available activation components")
    positions: list[str] = Field(description="Available token positions")
    color_options: list[str] = Field(description="Available color-by options")
    n_samples: int = Field(description="Total number of samples in dataset")


class Point3D(BaseModel):
    """A single 3D point with coordinates."""

    x: float
    y: float
    z: float


class EmbeddingResponse(BaseModel):
    """Response for /api/embedding/{layer}/{component}/{position} endpoint."""

    layer: int = Field(description="Transformer layer")
    component: str = Field(description="Activation component (resid_pre, attn_out, etc.)")
    position: str = Field(description="Token position")
    method: str = Field(description="Dimensionality reduction method (pca, umap, tsne)")
    n_samples: int = Field(description="Number of samples")
    coordinates: list[Point3D] = Field(description="3D coordinates for each sample")


class ColorValues(BaseModel):
    """Response for /api/metadata endpoint."""

    color_by: str = Field(description="Metadata field used for coloring")
    values: list[Any] = Field(description="Color values for each sample (can be numeric or categorical)")
    dtype: str = Field(description="Data type: 'numeric', 'categorical', or 'boolean'")


class SampleResponse(BaseModel):
    """Response for /api/sample/{idx} endpoint."""

    idx: int = Field(description="Sample index")
    text: str = Field(description="Full prompt text")
    time_horizon_months: float | None = Field(default=None, description="Time horizon in months")
    time_scale: str | None = Field(default=None, description="Time scale category")
    choice_type: str | None = Field(default=None, description="Type of intertemporal choice")
    short_term_first: bool | None = Field(default=None, description="Whether short-term option appears first")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional sample metadata")


class ProbeMetrics(BaseModel):
    """Linear probe metrics for a target."""

    train_accuracy: float | None = Field(default=None, description="Training accuracy")
    test_accuracy: float | None = Field(default=None, description="Test accuracy")
    train_r2: float | None = Field(default=None, description="Training R-squared")
    test_r2: float | None = Field(default=None, description="Test R-squared")


class PCAMetrics(BaseModel):
    """PCA metrics for a target."""

    explained_variance_ratio: list[float] | None = Field(default=None, description="Variance explained by each component")
    cumulative_variance: list[float] | None = Field(default=None, description="Cumulative variance explained")
    n_components: int | None = Field(default=None, description="Number of components computed")


class MetricsResponse(BaseModel):
    """Response for /api/metrics/{layer}/{component}/{position} endpoint."""

    layer: int = Field(description="Transformer layer")
    component: str = Field(description="Activation component")
    position: str = Field(description="Token position")
    linear_probe: ProbeMetrics | None = Field(default=None, description="Linear probe metrics if available")
    pca: PCAMetrics | None = Field(default=None, description="PCA metrics if available")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(description="Error message")
    detail: str | None = Field(default=None, description="Additional error details")


class HeatmapCell(BaseModel):
    """A single cell in a heatmap."""

    layer: int = Field(description="Layer number")
    position: str = Field(description="Position identifier")
    value: float | None = Field(description="Metric value (null if not available)")


class HeatmapResponse(BaseModel):
    """Response for /api/heatmap endpoint."""

    metric: str = Field(description="Metric being displayed")
    component: str = Field(description="Component used")
    layers: list[int] = Field(description="Layers in order")
    positions: list[str] = Field(description="Positions in order")
    cells: list[HeatmapCell] = Field(description="Heatmap cell data")
    min_value: float | None = Field(default=None, description="Minimum value for color scale")
    max_value: float | None = Field(default=None, description="Maximum value for color scale")
