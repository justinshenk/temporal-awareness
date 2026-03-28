"""Pydantic models for Geometry API responses."""

from typing import Any

from pydantic import BaseModel, Field


class PromptTemplateElement(BaseModel):
    """A clickable element in the prompt template UI."""

    name: str = Field(description="Position identifier (e.g., 'situation_marker', 'response_choice')")
    label: str = Field(description="Human-readable label for display")
    type: str = Field(description="Element type: marker, variable, static, semantic")
    available: bool = Field(description="Whether this position has data available")


class ConfigResponse(BaseModel):
    """Response for /api/config endpoint."""

    layers: list[int] = Field(description="Available transformer layers")
    components: list[str] = Field(description="Available activation components")
    positions: list[str] = Field(description="Available token positions")
    color_options: list[str] = Field(description="Available color-by options")
    n_samples: int = Field(description="Total number of samples in dataset")
    model_name: str = Field(default="", description="Name of the model being visualized")
    position_labels: dict[str, str] = Field(default_factory=dict, description="Human-readable labels for positions")
    prompt_template: list[PromptTemplateElement] = Field(default_factory=list, description="Prompt template structure for UI")
    semantic_to_positions: dict[str, list[str]] = Field(default_factory=dict, description="Mapping from semantic positions to token positions (for compatibility)")
    markers: dict[str, str] = Field(default_factory=dict, description="Section markers from prompt format (e.g., situation_marker: 'SITUATION:')")
    rel_pos_counts: dict[str, int] = Field(default_factory=dict, description="Number of tokens (rel_pos values) for each semantic position")
    available_methods: list[str] = Field(default_factory=lambda: ["pca"], description="Available dimensionality reduction methods (pca, umap, tsne)")


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
    # Flat array of coordinates: [x0, y0, z0, x1, y1, z1, ...] for performance
    coordinates_flat: list[float] = Field(description="Flat array of 3D coordinates [x0,y0,z0,x1,y1,z1,...]")
    sample_indices: list[int] = Field(description="Original sample indices for each coordinate")


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
    response_label: str | None = Field(default=None, description="Model's chosen option label (e.g., 'a)' or 'b)')")
    response_term: str | None = Field(default=None, description="Which term was chosen: 'short_term' or 'long_term'")
    response_text: str | None = Field(default=None, description="Full response text from the model")
    choice_confidence: float | None = Field(default=None, description="Model's confidence in the choice (0-1)")
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


class WarmupStatus(BaseModel):
    """Status of a warmup/precomputation task."""

    is_running: bool = Field(description="Whether warmup is currently running")
    progress: int = Field(default=0, description="Number of embeddings computed")
    total: int = Field(default=0, description="Total embeddings to compute")
    current_task: str | None = Field(default=None, description="Current task being computed")
    cached_pca: int = Field(default=0, description="Number of PCA embeddings in cache")
    cached_umap: int = Field(default=0, description="Number of UMAP embeddings in cache")
    cached_tsne: int = Field(default=0, description="Number of t-SNE embeddings in cache")


class WarmupResponse(BaseModel):
    """Response for warmup endpoints."""

    message: str = Field(description="Status message")
    status: WarmupStatus = Field(description="Current warmup status")


class TrajectoryPoint(BaseModel):
    """PC1 value at a specific x-axis position."""

    x_value: str = Field(description="X-axis value (layer number or position name)")
    values: list[float] = Field(description="PC1 values for all samples at this x position")
    sample_indices: list[int] = Field(description="Sample indices corresponding to each value")


class TrajectoryResponse(BaseModel):
    """Response for trajectory endpoints (1D x Layer or 1D x Position)."""

    component: str = Field(description="Activation component used")
    position: str | None = Field(default=None, description="Position (for layer trajectory)")
    layer: int | None = Field(default=None, description="Layer (for position trajectory)")
    method: str = Field(description="Embedding method (pca)")
    x_axis: str = Field(description="What the x-axis represents: 'layer' or 'position'")
    x_values: list[str] = Field(description="X-axis values in order")
    n_samples: int = Field(description="Number of samples")
    sample_indices: list[int] = Field(default_factory=list, description="Sample indices (for layer trajectory where all layers share same indices)")
    data: list[TrajectoryPoint] = Field(description="PC1 values at each x position")


class TokenInfo(BaseModel):
    """Information about a single token in a sample."""

    abs_pos: int = Field(description="Absolute position index in sequence")
    decoded_token: str = Field(description="Decoded token string")
    traj_section: str = Field(description="Section: 'prompt' or 'response'")
    format_pos: str | None = Field(default=None, description="Semantic position name")
    rel_pos: int = Field(default=-1, description="Relative position within format_pos")


class TokensResponse(BaseModel):
    """Response for /api/tokens/{sample_idx} endpoint."""

    sample_idx: int = Field(description="Sample index")
    tokens: list[TokenInfo] = Field(description="Token information for each position")
