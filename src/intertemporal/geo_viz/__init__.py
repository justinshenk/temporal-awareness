"""Geometric visualization of temporal representations.

Analyzes how time horizon information is encoded in model activations
using linear probes, PCA, and dimensionality reduction techniques.

Memory-optimized implementation with configurable parameters.
"""

from .geo_viz_config import (
    GeoVizConfig,
    RECOMMENDED_TARGETS,
    TargetSpec,
    # Memory optimization constants
    ACTIVATION_DTYPE,
    EXTRACTION_BUFFER_SIZE,
    ANALYSIS_GC_INTERVAL,
    MAX_STORED_PCA_COMPONENTS,
    USE_COMPRESSED_STORAGE,
    PLOT_GC_INTERVAL,
    MAX_TRAJECTORY_SAMPLES,
)
from .geo_viz_data import (
    ActivationData,
    collect_samples,
    extract_activations,
    get_time_horizon_months,
    load_cached_data,
    save_data,
)
from .geo_viz_analysis import (
    linear_probe_analysis,
    pca_correlation_analysis,
    compute_embeddings,
    run_streaming_analysis,
)
from .geo_viz_plotting import generate_all_plots
from .geo_viz_pipeline import run_geo_viz_pipeline

__all__ = [
    # Config
    "GeoVizConfig",
    "RECOMMENDED_TARGETS",
    "TargetSpec",
    # Memory constants
    "ACTIVATION_DTYPE",
    "EXTRACTION_BUFFER_SIZE",
    "ANALYSIS_GC_INTERVAL",
    "MAX_STORED_PCA_COMPONENTS",
    "USE_COMPRESSED_STORAGE",
    "PLOT_GC_INTERVAL",
    "MAX_TRAJECTORY_SAMPLES",
    # Data
    "ActivationData",
    "collect_samples",
    "extract_activations",
    "get_time_horizon_months",
    "load_cached_data",
    "save_data",
    # Analysis
    "linear_probe_analysis",
    "pca_correlation_analysis",
    "compute_embeddings",
    "run_streaming_analysis",
    # Plotting
    "generate_all_plots",
    # Pipeline
    "run_geo_viz_pipeline",
]
