"""Geometric visualization of temporal representations.

Analyzes how time horizon information is encoded in model activations
using linear probes, PCA, and dimensionality reduction techniques.
"""

from .geo_viz_config import GeoVizConfig, RECOMMENDED_TARGETS, TargetSpec
from .geo_viz_data import collect_samples, extract_activations, get_time_horizon_months, load_cached_data, save_data
from .geo_viz_analysis import (
    linear_probe_analysis,
    pca_correlation_analysis,
    compute_embeddings,
)
from .geo_viz_plotting import generate_all_plots
from .geo_viz_pipeline import run_geo_viz_pipeline

__all__ = [
    "GeoVizConfig",
    "RECOMMENDED_TARGETS",
    "TargetSpec",
    "collect_samples",
    "extract_activations",
    "get_time_horizon_months",
    "load_cached_data",
    "save_data",
    "linear_probe_analysis",
    "pca_correlation_analysis",
    "compute_embeddings",
    "generate_all_plots",
    "run_geo_viz_pipeline",
]
