"""Geometric visualization of temporal representations.

Analyzes how time horizon information is encoded in model activations
using linear probes, PCA, and dimensionality reduction techniques.

Memory-optimized implementation with configurable parameters.
"""

from .geometry_config import (
    GeometryConfig,
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
from .geometry_data import (
    ActivationData,
    ChoiceInfo,
    collect_samples,
    extract_activations,
    get_time_horizon_months,
    load_cached_data,
)
from .geometry_analysis import (
    linear_probe_analysis,
    pca_correlation_analysis,
    compute_embeddings,
    run_streaming_analysis,
    # New optional analyses
    CrossPositionSimilarityResult,
    ContinuousTimeProbeResult,
    compute_cross_position_similarity,
    compute_continuous_time_probe,
    # No-horizon projection analysis
    NoHorizonProjectionResult,
    run_no_horizon_analysis,
)
from .geometry_plotting import (
    generate_all_plots,
    plot_cross_position_similarity,
    plot_continuous_time_probe,
    plot_logit_lens,
    plot_no_horizon_projection,
)
from .geometry_logit_lens import (
    LogitLensResult,
    compute_logit_lens,
    run_logit_lens_analysis,
    run_logit_lens_from_cache,
)
from .geometry_pipeline import run_geometry_pipeline

__all__ = [
    # Config
    "GeometryConfig",
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
    "ChoiceInfo",
    "collect_samples",
    "extract_activations",
    "get_time_horizon_months",
    "load_cached_data",
    # Analysis
    "linear_probe_analysis",
    "pca_correlation_analysis",
    "compute_embeddings",
    "run_streaming_analysis",
    # New optional analyses
    "CrossPositionSimilarityResult",
    "ContinuousTimeProbeResult",
    "compute_cross_position_similarity",
    "compute_continuous_time_probe",
    # No-horizon projection analysis
    "NoHorizonProjectionResult",
    "run_no_horizon_analysis",
    # Plotting
    "generate_all_plots",
    "plot_cross_position_similarity",
    "plot_continuous_time_probe",
    "plot_logit_lens",
    "plot_no_horizon_projection",
    # Logit Lens
    "LogitLensResult",
    "compute_logit_lens",
    "run_logit_lens_analysis",
    "run_logit_lens_from_cache",
    # Pipeline
    "run_geometry_pipeline",
]
