"""Configuration for geometric visualization."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# =============================================================================
# Memory Optimization Constants
# =============================================================================

# Data type for activations (float32 = 50% memory savings vs float64)
ACTIVATION_DTYPE = np.float32

# Buffer size for streaming extraction (samples to accumulate before flushing to disk)
# Higher = fewer disk writes but more memory; Lower = more disk writes but less memory
EXTRACTION_BUFFER_SIZE = 100

# Buffer size for streaming analysis (targets to process before GC)
ANALYSIS_GC_INTERVAL = 50

# Number of PCA components to store (reduces disk usage)
# Set to None to store all computed components
MAX_STORED_PCA_COMPONENTS = 20

# Whether to use compressed numpy storage (.npz vs .npy)
# Compressed is smaller on disk but slower to load
USE_COMPRESSED_STORAGE = False

# Plotting batch sizes
PLOT_GC_INTERVAL = 20  # GC after every N target groups
MAX_TRAJECTORY_SAMPLES = 200  # Max samples to plot in trajectory (for clarity)


# =============================================================================
# Semantic Position Types (from DefaultPromptFormat)
# =============================================================================

# All valid semantic positions for extraction
# These map to regions in the prompt template, not absolute token indices
SEMANTIC_POSITIONS = {
    # Prompt markers
    "situation_marker",
    "task_marker",
    "consider_marker",
    "action_marker",
    "format_marker",
    "format_choice_prefix",
    "format_reasoning_prefix",

    # Option labels (a), b))
    "left_label",
    "right_label",

    # Option values (by presentation order)
    "left_time",
    "left_reward",
    "right_time",
    "right_reward",
    "time_horizon",
    "post_time_horizon",

    # Response positions
    "response_choice_prefix",
    "response_choice",
    "response_reasoning_prefix",
    "response_reasoning",
}



# =============================================================================
# Target Specification
# =============================================================================

@dataclass(slots=True)
class TargetSpec:
    """Specification for an activation extraction target.

    Uses __slots__ for reduced memory overhead.

    Attributes:
        layer: Transformer layer index
        component: Component type (resid_pre, resid_post, mlp_out, attn_out)
        position: Token position type (see SEMANTIC_POSITIONS)
    """

    layer: int
    component: str
    position: str

    def __post_init__(self):
        valid_components = {"resid_pre", "resid_post", "mlp_out", "attn_out"}

        if self.component not in valid_components:
            raise ValueError(f"Invalid component: {self.component}")

        if self.position not in SEMANTIC_POSITIONS:
            raise ValueError(
                f"Invalid position: {self.position}. "
                f"Valid semantic positions: {sorted(SEMANTIC_POSITIONS)}"
            )

    @property
    def key(self) -> str:
        """Unique key for this target.

        Format: L{layer}_{component}_{position}
        Example: L21_resid_post_response_choice, L13_resid_pre_short_term_reward
        """
        return f"L{self.layer}_{self.component}_{self.position}"

    @property
    def hook_name(self) -> str:
        """TransformerLens hook name."""
        patterns = {
            "resid_pre": f"blocks.{self.layer}.hook_resid_pre",
            "resid_post": f"blocks.{self.layer}.hook_resid_post",
            "mlp_out": f"blocks.{self.layer}.hook_mlp_out",
            "attn_out": f"blocks.{self.layer}.hook_attn_out",
        }
        return patterns[self.component]

    def __repr__(self) -> str:
        return self.key


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class GeoVizConfig:
    """Configuration for geometric visualization pipeline.

    Attributes:
        targets: List of activation extraction targets
        output_dir: Output directory for results
        model: Model identifier
        seed: Random seed for reproducibility
        max_samples: Maximum number of samples to use (None = all)
        n_pca_components: Number of PCA components to compute

        # Memory optimization settings
        extraction_buffer_size: Samples to buffer before flushing to disk
        use_compressed_storage: Use compressed .npz files (slower but smaller)
    """

    targets: list[TargetSpec] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: Path("out/geo_viz"))
    model: str = ""
    seed: int = 42
    max_samples: int | None = None
    n_pca_components: int = 50

    # Memory optimization settings
    extraction_buffer_size: int = EXTRACTION_BUFFER_SIZE
    use_compressed_storage: bool = USE_COMPRESSED_STORAGE

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    @classmethod
    def from_dict(cls, d: dict) -> "GeoVizConfig":
        """Create config from dictionary."""
        targets = [
            TargetSpec(**t) if isinstance(t, dict) else t for t in d.get("targets", [])
        ]
        return cls(
            targets=targets,
            output_dir=Path(d.get("output_dir", "out/geo_viz")),
            model=d.get("model", ""),
            seed=d.get("seed", 42),
            max_samples=d.get("max_samples"),
            n_pca_components=d.get("n_pca_components", 50),
            extraction_buffer_size=d.get("extraction_buffer_size", EXTRACTION_BUFFER_SIZE),
            use_compressed_storage=d.get("use_compressed_storage", USE_COMPRESSED_STORAGE),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "targets": [
                {"layer": t.layer, "component": t.component, "position": t.position}
                for t in self.targets
            ],
            "output_dir": str(self.output_dir),
            "model": self.model,
            "seed": self.seed,
            "max_samples": self.max_samples,
            "n_pca_components": self.n_pca_components,
            "extraction_buffer_size": self.extraction_buffer_size,
            "use_compressed_storage": self.use_compressed_storage,
        }


# =============================================================================
# Recommended Targets
# =============================================================================

RECOMMENDED_TARGETS = [
    # Response positions (where choice is decoded)
    TargetSpec(24, "resid_pre", "response_choice"),
    TargetSpec(21, "resid_post", "response_choice"),
    TargetSpec(21, "attn_out", "response_choice"),

    # Time horizon positions
    TargetSpec(21, "resid_post", "time_horizon"),
    TargetSpec(19, "mlp_out", "time_horizon"),

    # Option value positions (by presentation order)
    TargetSpec(21, "resid_post", "left_reward"),
    TargetSpec(21, "resid_post", "right_reward"),
    TargetSpec(21, "resid_post", "left_time"),
    TargetSpec(21, "resid_post", "right_time"),

    # Labels
    TargetSpec(13, "resid_post", "left_label"),
    TargetSpec(13, "resid_post", "right_label"),
]
