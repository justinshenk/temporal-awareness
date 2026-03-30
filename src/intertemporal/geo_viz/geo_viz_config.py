"""Configuration for geometric visualization."""

from dataclasses import dataclass, field
from pathlib import Path


# Named position types
NAMED_POSITIONS = {
    # Source positions (in prompt)
    "time_horizon",
    "short_term_time",
    "short_term_reward",
    "long_term_time",
    "long_term_reward",
    # Dest positions (in response)
    "response",
    # Legacy aggregate positions
    "source",  # All source positions combined
    "dest",    # Same as response
}


def is_absolute_position(pos: str) -> bool:
    """Check if position is an absolute index (e.g., P86, P145)."""
    return pos.startswith("P") and pos[1:].isdigit()


def parse_absolute_position(pos: str) -> int:
    """Parse absolute position string to index (e.g., P86 -> 86)."""
    if not is_absolute_position(pos):
        raise ValueError(f"Not an absolute position: {pos}")
    return int(pos[1:])


@dataclass
class TargetSpec:
    """Specification for an activation extraction target.

    Attributes:
        layer: Transformer layer index
        component: Component type (resid_pre, resid_post, mlp_out, attn_out)
        position: Token position type (see NAMED_POSITIONS)
    """

    layer: int
    component: str
    position: str

    def __post_init__(self):
        valid_components = {"resid_pre", "resid_post", "mlp_out", "attn_out"}

        if self.component not in valid_components:
            raise ValueError(f"Invalid component: {self.component}")
        # Allow named positions OR absolute positions (P86, P145, etc.)
        if self.position not in NAMED_POSITIONS and not is_absolute_position(self.position):
            raise ValueError(f"Invalid position: {self.position}. Valid: {NAMED_POSITIONS} or absolute (P86, P145, etc.)")

    @property
    def key(self) -> str:
        """Unique key for this target."""
        return f"L{self.layer}_{self.component}_P{self.position}"

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


@dataclass
class GeoVizConfig:
    """Configuration for geometric visualization pipeline.

    Attributes:
        targets: List of activation extraction targets
        output_dir: Output directory for results
        model: Model identifier
        seed: Random seed for reproducibility
        max_samples: Maximum number of samples to use (None = all)
        n_pca_components: Number of PCA components
        umap_n_neighbors: UMAP n_neighbors parameter
        umap_min_dist: UMAP min_dist parameter
    """

    targets: list[TargetSpec] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: Path("out/geo_viz"))
    model: str = ""
    seed: int = 42
    max_samples: int | None = None
    n_pca_components: int = 50
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1

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
            umap_n_neighbors=d.get("umap_n_neighbors", 30),
            umap_min_dist=d.get("umap_min_dist", 0.1),
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
            "umap_n_neighbors": self.umap_n_neighbors,
            "umap_min_dist": self.umap_min_dist,
        }


# Recommended targets based on our analysis
RECOMMENDED_TARGETS = [
    # Best performers for time decoding (dest positions)
    TargetSpec(24, "resid_pre", "dest"),
    TargetSpec(21, "resid_post", "dest"),
    TargetSpec(21, "attn_out", "dest"),
    TargetSpec(19, "mlp_out", "dest"),
    TargetSpec(31, "mlp_out", "dest"),
    # Source positions for comparison (should show no time signal)
    TargetSpec(21, "attn_out", "source"),
    TargetSpec(21, "resid_post", "source"),
    TargetSpec(19, "mlp_out", "source"),
]
