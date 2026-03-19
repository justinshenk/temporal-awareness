"""Experiment configuration for intertemporal experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...common import BaseSchema
from ..preference import PreferenceDataset
from ..prompt import PromptDatasetConfig

# Default coarse patching settings (empty dict or empty lists = skip)
# component options: "resid_pre", "resid_post", "attn_out", "mlp_out"
COARSE_PATCH: dict = {
    "enabled": True,
    "layer_steps": [1],
    "pos_steps": [1],
    "components": ["resid_pre", "resid_post", "attn_out", "mlp_out"],
}

# Default attribution patching settings (empty dict = skip)
# methods: "standard", "eap", "eap_ig"
# components: "resid_post", "attn_out", "mlp_out"
# grad_at: ["clean"], ["corrupted"], or ["clean", "corrupted"]
# quadrature: ["midpoint"], ["gauss-legendre"], ["gauss-chebyshev"], or combinations
ATT_PATCH: dict = {
    "enabled": True,
    "methods": ["standard", "eap", "eap_ig"],
    "components": ["resid_post", "attn_out", "mlp_out"],
    "ig_steps": 20,
    "grad_at": ["clean", "corrupted"],
    "quadrature": ["midpoint", "gauss-legendre", "gauss-chebyshev"],
}

# Default visualization settings
VIZ: dict = {
    "enabled": True,
    "regenerate_all": False,
    "only_agg": False,  # If True, skip per-pair visualizations
}

# Default geometric analysis settings (PCA of residual stream)
GEO: dict = {
    "enabled": False,
    "layers": None,  # None = all layers, or list of specific layers
    "positions": None,  # None = last token only, or list of positions
    "n_components": 3,  # Number of PCA components to track
}

# Default difference-in-means settings
DIFFMEANS: dict = {
    "enabled": True,
    "n_components": 10,  # Number of SVD components to track
}


@dataclass
class ExperimentConfig(BaseSchema):
    """Experiment configuration."""

    # Core settings
    model: str
    dataset_config: dict
    max_samples: int | None = None
    n_pairs: int | None = None

    # Coarse patching settings
    coarse_patch: dict = field(default_factory=lambda: COARSE_PATCH.copy())

    # Attribution patching settings
    att_patch: dict = field(default_factory=lambda: ATT_PATCH.copy())

    # Visualization settings
    viz: dict = field(default_factory=lambda: VIZ.copy())

    # Difference-in-means settings
    diffmeans: dict = field(default_factory=lambda: DIFFMEANS.copy())

    # Geometric analysis settings (PCA)
    geo: dict = field(default_factory=lambda: GEO.copy())

    @property
    def name(self) -> str:
        return self.dataset_config.get("name", "default")

    def get_prefix(self) -> str:
        cfg = PromptDatasetConfig.from_dict(self.dataset_config)
        return PreferenceDataset.make_prefix(cfg.get_id(), self.model)
