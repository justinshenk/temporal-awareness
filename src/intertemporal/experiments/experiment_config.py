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
    "pos_steps": [5],
    "components": ["resid_post", "attn_out", "mlp_out", "resid_pre"],
    # "components": ["resid_post"],
}

# Default attribution patching settings (empty dict = skip)
# methods: "standard", "eap", "eap_ig"
# components: "resid_post", "attn_out", "mlp_out"
# quadrature: ["midpoint"], ["gauss-legendre"], ["gauss-chebyshev"], or combinations
# Note: grad_at is determined by mode (noising=clean, denoising=corrupted)
ATT_PATCH: dict = {
    "enabled": True,
    "ig_steps": 30,
    "methods": ["standard", "eap_ig", "eap"],
    "components": ["mlp_out", "attn_out", "resid_post"],
    "quadrature": ["midpoint", "gauss-chebyshev", "gauss-legendre"],
    # "methods": ["standard"],
    # "components": ["mlp_out"],
    # "quadrature": ["midpoint"],
}


# Default visualization settings
VIZ: dict = {
    "enabled": True,
    "regenerate_all": False,
    "only_agg": False,  # If True, skip per-pair visualizations
}

# Default geometric analysis settings (PCA of residual stream)
GEO: dict = {
    "enabled": True,
    "layers": [0, 6, 13, 17, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 33, 34],
    "positions": [
        28,
        30,
        32,
        35,
        44,
        48,
        52,
        86,
        87,
        88,
        103,
        121,
        122,
        139,
        140,
        143,
        144,
        145,
    ],
    "n_components": 5,
}

# Default difference-in-means settings
DIFFMEANS: dict = {
    "enabled": True,
    "n_components": 10,  # Number of SVD components to track
}

# Default pair requirement settings (empty = no requirements, allows all valid pairs)
# Set "different_labels": True for multilabel experiments
PAIR_REQ: dict = {}


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

    # Pair requirements (filtering criteria for contrastive pairs)
    pair_req: dict = field(default_factory=lambda: PAIR_REQ.copy())

    @property
    def name(self) -> str:
        return self.dataset_config.get("name", "default")

    def get_prefix(self) -> str:
        cfg = PromptDatasetConfig.from_dict(self.dataset_config)
        return PreferenceDataset.make_prefix(cfg.get_id(), self.model)
