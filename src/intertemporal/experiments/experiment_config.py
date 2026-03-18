"""Experiment configuration for intertemporal experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...common import BaseSchema
from ..preference import PreferenceDataset
from ..prompt import PromptDatasetConfig

# Default coarse patching settings (empty dict or empty lists = skip)
# component options: "resid_pre", "resid_post", "attn_out", "mlp_out"
COARSE_PATCH: dict = {
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
    "enabled": False,
    "methods": ["standard", "eap", "eap_ig"],
    "components": ["resid_post", "attn_out", "mlp_out"],
    "ig_steps": [20],
    "grad_at": ["clean", "corrupted"],
    "quadrature": ["midpoint", "gauss-legendre", "gauss-chebyshev"],
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

    @property
    def name(self) -> str:
        return self.dataset_config.get("name", "default")

    def get_prefix(self) -> str:
        cfg = PromptDatasetConfig.from_dict(self.dataset_config)
        return PreferenceDataset.make_prefix(cfg.get_id(), self.model)
