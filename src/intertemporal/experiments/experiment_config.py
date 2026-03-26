"""Experiment configuration for intertemporal experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...common import BaseSchema
from ..preference import PreferenceDataset
from ..prompt import PromptDatasetConfig

# Default coarse patching settings (empty dict or empty lists = skip)
# component options: "resid_pre", "resid_post", "attn_out", "mlp_out"
COARSE_PATCH: dict = {
    "enabled": False,
    "no_cache": False,
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
    "enabled": False,
    "no_cache": False,
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
    "enabled": False,
    "no_cache": True,
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
    "enabled": False,
    "no_cache": True,
    "n_components": 10,  # Number of SVD components to track
}

# Default fine-grained activation patching settings
# Uses FAST attribution for heads/neurons (not sweeps), then causal patching for position/path/multi-site
FINE_PATCH: dict = {
    "enabled": False,
    "no_cache": True,
    # Head attribution (fast - uses specified layers, not sweep)
    "head_patching_enabled": True,
    "head_layers": [24, 21, 19, 29, 30],  # Key attention layers (fast attribution)
    # Position patching for top heads (causal)
    "position_patching_enabled": True,
    "n_top_heads_for_position": 5,
    "position_range": None,  # (start, end) or auto
    # Path patching (causal)
    "path_patching_enabled": True,
    "source_layers": [19, 21, 24],
    "dest_mlp_layers": [28, 31, 34],
    "dest_head_layers": [28, 29, 30, 31],
    "n_top_source_heads": 5,
    # Multi-site interaction (causal)
    "multi_site_enabled": True,
    "n_components_multi_site": 10,
    # Neuron attribution (fast - uses specified layer, not sweep)
    "neuron_patching_enabled": True,
    "neuron_target_layer": 31,
    "mlp_layers": [31, 24, 28],  # Key MLP layers (fast attribution)
    "n_top_neurons": 50,
    # Layer-position fine heatmap (causal)
    "layer_position_enabled": True,
    "layer_position_components": ["attn_out", "mlp_out"],
    "layer_position_layers": None,  # None = layers 15-35
}

# Default MLP neuron analysis settings
MLP_ANALYSIS: dict = {
    "enabled": False,
    "no_cache": True,
    "layers": [35, 31, 28, 19],  # Key MLP layers for horizon processing
    "n_top_neurons": 50,  # Number of top neurons to track per layer
}

# Default attention pattern analysis settings
ATTN_ANALYSIS: dict = {
    "enabled": True,
    "no_cache": True,
    "layers": [19, 21, 24],  # Key attention layers for horizon processing
    "store_patterns": True,  # Whether to store full attention patterns
    "dynamic_threshold": 0.1,  # Threshold for detecting dynamic attention changes
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

    # Fine-grained activation patching settings
    fine_patch: dict = field(default_factory=lambda: FINE_PATCH.copy())

    # Geometric analysis settings (PCA)
    geo: dict = field(default_factory=lambda: GEO.copy())

    # MLP neuron analysis settings
    mlp_analysis: dict = field(default_factory=lambda: MLP_ANALYSIS.copy())

    # Attention pattern analysis settings
    attn_analysis: dict = field(default_factory=lambda: ATTN_ANALYSIS.copy())

    # Pair requirements (filtering criteria for contrastive pairs)
    pair_req: dict = field(default_factory=lambda: PAIR_REQ.copy())

    @property
    def name(self) -> str:
        return self.dataset_config.get("name", "default")

    def get_prefix(self) -> str:
        cfg = PromptDatasetConfig.from_dict(self.dataset_config)
        return PreferenceDataset.make_prefix(cfg.get_id(), self.model)
