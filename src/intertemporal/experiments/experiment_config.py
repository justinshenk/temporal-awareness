"""Experiment configuration for intertemporal experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ...common import BaseSchema
from ..preference import PreferenceDataset
from ..prompt import PromptDatasetConfig

# =============================================================================
# Default General Configs
# =============================================================================

# Visualization
VIZ_CFG: dict = {
    "enabled": True,
}

# Pair requirements (empty = no filtering)
PAIR_REQ_CFG: dict = {}


# =============================================================================
# Default Step Configs
# =============================================================================

# Coarse patching: component options are "resid_pre", "resid_post", "attn_out", "mlp_out"
COARSE_CFG: dict = {
    "enabled": True,
    "no_cache": False,
    ######################
    ### CONFIG VALUES  ###
    ######################
    "layer_steps": [1],
    "pos_steps": [15],
    "components": ["resid_post", "attn_out", "mlp_out", "resid_pre"],
}

# Attribution patching
# - methods: "standard", "eap", "eap_ig"
# - components: "resid_post", "attn_out", "mlp_out"
# - quadrature: ["midpoint"], ["gauss-legendre"], ["gauss-chebyshev"], or combinations
ATTRIB_CFG: dict = {
    "enabled": True,
    "no_cache": False,
    ######################
    ### CONFIG VALUES  ###
    ######################
    "ig_steps": 20,
    "methods": ["standard", "eap_ig", "eap"],
    "components": ["mlp_out", "attn_out", "resid_post", "resid_pre"],
    "quadrature": ["midpoint", "gauss-chebyshev", "gauss-legendre"],
}

# Difference-of-means analysis
DIFFMEANS_CFG: dict = {
    "enabled": True,
    "no_cache": False,
    ######################
    ### CONFIG VALUES  ###
    ######################
    "n_components": 10,
}

# MLP neuron analysis
MLP_CFG: dict = {
    "enabled": True,
    "no_cache": False,
    ######################
    ### CONFIG VALUES  ###
    ######################
    "layers": [19, 21, 24, 28, 31, 34, 35],
    "n_top_neurons": 50,
}

# Attention pattern analysis
ATTN_CFG: dict = {
    "enabled": True,
    "no_cache": False,
    ######################
    ### CONFIG VALUES  ###
    ######################
    "layers": [18, 19, 21, 24, 28, 31, 34, 35],
    "store_patterns": True,
    "dynamic_threshold": 0.05,
}

# Fine-grained patching
FINE_CFG: dict = {
    "enabled": True,
    "no_cache": False,
    ######################
    ### CONFIG VALUES  ###
    ######################
    # Head attribution
    "head_patching_enabled": True,
    "head_layers": [19, 21, 24, 28, 29, 30, 31, 34],
    # Position patching for top heads
    "position_patching_enabled": True,
    "n_top_heads_for_position": 10,
    # Path patching
    "path_patching_enabled": True,
    "source_layers": [19, 21, 24],
    "dest_mlp_layers": [28, 29, 30, 31, 34],
    "dest_head_layers": [28, 29, 30, 31, 34],
    "n_top_source_heads": 5,
    # Multi-site interaction
    "multi_site_enabled": True,
    "n_components_multi_site": 5,
    # Neuron attribution
    "neuron_patching_enabled": True,
    "neuron_target_layer": 31,
    "mlp_layers": [19, 24, 28, 31, 34],
    "n_top_neurons": 50,
    # Layer-position fine heatmap
    "layer_position_enabled": True,
    "layer_position_components": ["attn_out", "mlp_out"],
    "layer_position_layers": None,  # None = layers 15-35
}


@dataclass
class ExperimentConfig(BaseSchema):
    """Experiment configuration."""

    # Core settings
    model: str
    dataset_config: dict

    # Filtering configs
    max_samples: int | None = None
    n_pairs: int | None = None
    pair_req_cfg: dict = field(default_factory=lambda: PAIR_REQ_CFG.copy())

    # Viz config
    viz_cfg: dict = field(default_factory=lambda: VIZ_CFG.copy())

    # Step configs
    coarse_cfg: dict = field(default_factory=lambda: COARSE_CFG.copy())
    attrib_cfg: dict = field(default_factory=lambda: ATTRIB_CFG.copy())
    diffmeans_cfg: dict = field(default_factory=lambda: DIFFMEANS_CFG.copy())
    mlp_cfg: dict = field(default_factory=lambda: MLP_CFG.copy())
    attn_cfg: dict = field(default_factory=lambda: ATTN_CFG.copy())
    fine_cfg: dict = field(default_factory=lambda: FINE_CFG.copy())

    @property
    def dataset_name(self) -> str:
        return self.dataset_config.get("name", "default")

    @property
    def step_cfgs(self) -> list[dict]:
        """All step configuration dicts."""
        return [
            self.coarse_cfg,
            self.attrib_cfg,
            self.diffmeans_cfg,
            self.mlp_cfg,
            self.attn_cfg,
            self.fine_cfg,
        ]

    def get_preference_data_prefix(self) -> str:
        cfg = PromptDatasetConfig.from_dict(self.dataset_config)
        return PreferenceDataset.make_prefix(cfg.get_id(), self.model)

    def save(self, output_dir: Path) -> Path:
        """Save config to experiment_config.json in output_dir.

        Also saves to original_experiment_config.json if it doesn't exist yet.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save current config
        path = output_dir / "experiment_config.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save original config only on first run
        original_path = output_dir / "original_experiment_config.json"
        if not original_path.exists():
            with open(original_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)

        return path

    @classmethod
    def load(cls, output_dir: Path) -> "ExperimentConfig | None":
        """Load config from original_experiment_config.json (preferred) or experiment_config.json."""
        output_dir = Path(output_dir)

        # Prefer original config (immutable first-run config)
        original_path = output_dir / "original_experiment_config.json"
        if original_path.exists():
            with open(original_path) as f:
                data = json.load(f)
            return cls.from_dict(data)

        # Fall back to current config
        path = output_dir / "experiment_config.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
