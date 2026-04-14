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

# Coarse patching: component options are "resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"
COARSE_CFG: dict = {
    "enabled": True,
    "no_cache": False,
    "no_viz": False,
    ######################
    ### CONFIG VALUES  ###
    ######################
    "layer_steps": [1],
    "pos_steps": [1],
    "components": ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"],
    # "components": ["resid_post"],
}

# Attribution patching
# - methods: "standard", "eap", "eap_ig"
# - components: "resid_post", "attn_out", "mlp_out"
# - quadrature: ["midpoint"], ["gauss-legendre"], ["gauss-chebyshev"], or combinations
ATTRIB_CFG: dict = {
    "enabled": True,
    "no_cache": False,
    "no_viz": False,
    ######################
    ### CONFIG VALUES  ###
    ######################
    "ig_steps": 20,
    "methods": ["standard", "eap_ig", "eap"],
    "components": ["resid_pre", "resid_mid", "mlp_out", "attn_out", "resid_post"],
    "quadrature": ["midpoint", "gauss-chebyshev", "gauss-legendre"],
    # "methods": ["standard"],
    # "components": ["resid_post"],
    # "quadrature": ["midpoint"],
}

# Difference-of-means analysis
DIFFMEANS_CFG: dict = {
    "enabled": True,
    "no_cache": False,
    "no_viz": False,
    ######################
    ### CONFIG VALUES  ###
    ######################
    "n_components": 10,
}

# MLP neuron analysis (includes per-neuron logit contribution)
MLP_CFG: dict = {
    "enabled": True,
    "no_cache": False,
    "no_viz": False,
    ######################
    ### CONFIG VALUES  ###
    ######################
    "layers": [18, 19, 21, 23, 24, 28, 31, 34, 35],
    "n_top_neurons": 50,
    # Neuron attribution is computed as part of MLP analysis
    # Each neuron's logit_contribution = activation_diff * W_out @ logit_direction
}

# Attention pattern analysis
ATTN_CFG: dict = {
    "enabled": True,
    "no_cache": False,
    "no_viz": False,
    ######################
    ### CONFIG VALUES  ###
    ######################
    "layers": [18, 19, 21, 23, 24, 28, 31, 34, 35],
    "store_patterns": True,
    "dynamic_threshold": 0.05,
    # Head attribution
    "head_attribution_enabled": True,
    # Position patching for top heads
    "position_patching_enabled": True,
    "n_top_heads_for_position": 10,
}

# Fine-grained patching (path patching, multi-site)
FINE_CFG: dict = {
    "enabled": True,
    "no_cache": False,
    "no_viz": False,
    ######################
    ### CONFIG VALUES  ###
    ######################
    # Path patching
    "path_patching_enabled": True,
    "dest_mlp_layers": [18, 19, 21, 23, 24, 28, 31, 34, 35],
    "dest_head_layers": [18, 19, 21, 23, 24, 28, 31, 34, 35],
    "n_top_source_heads": 10,
    # Multi-site interaction
    "multi_site_enabled": True,
    "n_components_multi_site": 10,
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

    def save(self, output_dir: Path, update_working: bool = False) -> Path:
        """Save config to output_dir.

        Files:
        - original_config.json: Immutable first-run config (never updated)
        - working_config.json: Mutable config loaded on subsequent runs
        - experiment_config.json: Current run config (always updated, for compatibility)

        Args:
            output_dir: Directory to save config files
            update_working: If True, update working_config.json with current config
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        # Save current config (for compatibility)
        path = output_dir / "experiment_config.json"
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save original config only on first run
        original_path = output_dir / "original_config.json"
        working_path = output_dir / "working_config.json"

        if not original_path.exists():
            # First run: create both original and working configs
            with open(original_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            with open(working_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif update_working:
            # Update working config when --update-config is used
            with open(working_path, "w") as f:
                json.dump(config_dict, f, indent=2)

        return path

    @classmethod
    def load_working(cls, output_dir: Path) -> "ExperimentConfig | None":
        """Load config from working_config.json."""
        output_dir = Path(output_dir)
        working_path = output_dir / "working_config.json"

        if working_path.exists():
            with open(working_path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        return None

    @classmethod
    def load(cls, output_dir: Path) -> "ExperimentConfig | None":
        """Load config from original_config.json (preferred) or legacy files."""
        output_dir = Path(output_dir)

        # Prefer new original_config.json
        original_path = output_dir / "original_config.json"
        if original_path.exists():
            with open(original_path) as f:
                data = json.load(f)
            return cls.from_dict(data)

        # Fall back to legacy original_experiment_config.json
        legacy_original = output_dir / "original_experiment_config.json"
        if legacy_original.exists():
            with open(legacy_original) as f:
                data = json.load(f)
            return cls.from_dict(data)

        # Fall back to experiment_config.json
        path = output_dir / "experiment_config.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
