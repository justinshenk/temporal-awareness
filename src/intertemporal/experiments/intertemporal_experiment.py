"""
Intertemporal preference experiment module.

Provides configs and the main experiment runner for analyzing
temporal preferences in language models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...common import ensure_dir, get_timestamp, profile, BaseSchema
from ...inference import InternalsConfig

from ..common import get_experiment_dir
from ..preference import (
    load_and_merge_preference_data,
    get_pref_dataset_dir,
    PreferenceDataset,
    generate_preference_data,
)
from .activation_patching import (
    run_activation_patching,
    IntertemporalActivationPatchingConfig,
    DualModeActivationPatchingResult,
)
from .attribution_patching import (
    run_attribution_patching,
    IntertemporalAttributionConfig,
)
from ..prompt import PromptDatasetConfig
from ...viz import visualize_activation_patching
from ...viz.patching_heatmaps import visualize_attribution_patching_result


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig(BaseSchema):
    """Configuration for intertemporal experiments."""

    # Important
    model: str
    dataset_config: dict

    # Optional
    internals_config: dict | None = None
    max_samples: int | None = None
    activation_patching_config: dict | None = None
    attribution_patching_config: dict | None = None

    # Attribution -> Activation patching flow
    use_attribution_targets: bool = False
    n_attribution_targets: int = 10

    @property
    def name(self) -> str:
        """Human-readable name derived from dataset_config."""
        return self.dataset_config.get("name", "default_name")

    def get_preference_dataset_prefix(self) -> str:
        """Get the prefix for preference dataset files: {dataset_id}_{model_name}."""
        dataset_cfg = PromptDatasetConfig.from_dict(self.dataset_config)
        return PreferenceDataset.make_prefix(dataset_cfg.get_id(), self.model)


@dataclass
class ExperimentContext:
    """Shared state passed between experiment steps."""

    cfg: ExperimentConfig
    pref_data: PreferenceDataset | None = None

    act_patching: Any | None = None

    att_patching: Any | None = None

    # Output
    output_dir: Path = field(default_factory=get_experiment_dir)
    timestamp: str | None = None

    @property
    def ts(self) -> Path:
        if not self.timestamp:
            self.timestamp = get_timestamp()
        return self.timestamp

    @property
    def viz_dir(self) -> Path:
        viz_dir = self.output_dir / "viz"
        ensure_dir(viz_dir)
        return viz_dir

    @property
    def data_dir(self) -> Path:
        data_dir = self.output_dir / "data"
        ensure_dir(data_dir)
        return data_dir

    @property
    def run_dir(self) -> Path:
        run_dir = self.output_dir / self.ts
        ensure_dir(run_dir)
        return run_dir


@profile("step_preference_data")
def step_preference_data(ctx: ExperimentContext) -> None:
    """Load preference data if it exists, otherwise generate it."""
    pref_dataset_prefix = ctx.cfg.get_preference_dataset_prefix()

    pref_data = load_and_merge_preference_data(
        pref_dataset_prefix, get_pref_dataset_dir()
    )

    if not pref_data:
        pref_data = generate_preference_data(
            model=ctx.cfg.model,
            dataset_config=ctx.cfg.dataset_config,
            internals_config=(
                InternalsConfig.from_dict(ctx.cfg.internals_config)
                if ctx.cfg.internals_config
                else None
            ),
            max_samples=ctx.cfg.max_samples,
        )
        print("Generated preference data")
    else:
        print("Loaded existing preference data")

    pref_data.print_summary()
    ctx.pref_data = pref_data


@profile("step_attribution_patching")
def step_attribution_patching(ctx: ExperimentContext) -> None:
    """Run attribution patching to identify important layers/positions."""
    cfg = None
    if ctx.cfg.attribution_patching_config:
        cfg = IntertemporalAttributionConfig.from_dict(
            ctx.cfg.attribution_patching_config
        )
    ctx.att_patching = run_attribution_patching(ctx.pref_data, cfg)


@profile("step_activation_patching")
def step_activation_patching(ctx: ExperimentContext) -> None:
    """Run activation patching, optionally guided by attribution results."""
    cfg = (
        IntertemporalActivationPatchingConfig.from_dict(
            ctx.cfg.activation_patching_config
        )
        if ctx.cfg.activation_patching_config
        else IntertemporalActivationPatchingConfig()
    )

    # Use attribution-identified layers if available
    if ctx.cfg.use_attribution_targets and ctx.att_patching:
        target = ctx.att_patching.get_layer_target(
            n_layers=ctx.cfg.n_attribution_targets
        )
        if target:
            cfg.target = target

    ctx.act_patching = run_activation_patching(ctx.pref_data, cfg)


@profile("step_visualization")
def step_visualization(ctx: ExperimentContext) -> None:
    """Generate visualizations for experiment results."""
    if ctx.act_patching:
        if isinstance(ctx.act_patching, DualModeActivationPatchingResult):
            # Visualize both modes
            if ctx.act_patching.denoising:
                viz_path = ctx.viz_dir / f"activation_patching_denoising_{ctx.ts}.png"
                visualize_activation_patching(
                    ctx.act_patching.denoising,
                    title="Activation Patching (Denoising)",
                    save_path=viz_path,
                )
            if ctx.act_patching.noising:
                viz_path = ctx.viz_dir / f"activation_patching_noising_{ctx.ts}.png"
                visualize_activation_patching(
                    ctx.act_patching.noising,
                    title="Activation Patching (Noising)",
                    save_path=viz_path,
                )
        else:
            # Single mode
            viz_path = ctx.viz_dir / f"activation_patching_{ctx.ts}.png"
            visualize_activation_patching(
                ctx.act_patching,
                title="Activation Patching Recovery by Layer",
                save_path=viz_path,
            )
    if ctx.att_patching:
        # Save visualization
        viz_path = ctx.viz_dir / f"attribution_patching_{ctx.ts}.png"
        visualize_attribution_patching_result(
            ctx.att_patching,
            save_path=viz_path,
        )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


@profile("run_experiment")
def run_experiment(cfg: ExperimentConfig) -> ExperimentContext:
    """Run the full intertemporal preference experiment."""

    ctx = ExperimentContext(cfg)

    # Step 1: Get Preference Dataset (Prompt Dataset + Model's responses)
    step_preference_data(ctx)

    # Step 2: Attribution Patching (identifies important layers/positions)
    step_attribution_patching(ctx)

    # Step 3: Activation Patching (tests causal effects at those positions)
    step_activation_patching(ctx)

    # Step 4: Visualization
    step_visualization(ctx)

    return ctx
