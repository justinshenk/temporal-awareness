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
from ...common.device_utils import clear_gpu_memory
from ...inference import InternalsConfig

from ..common import get_experiment_dir
from ..preference import (
    load_and_merge_preference_data,
    get_pref_dataset_dir,
    PreferenceDataset,
    generate_preference_data,
)
from ...activation_patching import ActivationPatchingTarget
from .activation_patching import (
    run_activation_patching,
    IntertemporalActivationPatchingConfig,
    DualModeActivationPatchingResult,
)
from .sweeps import run_layer_sweep, run_progressive_position_patching, SweepResults
from .attribution_patching import (
    run_attribution_patching,
    IntertemporalAttributionConfig,
)
from ..prompt import PromptDatasetConfig
from ...viz import visualize_activation_patching
from ...viz.patching_heatmaps import visualize_attribution_patching_result
from ...viz.patching_viz import (
    plot_layer_position_heatmap,
    plot_layer_metrics_line,
    plot_sweep_summary,
)


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

    # Sweep results (populated before main patching)
    sweep_results: Any | None = None  # SweepResults
    layer_sweep_result: Any | None = None  #  AggregatedActivationPatchingResult
    position_sweep_result: Any | None = None  # AggregatedActivationPatchingResult

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


@profile("step_sweep_phases")
def step_sweep_phases(ctx: ExperimentContext) -> None:
    """Run layer sweep and progressive multi-position patching."""
    cfg = (
        IntertemporalActivationPatchingConfig.from_dict(
            ctx.cfg.activation_patching_config
        )
        if ctx.cfg.activation_patching_config
        else IntertemporalActivationPatchingConfig()
    )

    n_pairs = cfg.n_pairs
    component = (
        cfg.target.component if hasattr(cfg.target, "component") else "resid_post"
    )
    mode = cfg.mode if cfg.mode != "both" else "denoising"

    # Phase 1: Layer sweep
    layer_recovery, layer_result = run_layer_sweep(
        ctx.pref_data, n_pairs=n_pairs, component=component, mode=mode
    )
    ctx.layer_sweep_result = layer_result

    # Clear memory after layer sweep
    clear_gpu_memory()

    # Get best layers (top 3)
    sorted_layers = sorted(layer_recovery.items(), key=lambda x: x[1], reverse=True)
    best_layers = [l for l, _ in sorted_layers[:3] if l is not None]

    if not best_layers:
        print("WARNING: No valid layers found in sweep")
        ctx.sweep_results = SweepResults(layer_recovery=layer_recovery)
        return

    # Phase 2: Progressive multi-position patching at best layer
    # This finds minimum positions needed for flip, without expensive per-position sweep
    best_layer = best_layers[0]

    # Use estimated sequence length (typical prompt + response is ~200 tokens)
    seq_len = 220

    # Create position list (all positions, to be used progressively)
    all_positions = list(range(seq_len))

    # Run progressive patching to find how many positions are needed
    position_recovery = run_progressive_position_patching(
        ctx.pref_data,
        layer=best_layer,
        sorted_positions=all_positions,  # Try positions in order
        n_pairs=n_pairs,
        component=component,
        mode=mode,
        max_positions=100,  # Try up to 100 positions
        step=10,  # Increment by 10 each time
    )

    # Clear memory after position sweep
    clear_gpu_memory()

    # Find the minimum positions that achieved good recovery
    best_n_pos = max(position_recovery.keys()) if position_recovery else 0
    for n_pos, recovery in sorted(position_recovery.items()):
        if recovery > 0.8:
            best_n_pos = n_pos
            break

    ctx.sweep_results = SweepResults(
        layer_recovery=layer_recovery,
        position_recovery=position_recovery,  # n_positions -> recovery
        best_layers=best_layers,
        best_positions=list(range(best_n_pos)),  # Positions 0 to best_n_pos
    )

    print(
        f"\nSweep complete: best_layers={best_layers}, min_positions_for_flip={best_n_pos}"
    )


@profile("step_activation_patching")
def step_activation_patching(ctx: ExperimentContext) -> None:
    """Run activation patching, using sweep results for targeting."""
    cfg = (
        IntertemporalActivationPatchingConfig.from_dict(
            ctx.cfg.activation_patching_config
        )
        if ctx.cfg.activation_patching_config
        else IntertemporalActivationPatchingConfig()
    )

    # Use sweep results for targeted patching if available
    if ctx.sweep_results and ctx.sweep_results.best_layers:
        # Use the minimum positions identified by progressive patching
        # If we found a small number of positions that achieve good recovery, use those
        # Otherwise fall back to patching all positions
        best_positions = ctx.sweep_results.best_positions
        if best_positions and len(best_positions) < 100:
            # Use explicit positions from progressive patching
            cfg.target = ActivationPatchingTarget(
                position_mode="explicit",
                token_positions=best_positions,
                layers=[ctx.sweep_results.best_layers[0]],  # Use best layer
                component=cfg.target.component
                if hasattr(cfg.target, "component")
                else "resid_post",
            )
            print(
                f"\nTargeted patching: L{ctx.sweep_results.best_layers[0]} with {len(best_positions)} positions"
            )
        else:
            # Patch all positions at the best layer
            # Note: Greedy verification may show "degenerate" because the model
            # outputs option content instead of the label, but the logprob-based
            # choice measurement is correct
            best_layer = ctx.sweep_results.best_layers[0]
            cfg.target = ActivationPatchingTarget(
                position_mode="all",
                layers=[best_layer],
                component=cfg.target.component
                if hasattr(cfg.target, "component")
                else "resid_post",
            )
            print(f"\nTargeted patching at layer L{best_layer}, position_mode=all")

    # Use attribution-identified layers if available (fallback)
    elif ctx.cfg.use_attribution_targets and ctx.att_patching:
        target = ctx.att_patching.get_layer_target(
            n_layers=ctx.cfg.n_attribution_targets
        )
        if target:
            cfg.target = target

    # Use alpha=1.0 (full replacement) - alpha<1.0 is too weak to change behavior
    # The layer sweep shows 97% recovery with alpha=1.0, so degeneration is rare
    cfg.alpha = 1.0
    print(f"Using alpha={cfg.alpha} (full replacement mode)")

    ctx.act_patching = run_activation_patching(ctx.pref_data, cfg)


@profile("step_visualization")
def step_visualization(ctx: ExperimentContext) -> None:
    """Generate visualizations for experiment results."""
    # Sweep visualizations (new)
    if ctx.sweep_results:
        # Sweep summary plot
        viz_path = ctx.viz_dir / f"sweep_summary_{ctx.ts}.png"
        plot_sweep_summary(
            ctx.sweep_results,
            title="Activation Patching Sweep Summary",
            save_path=viz_path,
        )

        # Layer-position heatmap
        if ctx.sweep_results.layer_recovery and ctx.sweep_results.position_recovery:
            viz_path = ctx.viz_dir / f"layer_position_heatmap_{ctx.ts}.png"
            plot_layer_position_heatmap(
                layer_recovery=ctx.sweep_results.layer_recovery,
                position_recovery=ctx.sweep_results.position_recovery,
                title="Layer and Position Recovery",
                save_path=viz_path,
            )

        # Layer metrics line plot
        if ctx.sweep_results.layer_recovery:
            viz_path = ctx.viz_dir / f"layer_metrics_{ctx.ts}.png"
            plot_layer_metrics_line(
                ctx.sweep_results.layer_recovery,
                title="Recovery by Layer",
                save_path=viz_path,
            )

    # Original activation patching visualization
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

    # Attribution patching visualization
    if ctx.att_patching:
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

    # Step 3: Layer and Position Sweeps (auto-discovery)
    step_sweep_phases(ctx)

    # Step 4: Targeted Activation Patching (tests causal effects at best positions)
    step_activation_patching(ctx)

    # Step 5: Visualization
    step_visualization(ctx)

    return ctx
