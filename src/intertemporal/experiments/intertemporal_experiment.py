"""Intertemporal preference experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...common import ensure_dir, get_timestamp, profile, get_device, BaseSchema
from ...inference import (
    InternalsConfig,
    get_recommended_backend_internals,
    InterventionTarget,
    COMPONENTS,
)
from ...binary_choice import BinaryChoiceRunner
from ...activation_patching import (
    patch_pair,
    ActPatchAggregatedResult,
    ActPatchPairResult,
)

from ..common import get_experiment_dir, get_pref_dataset_dir
from ..common.contrastive_preferences import get_contrastive_preferences
from ..preference import (
    PreferenceDataset,
    generate_preference_data,
    load_and_merge_preference_data,
)
from ..prompt import PromptDatasetConfig
from ..viz import (
    visualize_att_patching,
    visualize_coarse_patching,
    visualize_fine_patching,
    visualize_tokenization,
)
from ...viz.token_coloring import get_token_coloring_for_pair
from ...activation_patching.coarse_activation_patching import (
    CoarseActPatchResults,
    CoarseActPatchAggregatedResults,
    run_coarse_act_patching,
)
from ...attribution_patching import (
    attribute_pair,
    AttrPatchPairResult,
    AttrPatchAggregatedResults,
)


@dataclass
class ExperimentConfig(BaseSchema):
    """Experiment configuration."""

    model: str
    dataset_config: dict
    internals_config: dict | None = None
    max_samples: int | None = None
    n_pairs: int = 5

    @property
    def name(self) -> str:
        return self.dataset_config.get("name", "default")

    def get_prefix(self) -> str:
        cfg = PromptDatasetConfig.from_dict(self.dataset_config)
        return PreferenceDataset.make_prefix(cfg.get_id(), self.model)


@dataclass
class ExperimentContext:
    """Shared experiment state."""

    cfg: ExperimentConfig
    pref_data: PreferenceDataset | None = None

    output_dir: Path = field(default_factory=get_experiment_dir)
    timestamp: str = field(default_factory=get_timestamp)

    _pairs: list | None = field(default=None, init=False)
    _runner: BinaryChoiceRunner | None = field(default=None, init=False)

    coarse_patching: dict[int, CoarseActPatchResults] = field(default_factory=dict)
    fine_patching: dict[int, ActPatchPairResult] = field(default_factory=dict)
    att_patching: dict[int, AttrPatchPairResult] = field(default_factory=dict)

    coarse_agg: CoarseActPatchAggregatedResults | None = None
    fine_agg: ActPatchAggregatedResult | None = None
    att_agg: AttrPatchAggregatedResults | None = None

    @property
    def runner(self) -> BinaryChoiceRunner:
        """Cached runner for this experiment."""
        if self._runner is None:
            backend = get_recommended_backend_internals()
            self._runner = BinaryChoiceRunner(
                self.pref_data.model, device=get_device(), backend=backend
            )
        return self._runner

    @property
    def pairs(self):
        """Contrastive pairs (cached)."""
        if self._pairs is None:
            print("[ctx] Getting contrastive preferences...")
            all_pairs = get_contrastive_preferences(self.pref_data)
            print(
                f"[ctx] Found {len(all_pairs)} contrastive preferences, selecting {self.cfg.n_pairs}"
            )
            selected = all_pairs[: self.cfg.n_pairs]
            anchor_texts = self.pref_data.prompt_format_config.get_anchor_texts()
            first_interesting_marker = self.pref_data.prompt_format_config.get_prompt_marker_before_time_horizon()
            print("[ctx] Building contrastive pairs...")
            self._pairs = []
            for i, p in enumerate(selected):
                if p:
                    print(f"[ctx]   Building pair {i + 1}/{len(selected)}...")
                    pair = p.get_contrastive_pair(
                        self.runner,
                        anchor_texts=anchor_texts,
                        first_interesting_marker=first_interesting_marker,
                    )
                    if pair:
                        self._pairs.append(pair)
            print(f"[ctx] Built {len(self._pairs)} valid pairs")
        return self._pairs

    @property
    def viz_dir(self) -> Path:
        d = self.output_dir / "viz"
        ensure_dir(d)
        return d

    def get_union_target(self, component: str = "resid_post") -> InterventionTarget:
        """Get union target from attribution or coarse patching aggregates."""
        # Use attribution patching results if available
        if self.att_agg:
            target = self.att_agg.get_target()
            if target:
                return target

        # Fall back to coarse patching aggregated results
        if self.coarse_agg:
            return self.coarse_agg.get_union_target(component=component)

        return InterventionTarget.all(component=component)

    def get_runner(self) -> BinaryChoiceRunner:
        """Get cached runner for this model."""
        return self.runner

    @property
    def best_contrastive_pair(self):
        """First contrastive pair (for coarse patching)."""
        return self.pairs[0] if self.pairs else None


@profile("step_preference_data")
def step_preference_data(
    ctx: ExperimentContext, try_loading_data: bool = False
) -> None:
    """Load or generate preference data."""

    if try_loading_data:
        ctx.pref_data = load_and_merge_preference_data(
            ctx.cfg.get_prefix(), get_pref_dataset_dir()
        )
    if not ctx.pref_data:
        ctx.pref_data = generate_preference_data(
            model=ctx.cfg.model,
            dataset_config=ctx.cfg.dataset_config,
            internals_config=InternalsConfig.from_dict(ctx.cfg.internals_config)
            if ctx.cfg.internals_config
            else None,
            max_samples=ctx.cfg.max_samples,
        )

    print(ctx.pref_data)
    ctx.pref_data.print_summary()


@profile("step_attribution_patching")
def step_attribution_patching(ctx: ExperimentContext) -> None:
    """Run attribution patching on each contrastive pair."""
    ctx.att_agg = AttrPatchAggregatedResults()

    for pair_idx, pair in enumerate(ctx.pairs):
        print(f"\n[attr] Processing pair {pair_idx + 1}/{len(ctx.pairs)}")
        result = attribute_pair(ctx.runner, pair)
        ctx.att_patching[pair_idx] = result
        ctx.att_agg.add(result)

    ctx.att_agg.print_summary()


@profile("step_coarse_activation_patching")
def step_coarse_activation_patching(ctx: ExperimentContext) -> None:
    """Run layer and position sweeps on each contrastive pair."""
    ctx.coarse_agg = CoarseActPatchAggregatedResults()

    for pair_idx, pair in enumerate(ctx.pairs):
        print(f"\n[coarse] Processing pair {pair_idx + 1}/{len(ctx.pairs)}")
        result = run_coarse_act_patching(ctx.runner, pair)
        result.sample_id = pair_idx
        ctx.coarse_patching[pair_idx] = result
        ctx.coarse_agg.add(result)

    ctx.coarse_agg.print_summary()


@profile("step_fine_activation_patching")
def step_fine_activation_patching(ctx: ExperimentContext) -> None:
    """Run targeted activation patching on decomposed targets for each component."""
    ctx.fine_agg = ActPatchAggregatedResult()

    for component in COMPONENTS:
        target = ctx.get_union_target(component=component)
        targets = target.decompose()

        for pair_idx, pair in enumerate(ctx.pairs):
            print(
                f"\n[fine] Processing pair {pair_idx + 1}/{len(ctx.pairs)}, component={component}"
            )
            pair_result = patch_pair(ctx.runner, pair, targets)
            pair_result.sample_id = pair_idx
            ctx.fine_patching[pair_idx] = pair_result
            ctx.fine_agg.add(pair_result)

    ctx.fine_agg.print_summary()


def _save_trajectories(pair, pair_out_dir: Path) -> None:
    """Save short and long trajectories as JSON."""
    import json

    pair_out_dir.mkdir(parents=True, exist_ok=True)

    # Save short trajectory
    short_path = pair_out_dir / "short_token_tree.json"
    with open(short_path, "w") as f:
        json.dump(pair.short_traj.to_dict(), f, indent=2)

    # Save long trajectory
    long_path = pair_out_dir / "long_token_tree.json"
    with open(long_path, "w") as f:
        json.dump(pair.long_traj.to_dict(), f, indent=2)


@profile("step_visualize_results")
def step_visualize_results(ctx: ExperimentContext) -> None:
    """Visualize all patching results."""

    for pair_idx, pair in enumerate(ctx.pairs):
        pair_out_dir = ctx.viz_dir / f"pair_{pair_idx}"
        coloring = get_token_coloring_for_pair(pair, ctx.runner)
        position_labels = coloring.get_position_labels("short")
        section_markers = coloring.get_section_markers("short")

        # Save trajectories as JSON
        _save_trajectories(pair, pair_out_dir)

        # Tokenization visualization (single pair as list)
        visualize_tokenization([pair], ctx.runner, pair_out_dir, max_pairs=1)

        # Per-pair patching visualizations
        if pair_idx in ctx.att_patching:
            pair_result = ctx.att_patching[pair_idx]
            if pair_result.result.denoising:
                visualize_att_patching(
                    pair_result.result.denoising,
                    pair_out_dir / "denoising",
                    position_labels,
                    section_markers,
                )
            if pair_result.result.noising:
                visualize_att_patching(
                    pair_result.result.noising,
                    pair_out_dir / "noising",
                    position_labels,
                    section_markers,
                )
        if pair_idx in ctx.coarse_patching:
            visualize_coarse_patching(
                ctx.coarse_patching[pair_idx], pair_out_dir, coloring, pair=pair
            )
        if pair_idx in ctx.fine_patching:
            visualize_fine_patching(
                ctx.fine_patching[pair_idx],
                pair_out_dir,
                position_labels,
                section_markers,
            )

    # Aggregated visualizations
    agg_out_dir = ctx.viz_dir / "agg"
    if ctx.att_agg:
        visualize_att_patching(ctx.att_agg.denoising_agg, agg_out_dir / "denoising")
        visualize_att_patching(ctx.att_agg.noising_agg, agg_out_dir / "noising")
    visualize_coarse_patching(ctx.coarse_agg, agg_out_dir)
    visualize_fine_patching(ctx.fine_agg, agg_out_dir)


@profile("run_experiment")
def run_experiment(cfg: ExperimentConfig) -> ExperimentContext:
    """Run full experiment."""
    ctx = ExperimentContext(cfg)
    step_preference_data(ctx)

    if not ctx.pairs:
        print("No preference pairs!")
        return

    # step_attribution_patching(ctx)

    step_coarse_activation_patching(ctx)

    # step_fine_activation_patching(ctx)

    step_visualize_results(ctx)

    return ctx
