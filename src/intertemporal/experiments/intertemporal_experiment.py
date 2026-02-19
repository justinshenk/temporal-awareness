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
from ...activation_patching import patch_pair, ActPatchAggregatedResult

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
    visualize_logit_lens,
)
from .coarse_activation_patching import CoarseActPatchResults, run_coarse_act_patching
from .attribution_patching import run_attribution_patching


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
    coarse_patching: CoarseActPatchResults | None = None
    fine_patching: ActPatchAggregatedResult | None = None
    att_patching: object | None = None
    output_dir: Path = field(default_factory=get_experiment_dir)
    timestamp: str = field(default_factory=get_timestamp)
    _pairs: list | None = field(default=None, init=False)
    _runner: BinaryChoiceRunner | None = field(default=None, init=False)

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
        """Get union target from attribution or coarse patching."""
        # Use attribution patching results if available
        if self.att_patching:
            target = self.att_patching.get_union_target()
            if target:
                return target

        # Fall back to coarse patching results
        if self.coarse_patching:
            return self.coarse_patching.get_union_target(component=component)

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


@profile("step_coarse_activation_patching")
def step_coarse_activation_patching(ctx: ExperimentContext) -> None:
    """Run layer and position sweeps on best contrastive pair."""
    pair = ctx.best_contrastive_pair
    ctx.coarse_patching = run_coarse_act_patching(ctx.runner, pair)
    best_layers = ctx.coarse_patching.best_layers()
    best_pos = ctx.coarse_patching.best_n_positions()
    # Find best position by recovery
    pos_results = ctx.coarse_patching.position_results
    if pos_results:
        best_pos_by_score = max(
            pos_results.keys(), key=lambda p: pos_results[p].score()
        )
        best_pos_recovery = pos_results[best_pos_by_score].score()
        print(f"Best layers: {best_layers}")
        print(
            f"Best position range: {best_pos_by_score} (recovery={best_pos_recovery:.3f})"
        )
    else:
        print(f"Best layers: {best_layers}")


@profile("step_fine_activation_patching")
def step_fine_activation_patching(ctx: ExperimentContext) -> None:
    """Run targeted activation patching on decomposed targets for each component."""
    return
    result = ActPatchAggregatedResult()

    for component in COMPONENTS:
        target = ctx.get_union_target(component=component)
        targets = target.decompose()

        for pair in ctx.pairs:
            pair_result = patch_pair(ctx.runner, pair, targets)
            result.add(pair_result)

    ctx.fine_patching = result
    result.print_summary()


@profile("step_attribution_patching")
def step_attribution_patching(ctx: ExperimentContext) -> None:
    """Run attribution patching."""
    ctx.att_patching = run_attribution_patching(ctx.pref_data)


@profile("step_visualize_results")
def step_visualize_results(ctx: ExperimentContext) -> None:
    """Visualize all patching results."""
    visualize_att_patching(ctx.att_patching, ctx.viz_dir)
    visualize_coarse_patching(ctx.coarse_patching, ctx.viz_dir)
    visualize_fine_patching(ctx.fine_patching, ctx.viz_dir)
    visualize_tokenization(ctx.pairs, ctx.runner, ctx.viz_dir)
    visualize_logit_lens(ctx.pairs, ctx.runner, ctx.viz_dir)


@profile("run_experiment")
def run_experiment(cfg: ExperimentConfig) -> ExperimentContext:
    """Run full experiment."""
    ctx = ExperimentContext(cfg)
    step_preference_data(ctx)

    if not ctx.pairs:
        print("No preference pairs!")
        return

    # step_attribution_patching(ctx)
    # step_coarse_activation_patching(ctx)
    # step_fine_activation_patching(ctx)
    step_visualize_results(ctx)
    return ctx
