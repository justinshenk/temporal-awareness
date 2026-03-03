"""Experiment configuration and context for intertemporal experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ...common import ensure_dir, get_timestamp, get_device, BaseSchema
from ...common.token_tree import TokenTree
from ...common.contrastive_pair import ContrastivePair
from ...common.analysis.analyze import analyze_token_tree
from ...inference import (
    get_recommended_backend_internals,
    InterventionTarget,
)
from ...binary_choice import BinaryChoiceRunner
from ...activation_patching import (
    ActPatchAggregatedResult,
    ActPatchPairResult,
)

from ..common import get_experiment_dir
from ..common.contrastive_preferences import (
    get_contrastive_preferences,
    ContrastivePreferences,
)
from ..preference import PreferenceDataset
from ..prompt import PromptDatasetConfig
from ...activation_patching.coarse_activation_patching import (
    CoarseActPatchResults,
    CoarseActPatchAggregatedResults,
)
from ...attribution_patching import AttrPatchPairResult, AttrPatchAggregatedResults


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
    _pref_pairs: list[ContrastivePreferences] | None = field(default=None, init=False)
    _pair_to_pref_idx: dict[int, int] = field(default_factory=dict, init=False)

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
            all_pref_pairs = get_contrastive_preferences(self.pref_data)
            print(
                f"[ctx] Found {len(all_pref_pairs)} contrastive preferences, selecting {self.cfg.n_pairs}"
            )
            selected = all_pref_pairs[: self.cfg.n_pairs]
            anchor_texts = self.pref_data.prompt_format_config.get_anchor_texts()
            first_interesting_marker = self.pref_data.prompt_format_config.get_prompt_marker_before_time_horizon()
            print("[ctx] Building contrastive pairs...")
            self._pairs = []
            self._pref_pairs = []
            self._pair_to_pref_idx = {}
            for i, p in enumerate(selected):
                if p:
                    print(f"[ctx]   Building pair {i + 1}/{len(selected)}...")
                    pair = p.get_contrastive_pair(
                        self.runner,
                        anchor_texts=anchor_texts,
                        first_interesting_marker=first_interesting_marker,
                    )
                    if pair:
                        pair_idx = len(self._pairs)
                        pref_idx = len(self._pref_pairs)
                        self._pairs.append(pair)
                        self._pref_pairs.append(p)
                        self._pair_to_pref_idx[pair_idx] = pref_idx
            print(f"[ctx] Built {len(self._pairs)} valid pairs")
        return self._pairs

    @property
    def pref_pairs(self) -> list[ContrastivePreferences]:
        """ContrastivePreferences objects corresponding to pairs (cached)."""
        if self._pref_pairs is None:
            _ = self.pairs  # Trigger lazy loading
        return self._pref_pairs or []

    def get_pref_pair(self, pair_idx: int) -> ContrastivePreferences | None:
        """Get the ContrastivePreferences object for a given pair index.

        Args:
            pair_idx: Index into self.pairs

        Returns:
            The corresponding ContrastivePreferences, or None if not found
        """
        if self._pref_pairs is None:
            _ = self.pairs  # Trigger lazy loading
        pref_idx = self._pair_to_pref_idx.get(pair_idx)
        if pref_idx is not None and self._pref_pairs:
            return self._pref_pairs[pref_idx]
        return None

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
        """Get the cached runner (alias for runner property)."""
        return self.runner

    @property
    def best_contrastive_pair(self):
        """First contrastive pair (for coarse patching)."""
        return self.pairs[0] if self.pairs else None

    def save_token_trees(self, pair_idx: int, pair: ContrastivePair, output_dir: Path) -> None:
        """Save analyzed TokenTree for a contrastive pair.

        Creates a combined tree with both short and long trajectories,
        analyzed with fork metrics at the decision point.

        Args:
            pair_idx: Index of the pair
            pair: The ContrastivePair to save
            output_dir: Directory to save the token tree JSON
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build combined tree with both trajectories and fork analysis
        tree = TokenTree.from_trajectories(
            [pair.short_traj, pair.long_traj],
            groups_per_traj=[[0], [1]],
            fork_arms=[(0, 1)],
        )
        analyze_token_tree(tree)
        tree.pop_heavy()

        with open(output_dir / "token_tree.json", "w") as f:
            json.dump(tree.to_dict(), f, indent=2)
