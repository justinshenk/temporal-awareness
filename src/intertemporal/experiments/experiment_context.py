"""Experiment context for intertemporal experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ...common import ensure_dir, get_timestamp, get_device
from ...common.file_io import save_json
from ...common.logging import log, log_progress
from ...common.token_tree import TokenTree
from ...common.contrastive_pair import ContrastivePair
from ...inference import (
    get_recommended_backend_internals,
    InterventionTarget,
)
from ...inference.backends import ModelBackend
from ...binary_choice import BinaryChoiceRunner
from ...activation_patching import (
    ActPatchAggregatedResult,
    ActPatchPairResult,
)
from ...activation_patching.coarse import (
    CoarseActPatchResults,
    CoarseActPatchAggregatedResults,
)
from ...attribution_patching import AttrPatchPairResult, AttrPatchAggregatedResults

from ..common import get_experiment_dir
from ..common.contrastive_utils import get_contrastive_preferences
from ..common.contrastive_preferences import ContrastivePreferences
from ..preference import PreferenceDataset
from .experiment_config import ExperimentConfig


@dataclass
class ExperimentContext:
    """Shared experiment state."""

    cfg: ExperimentConfig
    pref_data: PreferenceDataset | None = None

    output_dir: Path | None = field(default=None)
    timestamp: str = field(default_factory=get_timestamp)

    # Backend override (None = auto-detect via get_recommended_backend_internals)
    backend: str | None = None

    def __post_init__(self) -> None:
        if self.output_dir is None:
            self.output_dir = get_experiment_dir() / self.cfg.get_id()

    _pairs: list | None = field(default=None, init=False)
    _runner: BinaryChoiceRunner | None = field(default=None, init=False)
    _pref_pairs: list[ContrastivePreferences] | None = field(default=None, init=False)
    _pair_to_pref_idx: dict[int, int] = field(default_factory=dict, init=False)

    # coarse_patching keyed by (pair_idx, component) tuple
    coarse_patching: dict[tuple[int, str], CoarseActPatchResults] = field(
        default_factory=dict
    )
    fine_patching: dict[int, ActPatchPairResult] = field(default_factory=dict)
    att_patching: dict[int, AttrPatchPairResult] = field(default_factory=dict)

    coarse_agg_by_component: dict[str, CoarseActPatchAggregatedResults] = field(
        default_factory=dict
    )
    fine_agg: ActPatchAggregatedResult | None = None
    att_agg: AttrPatchAggregatedResults | None = None

    @property
    def runner(self) -> BinaryChoiceRunner:
        """Cached runner for this experiment."""
        if self._runner is None:
            # Use backend override if provided, otherwise auto-detect
            if self.backend:
                backend = ModelBackend(self.backend)
            else:
                backend = get_recommended_backend_internals()
            self._runner = BinaryChoiceRunner(
                self.pref_data.model, device=get_device(), backend=backend
            )
        return self._runner

    @property
    def pairs(self) -> list[ContrastivePair]:
        """Contrastive pairs (cached)."""
        if self._pairs is None:
            self._build_pairs()
        return self._pairs

    def _build_pairs(self) -> None:
        """Build contrastive pairs from preference data."""
        # Get all contrastive preferences
        all_prefs = get_contrastive_preferences(self.pref_data)
        n_select = self.cfg.n_pairs or len(all_prefs)
        selected = all_prefs[:n_select]
        log(f"[ctx] Found {len(all_prefs)} contrastive prefs, using {len(selected)}")

        # Get position mapping config from prompt format
        fmt = self.pref_data.prompt_format_config
        anchor_texts = fmt.get_anchor_texts()
        first_interesting = fmt.get_prompt_marker_before_time_horizon()

        # Build pairs
        self._pairs = []
        self._pref_pairs = []
        self._pair_to_pref_idx = {}

        for i, pref in enumerate(selected):
            log_progress(i + 1, len(selected), prefix="[ctx] Building pair ")
            pair = pref.get_contrastive_pair(
                self.runner,
                anchor_texts=anchor_texts,
                first_interesting_marker=first_interesting,
            )
            if pair:
                idx = len(self._pairs)
                self._pairs.append(pair)
                self._pref_pairs.append(pref)
                self._pair_to_pref_idx[idx] = idx

        log(f"[ctx] Built {len(self._pairs)} valid pairs")

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
        if self.att_agg:
            target = self.att_agg.get_target()
            if target:
                return target

        if component in self.coarse_agg_by_component:
            return self.coarse_agg_by_component[component].get_union_target(
                component=component
            )

        return InterventionTarget.all(component=component)

    def get_runner(self) -> BinaryChoiceRunner:
        """Get the cached runner (alias for runner property)."""
        return self.runner

    @property
    def best_contrastive_pair(self):
        """First contrastive pair (for coarse patching)."""
        return self.pairs[0] if self.pairs else None

    def save_token_trees(
        self, pair_idx: int, pair: ContrastivePair, output_dir: Path
    ) -> None:
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

        # Build combined tree with both trajectories
        tree = TokenTree.from_trajectories(
            [pair.clean_traj, pair.corrupted_traj],
            groups_per_traj=[[0], [1]],
            fork_arms=[(0, 1)],
        )
        tree.pop_heavy()

        with open(output_dir / "token_tree.json", "w") as f:
            json.dump(tree.to_dict(), f, indent=2)

    # ─── Save/Load methods for cached results ───

    def get_coarse_pair_path(self, pair_idx: int, component: str) -> Path:
        return (
            self.output_dir
            / f"pair_{pair_idx}"
            / f"sweep_{component}"
            / "coarse_results.json"
        )

    def save_coarse_pair(self, pair_idx: int, component: str) -> None:
        """Save per-pair coarse patching results for re-visualization."""
        key = (pair_idx, component)
        if key in self.coarse_patching:
            result = self.coarse_patching[key]
            path = self.get_coarse_pair_path(pair_idx, component)
            path.parent.mkdir(parents=True, exist_ok=True)
            result.pop_heavy()
            save_json(result.to_dict(), path)

    def load_coarse_pair(self, pair_idx: int, component: str) -> bool:
        """Load per-pair coarse patching results."""
        path = self.get_coarse_pair_path(pair_idx, component)
        if path.exists():
            self.coarse_patching[(pair_idx, component)] = (
                CoarseActPatchResults.from_json(path)
            )
            return True
        return False

    def save_coarse_agg(self) -> None:
        """Save all component aggregated results."""
        for component, agg in self.coarse_agg_by_component.items():
            path = self.output_dir / f"coarse_agg_{component}.json"
            ensure_dir(path.parent)
            log(f"[coarse] Saving aggregated results to {path}...")
            agg.pop_heavy()
            save_json(agg.to_dict(), path)
        log("[coarse] Saved.")

    def load_coarse_agg(self, components: list[str]) -> bool:
        """Load aggregated results for specified components."""
        any_loaded = False
        for component in components:
            path = self.output_dir / f"coarse_agg_{component}.json"
            if path.exists():
                self.coarse_agg_by_component[component] = (
                    CoarseActPatchAggregatedResults.from_json(path)
                )
                any_loaded = True
        return any_loaded

    def get_att_agg_path(self) -> Path:
        return self.output_dir / "att_agg.json"

    def save_att_agg(self) -> None:
        if self.att_agg:
            path = self.get_att_agg_path()
            log(f"[attr] Saving aggregated results to {path}...")
            save_json(self.att_agg.to_dict(), path)
            log("[attr] Saved.")

    def load_att_agg(self) -> bool:
        path = self.get_att_agg_path()
        if path.exists():
            self.att_agg = AttrPatchAggregatedResults.from_json(path)
            return True
        return False

    def get_fine_agg_path(self) -> Path:
        return self.output_dir / "fine_agg.json"

    def save_fine_agg(self) -> None:
        if self.fine_agg:
            path = self.get_fine_agg_path()
            log(f"[fine] Saving aggregated results to {path}...")
            save_json(self.fine_agg.to_dict(), path)
            log("[fine] Saved.")

    def load_fine_agg(self) -> bool:
        path = self.get_fine_agg_path()
        if path.exists():
            self.fine_agg = ActPatchAggregatedResult.from_json(path)
            return True
        return False
