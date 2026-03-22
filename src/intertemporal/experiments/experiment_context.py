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
    get_recommended_backend_interventions,
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

from .diffmeans import DiffMeansPairResult, DiffMeansAggregatedResults
from .geo import GeoPairResult, GeoAggregatedResults
from .processing import ProcessedResults
from ..common import get_experiment_dir
from ..common.contrastive_utils import get_contrastive_preferences, PrefPairRequirement
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

    # Backend override (None = auto-detect via get_recommended_backend_interventions)
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

    # Diffmeans results
    diffmeans_patching: dict[int, DiffMeansPairResult] = field(default_factory=dict)
    diffmeans_agg: DiffMeansAggregatedResults | None = None

    # Geo (PCA) results
    geo_patching: dict[int, GeoPairResult] = field(default_factory=dict)
    geo_agg: GeoAggregatedResults | None = None

    # Processed results (computed in step_process_results)
    processed_results: ProcessedResults | None = None

    @property
    def runner(self) -> BinaryChoiceRunner:
        """Cached runner for this experiment."""
        if self._runner is None:
            # Use backend override if provided, otherwise auto-detect
            if self.backend:
                backend = ModelBackend(self.backend)
            else:
                backend = get_recommended_backend_interventions()
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
        # Build pair requirements from config
        pair_req = None
        if self.cfg.pair_req:
            pair_req = PrefPairRequirement.from_dict(self.cfg.pair_req)

        # Get all contrastive preferences
        all_prefs = get_contrastive_preferences(self.pref_data, req=pair_req)
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

    @property
    def pairs_dir(self) -> Path:
        """Get the directory containing all pair subdirectories."""
        return self.output_dir / "pairs"

    def get_pair_dir(self, pair_idx: int) -> Path:
        """Get the directory for a specific pair."""
        return self.pairs_dir / f"pair_{pair_idx}"

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
            self.get_pair_dir(pair_idx)
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

    def get_coarse_agg_dir(self) -> Path:
        """Get the directory for aggregated coarse patching results."""
        return self.output_dir / "agg_coarse"

    def save_coarse_agg(self) -> None:
        """Save all component aggregated results."""
        coarse_dir = self.get_coarse_agg_dir()
        coarse_dir.mkdir(parents=True, exist_ok=True)
        for component, agg in self.coarse_agg_by_component.items():
            path = coarse_dir / f"{component}.json"
            log(f"[coarse] Saving aggregated results to {path}...")
            agg.pop_heavy()
            save_json(agg.to_dict(), path)
        log("[coarse] Saved.")

    def load_coarse_agg(self, components: list[str]) -> bool:
        """Load aggregated results for specified components."""
        coarse_dir = self.get_coarse_agg_dir()
        any_loaded = False
        for component in components:
            path = coarse_dir / f"{component}.json"
            # Fallback to legacy paths
            if not path.exists():
                path = self.output_dir / "agg" / "coarse" / f"{component}.json"
            if not path.exists():
                path = self.output_dir / "coarse_agg" / f"{component}.json"
            if path.exists():
                self.coarse_agg_by_component[component] = (
                    CoarseActPatchAggregatedResults.from_json(path)
                )
                any_loaded = True
        return any_loaded

    def get_att_agg_dir(self) -> Path:
        """Get the directory for aggregated attribution results."""
        return self.output_dir / "agg_att"

    def get_att_agg_path(self) -> Path:
        """Legacy path for backwards compatibility check."""
        return self.output_dir / "att_agg.json"

    def save_att_agg(self) -> None:
        """Save aggregated attribution results to folder structure."""
        if not self.att_agg:
            return

        att_dir = self.get_att_agg_dir()
        att_dir.mkdir(parents=True, exist_ok=True)
        log(f"[attr] Saving aggregated results to {att_dir}...")

        # Save denoising and noising separately
        if self.att_agg.denoising_agg:
            save_json(
                self.att_agg.denoising_agg.to_dict(),
                att_dir / "denoising.json",
            )
        if self.att_agg.noising_agg:
            save_json(
                self.att_agg.noising_agg.to_dict(),
                att_dir / "noising.json",
            )

        # Also save the full aggregated result for compatibility
        save_json(self.att_agg.to_dict(), att_dir / "att_agg.json")
        log("[attr] Saved.")

    def load_att_agg(self) -> bool:
        """Load aggregated attribution results from folder or legacy path."""
        # Try new folder structure first
        att_dir = self.get_att_agg_dir()
        if (att_dir / "att_agg.json").exists():
            self.att_agg = AttrPatchAggregatedResults.from_json(
                att_dir / "att_agg.json"
            )
            return True

        # Fallback to legacy paths
        legacy_paths = [
            self.output_dir / "att_agg" / "att_agg.json",
            self.output_dir / "agg" / "att" / "att_agg.json",
            self.get_att_agg_path(),
        ]
        for path in legacy_paths:
            if path.exists():
                self.att_agg = AttrPatchAggregatedResults.from_json(path)
                return True

        return False

    def get_att_pair_dir(self, pair_idx: int) -> Path:
        """Get directory for per-pair attribution results."""
        return self.get_pair_dir(pair_idx) / "att_patching"

    def get_att_pair_path(self, pair_idx: int) -> Path:
        """Get path for per-pair attribution results JSON."""
        return self.get_att_pair_dir(pair_idx) / "att_results.json"

    def save_att_pair(self, pair_idx: int) -> None:
        """Save per-pair attribution results."""
        if pair_idx not in self.att_patching:
            return
        result = self.att_patching[pair_idx]
        path = self.get_att_pair_path(pair_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(result.to_dict(), path)

    def load_att_pair(self, pair_idx: int) -> bool:
        """Load per-pair attribution results."""
        path = self.get_att_pair_path(pair_idx)
        if path.exists():
            self.att_patching[pair_idx] = AttrPatchPairResult.from_json(path)
            return True
        return False

    def detect_cached_att_pairs_legacy(self) -> list[int]:
        """Detect which pairs have cached attribution results (legacy path)."""
        cached = []
        if not self.pairs_dir.exists():
            return cached
        for pair_dir in self.pairs_dir.glob("pair_*"):
            att_path = pair_dir / "att_patching" / "att_results.json"
            if att_path.exists():
                pair_idx = int(pair_dir.name.split("_")[1])
                cached.append(pair_idx)
        return sorted(cached)

    def get_contrastive_pref_path(self, pair_idx: int) -> Path:
        """Get path for per-pair contrastive preference JSON."""
        return self.get_pair_dir(pair_idx) / "contrastive_preference.json"

    def save_contrastive_pref(self, pair_idx: int) -> None:
        """Save contrastive preference summary for a pair.

        Uses to_summary_dict() to save only key properties, excluding heavy
        data like choice trees and trajectories.
        """
        pref = self.get_pref_pair(pair_idx)
        if pref is None:
            return
        path = self.get_contrastive_pref_path(pair_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(pref.to_summary_dict(), path)

    def save_all_contrastive_prefs(self) -> None:
        """Save contrastive preferences for all pairs."""
        for pair_idx in range(len(self.pairs)):
            self.save_contrastive_pref(pair_idx)
        log(f"[ctx] Saved contrastive preferences for {len(self.pairs)} pairs")

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

    # ─── Diffmeans save/load methods ───

    def get_diffmeans_pair_dir(self, pair_idx: int) -> Path:
        """Get directory for per-pair diffmeans results."""
        return self.get_pair_dir(pair_idx) / "diffmeans"

    def get_diffmeans_pair_path(self, pair_idx: int) -> Path:
        """Get path for per-pair diffmeans results JSON."""
        return self.get_diffmeans_pair_dir(pair_idx) / "diffmeans_results.json"

    def save_diffmeans_pair(self, pair_idx: int) -> None:
        """Save per-pair diffmeans results."""
        if pair_idx not in self.diffmeans_patching:
            return
        result = self.diffmeans_patching[pair_idx]
        path = self.get_diffmeans_pair_path(pair_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(result.to_dict(), path)

    def load_diffmeans_pair(self, pair_idx: int) -> bool:
        """Load per-pair diffmeans results."""
        path = self.get_diffmeans_pair_path(pair_idx)
        if path.exists():
            self.diffmeans_patching[pair_idx] = DiffMeansPairResult.from_json(path)
            return True
        return False

    def get_diffmeans_agg_dir(self) -> Path:
        """Get directory for aggregated diffmeans results."""
        return self.output_dir / "agg_diffmeans"

    def save_diffmeans_agg(self) -> None:
        """Save aggregated diffmeans results."""
        if not self.diffmeans_agg:
            return
        agg_dir = self.get_diffmeans_agg_dir()
        agg_dir.mkdir(parents=True, exist_ok=True)
        path = agg_dir / "diffmeans_agg.json"
        log(f"[diffmeans] Saving aggregated results to {path}...")
        save_json(self.diffmeans_agg.to_dict(), path)
        log("[diffmeans] Saved.")

    def load_diffmeans_agg(self) -> bool:
        """Load aggregated diffmeans results."""
        path = self.get_diffmeans_agg_dir() / "diffmeans_agg.json"
        # Fallback to legacy path
        if not path.exists():
            path = self.output_dir / "agg" / "diffmeans" / "diffmeans_agg.json"
        if path.exists():
            self.diffmeans_agg = DiffMeansAggregatedResults.from_json(path)
            return True
        return False

    def detect_cached_diffmeans_pairs(self) -> list[int]:
        """Detect all pair indices that have cached diffmeans results."""
        cached = []
        pair_idx = 0
        while True:
            pair_dir = self.get_pair_dir(pair_idx)
            if not pair_dir.exists():
                break
            if self.get_diffmeans_pair_path(pair_idx).exists():
                cached.append(pair_idx)
            pair_idx += 1
        return cached

    # ─── Geo save/load methods ───

    def get_geo_pair_dir(self, pair_idx: int) -> Path:
        """Get directory for per-pair geo results."""
        return self.get_pair_dir(pair_idx) / "geo"

    def get_geo_pair_path(self, pair_idx: int) -> Path:
        """Get path for per-pair geo results JSON."""
        return self.get_geo_pair_dir(pair_idx) / "geo_results.json"

    def save_geo_pair(self, pair_idx: int) -> None:
        """Save per-pair geo results."""
        if pair_idx not in self.geo_patching:
            return
        result = self.geo_patching[pair_idx]
        path = self.get_geo_pair_path(pair_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(result.to_dict(), path)

    def load_geo_pair(self, pair_idx: int) -> bool:
        """Load per-pair geo results."""
        path = self.get_geo_pair_path(pair_idx)
        if path.exists():
            self.geo_patching[pair_idx] = GeoPairResult.from_json(path)
            return True
        return False

    def get_geo_agg_dir(self) -> Path:
        """Get directory for aggregated geo results."""
        return self.output_dir / "agg_geo"

    def save_geo_agg(self) -> None:
        """Save aggregated geo results."""
        if not self.geo_agg:
            return
        agg_dir = self.get_geo_agg_dir()
        agg_dir.mkdir(parents=True, exist_ok=True)
        path = agg_dir / "geo_agg.json"
        log(f"[geo] Saving aggregated results to {path}...")
        save_json(self.geo_agg.to_dict(), path)
        log("[geo] Saved.")

    def load_geo_agg(self) -> bool:
        """Load aggregated geo results."""
        path = self.get_geo_agg_dir() / "geo_agg.json"
        if path.exists():
            self.geo_agg = GeoAggregatedResults.from_json(path)
            return True
        return False

    def detect_cached_geo_pairs(self) -> list[int]:
        """Detect all pair indices that have cached geo results."""
        cached = []
        pair_idx = 0
        while True:
            pair_dir = self.get_pair_dir(pair_idx)
            if not pair_dir.exists():
                break
            if self.get_geo_pair_path(pair_idx).exists():
                cached.append(pair_idx)
            pair_idx += 1
        return cached

    # ─── Unload methods for memory management ───

    def unload_att_agg(self) -> None:
        """Clear attribution aggregated results from memory."""
        self.att_agg = None
        self.att_patching.clear()

    def unload_coarse_agg(self, component: str | None = None) -> None:
        """Clear coarse aggregated results from memory.

        Args:
            component: Specific component to unload, or None for all
        """
        if component:
            self.coarse_agg_by_component.pop(component, None)
            # Also clear per-pair results for this component
            keys_to_remove = [k for k in self.coarse_patching if k[1] == component]
            for k in keys_to_remove:
                del self.coarse_patching[k]
        else:
            self.coarse_agg_by_component.clear()
            self.coarse_patching.clear()

    def unload_fine_agg(self) -> None:
        """Clear fine aggregated results from memory."""
        self.fine_agg = None
        self.fine_patching.clear()

    def unload_diffmeans_agg(self) -> None:
        """Clear diffmeans aggregated results from memory."""
        self.diffmeans_agg = None
        self.diffmeans_patching.clear()

    def unload_all(self) -> None:
        """Clear all aggregated results from memory."""
        self.unload_att_agg()
        self.unload_coarse_agg()
        self.unload_fine_agg()
        self.unload_diffmeans_agg()

    # ─── Cache detection methods ───

    def detect_cached_coarse_pairs(self, component: str) -> list[int]:
        """Detect all pair indices that have cached coarse results for a component.

        Args:
            component: Component name (e.g., 'resid_post')

        Returns:
            List of pair indices with cached results
        """
        cached = []
        pair_idx = 0
        while True:
            pair_dir = self.get_pair_dir(pair_idx)
            if not pair_dir.exists():
                break
            results_path = pair_dir / f"sweep_{component}" / "coarse_results.json"
            if results_path.exists():
                cached.append(pair_idx)
            pair_idx += 1
        return cached

    def detect_cached_att_pairs(self) -> list[int]:
        """Detect all pair indices that have cached attribution results.

        Returns:
            List of pair indices with cached results
        """
        cached = []
        pair_idx = 0
        while True:
            pair_dir = self.get_pair_dir(pair_idx)
            if not pair_dir.exists():
                break
            # Check for att_patching subdirectory with any results
            att_dir = pair_dir / "att_patching"
            if att_dir.exists() and any(att_dir.iterdir()):
                cached.append(pair_idx)
            pair_idx += 1
        return cached

    def detect_cached_components(self) -> list[str]:
        """Detect which components have cached coarse patching results.

        Returns:
            List of component names with cached results
        """
        components = []
        pair_0 = self.get_pair_dir(0)
        if pair_0.exists():
            for d in pair_0.iterdir():
                if d.is_dir() and d.name.startswith("sweep_"):
                    comp = d.name.replace("sweep_", "")
                    if (d / "coarse_results.json").exists():
                        components.append(comp)
        return components

    # ─── Processed results save/load methods ───

    def get_processed_results_path(self) -> Path:
        """Get path for processed results JSON."""
        return self.output_dir / "processed_results.json"

    def save_processed_results(self) -> None:
        """Save processed results to disk."""
        if not self.processed_results:
            return
        path = self.get_processed_results_path()
        log(f"[process] Saving processed results to {path}...")
        save_json(self.processed_results.to_dict(), path)
        log("[process] Saved.")
