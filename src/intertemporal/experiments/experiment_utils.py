"""Experiment utilities: storage mixin and step helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ...common.file_io import save_json
from ...common.logging import log
from ...activation_patching.coarse import (
    CoarseActPatchResults,
    CoarseActPatchAggregatedResults,
)
from ...attribution_patching import AttrPatchPairResult, AttrPatchAggregatedResults
from .diffmeans import DiffMeansPairResult, DiffMeansAggregatedResults
from .mlp import MLPPairResult, MLPAggregatedResults
from .attn import AttnPairResult, AttnAggregatedResults
from .fine import FineResults, FineAggregatedResults, visualize_fine
from ..viz import visualize_all_att_aggregated_slices
from .coarse.coarse_viz import (
    visualize_all_aggregated,
    visualize_coarse_patching,
    visualize_component_comparison,
)
from .diffmeans.diffmeans_viz import visualize_diffmeans, visualize_diffmeans_pair
from .mlp.mlp_viz import visualize_mlp_analysis, visualize_mlp_pair, visualize_all_mlp_slices
from .attn.attn_viz import visualize_attn_analysis, visualize_attn_pair, visualize_all_attn_slices
from ..viz.att_patching_viz import visualize_att_patching

if TYPE_CHECKING:
    from .experiment_context import ExperimentContext
    from .analysis import ProcessedResults
    from .diffmeans import DiffMeansConfig


class ExperimentMixin:
    """Mixin providing save/load/detect methods for experiment results.

    Sections ordered to match run_experiment steps:
    1. attrib, 2. coarse, 3. diffmeans, 4. mlp, 5. attn, 6. fine
    """

    output_dir: Path
    processed_results: "ProcessedResults | None"

    @property
    def agg_dir(self) -> Path:
        return self.output_dir / "aggregated"

    # Field declarations (ordered to match run_experiment)
    attrib_patching: dict
    attrib_agg: AttrPatchAggregatedResults | None
    coarse_patching: dict
    coarse_agg_by_component: dict
    diffmeans_patching: dict
    diffmeans_agg: DiffMeansAggregatedResults | None
    mlp: dict
    mlp_agg: MLPAggregatedResults | None
    attn: dict
    attn_agg: AttnAggregatedResults | None
    fine: dict
    fine_agg: FineAggregatedResults | None

    def get_pair_dir(self, pair_idx: int) -> Path:
        raise NotImplementedError

    # ─── Unified cache checking ───

    def is_pair_cached(self, step: str, pair_idx: int, component: str | None = None) -> bool:
        """Check if a pair is cached for a given step.

        Args:
            step: Step name (attrib, coarse, diffmeans, mlp, attn, fine)
            pair_idx: Pair index
            component: Component name (required for coarse step)

        Returns:
            True if cached results exist for this pair/step
        """
        if step == "attrib":
            return self.get_attrib_pair_path(pair_idx).exists()
        elif step == "coarse":
            if component is None:
                raise ValueError("coarse step requires component")
            return self.get_coarse_pair_path(pair_idx, component).exists()
        elif step == "diffmeans":
            return self.get_diffmeans_pair_path(pair_idx).exists()
        elif step == "mlp":
            return self.get_mlp_pair_path(pair_idx).exists()
        elif step == "attn":
            return self.get_attn_pair_path(pair_idx).exists()
        elif step == "fine":
            return self.get_fine_pair_path(pair_idx).exists()
        else:
            raise ValueError(f"Unknown step: {step}")

    # ─── Processed results ───

    def get_analysis_dir(self) -> Path:
        """Get directory for analysis results (processed, horizon, pair)."""
        return self.agg_dir / "analysis"

    def get_processed_results_path(self) -> Path:
        """Get path for processed results JSON."""
        return self.get_analysis_dir() / "processed_results.json"

    def save_processed_results(self) -> None:
        """Save processed results to disk."""
        if not self.processed_results:
            return
        path = self.get_processed_results_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        log(f"[process] Saving processed results to {path}...")
        save_json(self.processed_results.to_dict(), path)
        log("[process] Saved.")

    # ─── Attrib ───

    def get_attrib_agg_dir(self) -> Path:
        """Get the directory for aggregated attribution results."""
        return self.agg_dir / "attrib"

    def get_attrib_pair_dir(self, pair_idx: int) -> Path:
        """Get directory for per-pair attribution results."""
        return self.get_pair_dir(pair_idx) / "attrib"

    def get_attrib_pair_path(self, pair_idx: int) -> Path:
        """Get path for per-pair attribution results JSON."""
        return self.get_attrib_pair_dir(pair_idx) / "attrib_results.json"

    def save_attrib_pair(self, pair_idx: int) -> None:
        """Save per-pair attribution results."""
        if pair_idx not in self.attrib_patching:
            return
        result = self.attrib_patching[pair_idx]
        path = self.get_attrib_pair_path(pair_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(result.to_dict(), path)

    def load_attrib_pair(self, pair_idx: int) -> bool:
        """Load per-pair attribution results."""
        path = self.get_attrib_pair_path(pair_idx)
        if path.exists():
            self.attrib_patching[pair_idx] = AttrPatchPairResult.from_json(path)
            return True
        return False

    def save_attrib_agg(self) -> None:
        """Save aggregated attribution results to folder structure."""
        if not self.attrib_agg:
            return
        att_dir = self.get_attrib_agg_dir()
        att_dir.mkdir(parents=True, exist_ok=True)
        log(f"[attrib] Saving aggregated results to {att_dir}...")
        if self.attrib_agg.denoising_agg:
            save_json(self.attrib_agg.denoising_agg.to_dict(), att_dir / "denoising.json")
        if self.attrib_agg.noising_agg:
            save_json(self.attrib_agg.noising_agg.to_dict(), att_dir / "noising.json")
        save_json(self.attrib_agg.to_dict(), att_dir / "attrib_agg.json")
        # Note: Don't clear here - viz needs the data. Memory freed by unload_attrib_agg()
        log("[attrib] Saved.")

    def load_attrib_agg(self, _=None) -> bool:
        """Load aggregated attribution results."""
        path = self.get_attrib_agg_dir() / "attrib_agg.json"
        if path.exists():
            self.attrib_agg = AttrPatchAggregatedResults.from_json(path)
            return True
        return False

    def detect_cached_attrib_pairs(self) -> list[int]:
        """Detect all pair indices that have cached attribution results."""
        cached = []
        pair_idx = 0
        while True:
            pair_dir = self.get_pair_dir(pair_idx)
            if not pair_dir.exists():
                break
            if self.get_attrib_pair_path(pair_idx).exists():
                cached.append(pair_idx)
            pair_idx += 1
        return cached

    def unload_attrib_agg(self) -> None:
        """Clear attribution aggregated results from memory."""
        self.attrib_agg = None
        self.attrib_patching.clear()

    # ─── Coarse ───

    def get_coarse_agg_dir(self) -> Path:
        """Get the directory for aggregated coarse patching results."""
        return self.agg_dir / "coarse"

    def get_coarse_pair_path(self, pair_idx: int, component: str) -> Path:
        return self.get_pair_dir(pair_idx) / "coarse" / f"sweep_{component}" / "coarse_results.json"

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
            self.coarse_patching[(pair_idx, component)] = CoarseActPatchResults.from_json(path)
            return True
        return False

    def save_coarse_agg(self) -> None:
        """Save all component aggregated results."""
        coarse_dir = self.get_coarse_agg_dir()
        coarse_dir.mkdir(parents=True, exist_ok=True)
        for component, agg in self.coarse_agg_by_component.items():
            path = coarse_dir / f"{component}.json"
            log(f"[coarse] Saving aggregated results to {path}...")
            agg.pop_heavy()
            save_json(agg.to_dict(), path)
            # Note: Don't clear by_sample here - it causes empty aggregated results
            # Memory is managed by pop_heavy() and unload_coarse_agg() instead
        log("[coarse] Saved.")

    def load_coarse_agg(self, config=None) -> bool:
        """Load aggregated results for ALL cached components.

        Always loads all cached components, not just config.components.
        This ensures visualization includes all available data even when
        running with a subset of components.
        """
        # Always load ALL cached components, not just config.components
        components = self.detect_cached_components()
        coarse_dir = self.get_coarse_agg_dir()
        any_loaded = False
        for component in components:
            path = coarse_dir / f"{component}.json"
            if path.exists():
                self.coarse_agg_by_component[component] = CoarseActPatchAggregatedResults.from_json(path)
                any_loaded = True
        return any_loaded

    def rebuild_coarse_agg_from_pairs(self) -> None:
        """Rebuild ALL coarse aggregates from per-pair cached data.

        This ensures aggregated data is consistent and includes all pairs,
        regardless of which components were in the original config.
        Call this before saving/visualizing aggregated results.
        """
        components = self.detect_cached_components()
        if not components:
            log("[coarse] No cached components found for rebuild")
            return

        log(f"[coarse] Rebuilding aggregates from per-pair data for: {components}")

        # Clear existing aggregators and rebuild from scratch
        self.coarse_agg_by_component.clear()

        for component in components:
            agg = CoarseActPatchAggregatedResults()
            cached_pairs = self.detect_cached_coarse_pairs(component)
            log(f"[coarse] {component}: found {len(cached_pairs)} cached pairs")

            for pair_idx in cached_pairs:
                if self.load_coarse_pair(pair_idx, component):
                    key = (pair_idx, component)
                    agg.add(self.coarse_patching[key])
                    del self.coarse_patching[key]

            if agg.n_samples > 0:
                self.coarse_agg_by_component[component] = agg

        log(f"[coarse] Rebuild complete: {list(self.coarse_agg_by_component.keys())}")

    def detect_cached_coarse_pairs(self, component: str) -> list[int]:
        """Detect all pair indices that have cached coarse results for a component."""
        cached = []
        pair_idx = 0
        while True:
            pair_dir = self.get_pair_dir(pair_idx)
            if not pair_dir.exists():
                break
            if (pair_dir / "coarse" / f"sweep_{component}" / "coarse_results.json").exists():
                cached.append(pair_idx)
            pair_idx += 1
        return cached

    def detect_cached_components(self) -> list[str]:
        """Detect which components have cached coarse patching results."""
        components = []
        pair_0 = self.get_pair_dir(0)
        if pair_0.exists():
            coarse_dir = pair_0 / "coarse"
            if coarse_dir.exists():
                for d in coarse_dir.iterdir():
                    if d.is_dir() and d.name.startswith("sweep_"):
                        comp = d.name.replace("sweep_", "")
                        if (d / "coarse_results.json").exists():
                            components.append(comp)
        return components

    def get_all_cached_components_for_pair(self, pair_idx: int) -> list[str]:
        """Get all components with cached coarse patching data for a specific pair.

        Args:
            pair_idx: Pair index

        Returns:
            List of component names that have cached results for this pair
        """
        components = []
        pair_dir = self.get_pair_dir(pair_idx)
        if pair_dir.exists():
            coarse_dir = pair_dir / "coarse"
            if coarse_dir.exists():
                for d in coarse_dir.iterdir():
                    if d.is_dir() and d.name.startswith("sweep_"):
                        comp = d.name.replace("sweep_", "")
                        if (d / "coarse_results.json").exists():
                            components.append(comp)
        return components

    def unload_coarse_agg(self, component: str | None = None) -> None:
        """Clear coarse aggregated results from memory."""
        if component:
            self.coarse_agg_by_component.pop(component, None)
            keys_to_remove = [k for k in self.coarse_patching if k[1] == component]
            for k in keys_to_remove:
                del self.coarse_patching[k]
        else:
            self.coarse_agg_by_component.clear()
            self.coarse_patching.clear()

    # ─── Diffmeans ───

    def get_diffmeans_agg_dir(self) -> Path:
        """Get directory for aggregated diffmeans results."""
        return self.agg_dir / "diffmeans"

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

    def save_diffmeans_agg(self) -> None:
        """Save aggregated diffmeans results."""
        if not self.diffmeans_agg:
            return
        agg_dir = self.get_diffmeans_agg_dir()
        agg_dir.mkdir(parents=True, exist_ok=True)
        path = agg_dir / "diffmeans_agg.json"
        log(f"[diffmeans] Saving aggregated results to {path}...")
        save_json(self.diffmeans_agg.to_dict(), path)
        # Note: Don't clear here - viz needs the data. Memory freed by unload_diffmeans_agg()
        log("[diffmeans] Saved.")

    def load_diffmeans_agg(self, _=None) -> bool:
        """Load aggregated diffmeans results."""
        path = self.get_diffmeans_agg_dir() / "diffmeans_agg.json"
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

    def unload_diffmeans_agg(self) -> None:
        """Clear diffmeans aggregated results from memory."""
        self.diffmeans_agg = None
        self.diffmeans_patching.clear()

    # ─── MLP ───

    def get_mlp_agg_dir(self) -> Path:
        """Get directory for aggregated MLP results."""
        return self.agg_dir / "mlp"

    def get_mlp_pair_dir(self, pair_idx: int) -> Path:
        """Get directory for per-pair MLP results."""
        return self.get_pair_dir(pair_idx) / "mlp"

    def get_mlp_pair_path(self, pair_idx: int) -> Path:
        """Get path for per-pair MLP results JSON."""
        return self.get_mlp_pair_dir(pair_idx) / "mlp.json"

    def save_mlp_pair(self, pair_idx: int) -> None:
        """Save per-pair MLP results."""
        if pair_idx not in self.mlp:
            return
        result = self.mlp[pair_idx]
        path = self.get_mlp_pair_path(pair_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(result.to_dict(), path)

    def load_mlp_pair(self, pair_idx: int) -> bool:
        """Load per-pair MLP results."""
        path = self.get_mlp_pair_path(pair_idx)
        if path.exists():
            self.mlp[pair_idx] = MLPPairResult.from_json(path)
            return True
        return False

    def save_mlp_agg(self) -> None:
        """Save aggregated MLP results."""
        if not self.mlp_agg:
            return
        agg_dir = self.get_mlp_agg_dir()
        agg_dir.mkdir(parents=True, exist_ok=True)
        path = agg_dir / "mlp_agg.json"
        log(f"[mlp] Saving aggregated results to {path}...")
        save_json(self.mlp_agg.to_dict(), path)
        # Note: Don't clear here - viz needs the data. Memory freed by unload_mlp_agg()
        log("[mlp] Saved.")

    def load_mlp_agg(self, _=None) -> bool:
        """Load aggregated MLP results."""
        path = self.get_mlp_agg_dir() / "mlp_agg.json"
        if path.exists():
            self.mlp_agg = MLPAggregatedResults.from_json(path)
            return True
        return False

    def detect_cached_mlp_pairs(self) -> list[int]:
        """Detect all pair indices that have cached MLP results."""
        cached = []
        pair_idx = 0
        while True:
            pair_dir = self.get_pair_dir(pair_idx)
            if not pair_dir.exists():
                break
            if self.get_mlp_pair_path(pair_idx).exists():
                cached.append(pair_idx)
            pair_idx += 1
        return cached

    def unload_mlp_agg(self) -> None:
        """Clear MLP results from memory."""
        self.mlp_agg = None
        self.mlp.clear()

    # ─── Attn ───

    def get_attn_agg_dir(self) -> Path:
        """Get directory for aggregated attention results."""
        return self.agg_dir / "attn"

    def get_attn_pair_dir(self, pair_idx: int) -> Path:
        """Get directory for per-pair attention results."""
        return self.get_pair_dir(pair_idx) / "attn"

    def get_attn_pair_path(self, pair_idx: int) -> Path:
        """Get path for per-pair attention results JSON."""
        return self.get_attn_pair_dir(pair_idx) / "attn_results.json"

    def save_attn_pair(self, pair_idx: int, store_patterns: bool = False) -> None:
        """Save per-pair attention results."""
        if pair_idx not in self.attn:
            return
        result = self.attn[pair_idx]
        path = self.get_attn_pair_path(pair_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not store_patterns:
            result.pop_heavy()
        save_json(result.to_dict(), path)

    def load_attn_pair(self, pair_idx: int) -> bool:
        """Load per-pair attention results."""
        path = self.get_attn_pair_path(pair_idx)
        if path.exists():
            self.attn[pair_idx] = AttnPairResult.from_json(path)
            return True
        return False

    def save_attn_agg(self) -> None:
        """Save aggregated attention results."""
        if not self.attn_agg:
            return
        agg_dir = self.get_attn_agg_dir()
        agg_dir.mkdir(parents=True, exist_ok=True)
        path = agg_dir / "attn_agg.json"
        log(f"[attn] Saving aggregated results to {path}...")
        save_json(self.attn_agg.to_dict(), path)
        # Note: Don't clear here - viz needs the data. Memory freed by unload_attn_agg()
        log("[attn] Saved.")

    def load_attn_agg(self, _=None) -> bool:
        """Load aggregated attention results."""
        path = self.get_attn_agg_dir() / "attn_agg.json"
        if path.exists():
            self.attn_agg = AttnAggregatedResults.from_json(path)
            return True
        return False

    def detect_cached_attn_pairs(self) -> list[int]:
        """Detect all pair indices that have cached attention results."""
        cached = []
        pair_idx = 0
        while True:
            pair_dir = self.get_pair_dir(pair_idx)
            if not pair_dir.exists():
                break
            if self.get_attn_pair_path(pair_idx).exists():
                cached.append(pair_idx)
            pair_idx += 1
        return cached

    def unload_attn_agg(self) -> None:
        """Clear attention results from memory."""
        self.attn_agg = None
        self.attn.clear()

    # ─── Fine ───

    def get_fine_agg_dir(self) -> Path:
        """Get directory for aggregated fine-grained results."""
        return self.agg_dir / "fine"

    def get_fine_pair_dir(self, pair_idx: int) -> Path:
        """Get directory for per-pair fine-grained patching results."""
        return self.get_pair_dir(pair_idx) / "fine"

    def get_fine_pair_path(self, pair_idx: int) -> Path:
        """Get path for per-pair fine-grained patching results JSON."""
        return self.get_fine_pair_dir(pair_idx) / "fine.json"

    def save_fine_pair(self, pair_idx: int) -> None:
        """Save per-pair fine-grained patching results."""
        if pair_idx not in self.fine:
            return
        result = self.fine[pair_idx]
        path = self.get_fine_pair_path(pair_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(result.to_dict(), path)

    def load_fine_pair(self, pair_idx: int) -> bool:
        """Load per-pair fine-grained patching results."""
        path = self.get_fine_pair_path(pair_idx)
        if path.exists():
            self.fine[pair_idx] = FineResults.from_json(path)
            return True
        return False

    def save_fine_agg(self) -> None:
        """Save aggregated fine-grained results."""
        if not self.fine_agg:
            return
        agg_dir = self.get_fine_agg_dir()
        agg_dir.mkdir(parents=True, exist_ok=True)
        path = agg_dir / "fine_agg.json"
        log(f"[fine] Saving aggregated results to {path}...")
        save_json(self.fine_agg.to_dict(), path)
        # Note: Don't clear here - viz needs the data. Memory freed by unload_fine_agg()
        log("[fine] Saved.")

    def load_fine_agg(self, _=None) -> bool:
        """Load aggregated fine-grained results."""
        path = self.get_fine_agg_dir() / "fine_agg.json"
        if path.exists():
            self.fine_agg = FineAggregatedResults.from_json(path)
            return True
        return False

    def detect_cached_fine_pairs(self) -> list[int]:
        """Detect all pair indices that have cached fine-grained results."""
        cached = []
        pair_idx = 0
        while True:
            pair_dir = self.get_pair_dir(pair_idx)
            if not pair_dir.exists():
                break
            if self.get_fine_pair_path(pair_idx).exists():
                cached.append(pair_idx)
            pair_idx += 1
        return cached

    def unload_fine_agg(self) -> None:
        """Clear fine-grained patching results from memory."""
        self.fine_agg = None
        self.fine.clear()

    def load_fine_for_viz(self, _) -> bool:
        """Load cached fine-grained results for visualization."""
        cached = self.detect_cached_fine_pairs()
        if not cached:
            return False
        for pair_idx in cached:
            self.load_fine_pair(pair_idx)
        return True

    # ─── Visualization ───

    def make_attrib_viz(self, _) -> Callable[[], None]:
        """Return visualization function for attribution results."""
        return lambda: visualize_all_att_aggregated_slices(
            self.attrib_agg,
            self.agg_dir / "attrib",
            self.pref_pairs if hasattr(self, "_pref_pairs") and self._pref_pairs else None,
        )

    def make_coarse_viz(self, _) -> Callable[[], None]:
        """Return visualization function for coarse patching results."""
        # Get position mapping from pair_0 for semantic labels in aggregated viz
        position_mapping = self.get_position_mapping(0, sample="short")
        return lambda: visualize_all_aggregated(
            self.coarse_agg_by_component,
            self.agg_dir / "coarse",
            self.pref_pairs if hasattr(self, "_pref_pairs") and self._pref_pairs else None,
            self.output_dir,
            self.processed_results,
            position_mapping,
        )

    def make_diffmeans_viz(self, config: "DiffMeansConfig") -> Callable[[], None]:
        """Return visualization function for diffmeans results."""
        return lambda: visualize_diffmeans(
            self.diffmeans_agg, self.agg_dir / "diffmeans", config=config
        )

    def make_mlp_viz(self, _) -> Callable[[], None]:
        """Return visualization function for MLP results."""
        return lambda: visualize_all_mlp_slices(
            self.mlp_agg,
            self.agg_dir / "mlp",
            self.pref_pairs if hasattr(self, "_pref_pairs") and self._pref_pairs else None,
        )

    def make_attn_viz(self, _) -> Callable[[], None]:
        """Return visualization function for attention results."""
        return lambda: visualize_all_attn_slices(
            self.attn_agg,
            self.agg_dir / "attn",
            self.pref_pairs if hasattr(self, "_pref_pairs") and self._pref_pairs else None,
            pairs_dir=self.output_dir / "pairs",
        )

    def make_fine_viz(self, _) -> Callable[[], None]:
        """Return visualization function for fine-grained results."""
        def _viz():
            for pair_idx, result in self.fine.items():
                mapping = (
                    self.position_mappings.get(pair_idx, (None, None))[1]
                    if hasattr(self, "_position_mappings") and self._position_mappings
                    else None
                )
                visualize_fine(
                    result, self.get_pair_dir(pair_idx) / "fine", mapping
                )
        return _viz

    # ─── Per-pair visualization ───

    def viz_attrib_pair(self, pair_idx: int) -> None:
        """Generate visualizations for a single attribution pair."""
        if pair_idx not in self.attrib_patching:
            return
        pair_result = self.attrib_patching[pair_idx]
        out_dir = self.get_attrib_pair_dir(pair_idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Use "short" sample mapping (the longer sequence with all semantic positions)
        mapping = self.get_position_mapping(pair_idx, sample="short")
        position_labels = (
            [p.format_pos or f"P{p.index}" for p in mapping.positions]
            if mapping
            else None
        )
        # Visualize both denoising and noising results
        if pair_result.result.denoising:
            visualize_att_patching(
                pair_result.result.denoising,
                out_dir,
                position_labels=position_labels,
                pair_idx=pair_idx,
            )
        if pair_result.result.noising:
            visualize_att_patching(
                pair_result.result.noising,
                out_dir,
                position_labels=position_labels,
                pair_idx=pair_idx,
            )

    def viz_coarse_pair(self, pair_idx: int, component: str) -> None:
        """Generate visualizations for a single coarse patching pair."""
        key = (pair_idx, component)
        if key not in self.coarse_patching:
            return
        result = self.coarse_patching[key]
        out_dir = self.get_pair_dir(pair_idx) / "coarse" / f"sweep_{component}"
        out_dir.mkdir(parents=True, exist_ok=True)
        visualize_coarse_patching(result, out_dir)

    def viz_coarse_pair_component_comparison(
        self, pair_idx: int, components: list[str] | None = None
    ) -> None:
        """Generate component_comparison visualizations for a pair.

        Loads ALL cached components for this pair (not just the specified ones),
        ensuring component comparison plots show all available data.

        Args:
            pair_idx: Pair index
            components: Ignored - always uses all cached components for this pair
        """
        # Get ALL cached components for this pair (ignore the components arg)
        all_cached = self.get_all_cached_components_for_pair(pair_idx)

        # Load all cached components
        results_by_component = {}
        for component in all_cached:
            key = (pair_idx, component)
            if key not in self.coarse_patching:
                self.load_coarse_pair(pair_idx, component)
            if key in self.coarse_patching:
                results_by_component[component] = self.coarse_patching[key]

        if len(results_by_component) < 2:
            return

        out_dir = self.get_pair_dir(pair_idx) / "coarse" / "component_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use "short" sample mapping (the longer sequence with time horizon constraint)
        # since coarse patching positions can come from either sample
        mapping = self.get_position_mapping(pair_idx, sample="short")
        visualize_component_comparison(results_by_component, out_dir, 1, mapping)

    def viz_diffmeans_pair(self, pair_idx: int) -> None:
        """Generate visualizations for a single diffmeans pair."""
        if pair_idx not in self.diffmeans_patching:
            return
        result = self.diffmeans_patching[pair_idx]
        out_dir = self.get_diffmeans_pair_dir(pair_idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        visualize_diffmeans_pair(result, out_dir)

    def viz_mlp_pair(self, pair_idx: int, mapping=None) -> None:
        """Generate visualizations for a single MLP pair."""
        if pair_idx not in self.mlp:
            return
        result = self.mlp[pair_idx]
        out_dir = self.get_mlp_pair_dir(pair_idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        visualize_mlp_pair(result, out_dir, position_mapping=mapping)

    def viz_attn_pair(self, pair_idx: int, mapping=None) -> None:
        """Generate visualizations for a single attention pair."""
        if pair_idx not in self.attn:
            return
        result = self.attn[pair_idx]
        out_dir = self.get_attn_pair_dir(pair_idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        visualize_attn_pair(result, out_dir, pair_idx=pair_idx)
        # Drop heavy attention patterns now that per-pair viz is done.
        result.pop_heavy()

    def viz_fine_pair(self, pair_idx: int, mapping=None) -> None:
        """Generate visualizations for a single fine-grained pair.

        NOTE: mapping parameter is kept for API compatibility but no longer used.
        Layer-position visualizations are now in attn_viz and mlp_viz.
        """
        if pair_idx not in self.fine:
            return
        result = self.fine[pair_idx]
        out_dir = self.get_fine_pair_dir(pair_idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        visualize_fine(result, out_dir)

    # ─── Unload all ───

    def unload_all(self) -> None:
        """Clear all aggregated results from memory."""
        self.unload_attrib_agg()
        self.unload_coarse_agg()
        self.unload_diffmeans_agg()
        self.unload_mlp_agg()
        self.unload_attn_agg()
        self.unload_fine_agg()
