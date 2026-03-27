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
from ...activation_patching.fine import FinePatchingResults
from ...attribution_patching import AttrPatchPairResult, AttrPatchAggregatedResults

from .diffmeans import DiffMeansPairResult, DiffMeansAggregatedResults
from .geo import GeoPairResult, GeoAggregatedResults
from .mlp_analysis import MLPPairResult, MLPAggregatedResults
from .attn_analysis import AttnPairResult, AttnAggregatedResults
from .fine_grained import FineGrainedResults
from .processing import ProcessedResults
from ..common import get_experiment_dir
from ..common.contrastive_utils import get_contrastive_preferences, PrefPairRequirement
from ..common.contrastive_preferences import ContrastivePreferences
from ..preference import PreferenceDataset
from ..prompt import PromptDataset
from ..common.sample_position_mapping import SamplePositionMapping
from ..viz.tokenization_viz import (
    visualize_pair_alignment,
    visualize_position_mapping_pair,
    visualize_tokenization,
    visualize_tokenization_from_position_mapping,
)
from .experiment_config import ExperimentConfig


@dataclass
class ExperimentContext:
    """Shared experiment state."""

    cfg: ExperimentConfig
    pref_data: PreferenceDataset | None = None
    prompt_dataset: PromptDataset | None = None

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
    # Position mappings per pair: {pair_idx: (short_mapping, long_mapping)}
    _position_mappings: dict[int, tuple[SamplePositionMapping, SamplePositionMapping]] = field(
        default_factory=dict, init=False
    )

    # coarse_patching keyed by (pair_idx, component) tuple
    coarse_patching: dict[tuple[int, str], CoarseActPatchResults] = field(
        default_factory=dict
    )
    fine_patching: dict[int, ActPatchPairResult] = field(default_factory=dict)
    attrib_patching: dict[int, AttrPatchPairResult] = field(default_factory=dict)

    coarse_agg_by_component: dict[str, CoarseActPatchAggregatedResults] = field(
        default_factory=dict
    )
    fine_agg: ActPatchAggregatedResult | None = None
    attrib_agg: AttrPatchAggregatedResults | None = None

    # Diffmeans results
    diffmeans_patching: dict[int, DiffMeansPairResult] = field(default_factory=dict)
    diffmeans_agg: DiffMeansAggregatedResults | None = None

    # Geo (PCA) results
    geo_patching: dict[int, GeoPairResult] = field(default_factory=dict)
    geo_agg: GeoAggregatedResults | None = None

    # MLP neuron analysis results
    mlp_analysis: dict[int, MLPPairResult] = field(default_factory=dict)
    mlp_agg: MLPAggregatedResults | None = None

    # Attention pattern analysis results
    attn_analysis: dict[int, AttnPairResult] = field(default_factory=dict)
    attn_agg: AttnAggregatedResults | None = None

    # Fine-grained patching results (comprehensive: plots 17-26)
    fine_grained_patching: dict[int, FineGrainedResults] = field(default_factory=dict)

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

        # Build pairs
        self._pairs = []
        self._pref_pairs = []
        self._pair_to_pref_idx = {}

        for i, pref in enumerate(selected):
            log_progress(i + 1, len(selected), prefix="[ctx] Building pair ")

            # Build SamplePositionMappings from PromptSample (uses structured Prompt)
            short_prompt_sample = self.get_prompt_sample(pref.short_term.sample_idx)
            long_prompt_sample = self.get_prompt_sample(pref.long_term.sample_idx)

            if not short_prompt_sample or not long_prompt_sample:
                raise ValueError(
                    f"prompt_dataset required but missing PromptSample for "
                    f"sample_idx {pref.short_term.sample_idx} or {pref.long_term.sample_idx}"
                )

            short_term_mapping = SamplePositionMapping.build(
                short_prompt_sample, self.runner, pref=pref.short_term
            )
            long_term_mapping = SamplePositionMapping.build(
                long_prompt_sample, self.runner, pref=pref.long_term
            )

            pair = pref.get_contrastive_pair(
                self.runner,
                short_term_mapping=short_term_mapping,
                long_term_mapping=long_term_mapping,
            )
            if pair:
                idx = len(self._pairs)
                self._pairs.append(pair)
                self._pref_pairs.append(pref)
                self._pair_to_pref_idx[idx] = idx
                self._position_mappings[idx] = (short_term_mapping, long_term_mapping)

        log(f"[ctx] Built {len(self._pairs)} valid pairs")

    def get_prompt_sample(self, sample_idx: int):
        """Get PromptSample by sample_idx from prompt_dataset."""
        if self.prompt_dataset is None:
            return None
        for sample in self.prompt_dataset.samples:
            if sample.sample_idx == sample_idx:
                return sample
        return None

    @property
    def position_mappings(self) -> dict[int, tuple[SamplePositionMapping, SamplePositionMapping]]:
        """Position mappings per pair: {pair_idx: (short_mapping, long_mapping)}."""
        if not self._position_mappings and self._pairs is None:
            _ = self.pairs  # Trigger lazy loading
        return self._position_mappings

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
        if self.attrib_agg:
            target = self.attrib_agg.get_target()
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
            / "coarse"
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
        return self.output_dir / "aggregated" / "coarse"

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
            if path.exists():
                self.coarse_agg_by_component[component] = (
                    CoarseActPatchAggregatedResults.from_json(path)
                )
                any_loaded = True
        return any_loaded

    def get_attrib_agg_dir(self) -> Path:
        """Get the directory for aggregated attribution results."""
        return self.output_dir / "aggregated" / "attrib"

    def save_attrib_agg(self) -> None:
        """Save aggregated attribution results to folder structure."""
        if not self.attrib_agg:
            return

        att_dir = self.get_attrib_agg_dir()
        att_dir.mkdir(parents=True, exist_ok=True)
        log(f"[attr] Saving aggregated results to {att_dir}...")

        # Save denoising and noising separately
        if self.attrib_agg.denoising_agg:
            save_json(
                self.attrib_agg.denoising_agg.to_dict(),
                att_dir / "denoising.json",
            )
        if self.attrib_agg.noising_agg:
            save_json(
                self.attrib_agg.noising_agg.to_dict(),
                att_dir / "noising.json",
            )

        # Also save the full aggregated result for compatibility
        save_json(self.attrib_agg.to_dict(), att_dir / "attrib_agg.json")
        log("[attr] Saved.")

    def load_attrib_agg(self) -> bool:
        """Load aggregated attribution results."""
        attrib_dir = self.get_attrib_agg_dir()
        path = attrib_dir / "attrib_agg.json"
        if path.exists():
            self.attrib_agg = AttrPatchAggregatedResults.from_json(path)
            return True
        return False

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

    def get_position_mapping_path(self, pair_idx: int, sample: str = "long") -> Path:
        """Get path for per-pair position mapping JSON.

        Args:
            pair_idx: Index of the pair
            sample: "long" for long-term/corrupted, "short" for short-term/clean
        """
        if sample == "short":
            return self.get_pair_dir(pair_idx) / "position_mapping_short.json"
        return self.get_pair_dir(pair_idx) / "sample_position_mapping.json"

    def save_position_mapping(self, pair_idx: int) -> None:
        """Save position mappings for both samples in a pair.

        Builds SamplePositionMapping for both short-term (clean) and long-term
        (corrupted) trajectories, saves them, and generates position_mapping.png
        showing both side-by-side.
        """
        pref = self.get_pref_pair(pair_idx)
        if pref is None:
            return

        pair_dir = self.get_pair_dir(pair_idx)
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Build mapping for short-term (clean) sample
        short_prompt_sample = self.get_prompt_sample(pref.short_term.sample_idx)
        if not short_prompt_sample:
            raise ValueError(f"Missing PromptSample for sample_idx {pref.short_term.sample_idx}")
        mapping_short = SamplePositionMapping.build(
            short_prompt_sample, self.runner, pref=pref.short_term
        )
        save_json(mapping_short.to_dict(), self.get_position_mapping_path(pair_idx, "short"))

        # Build mapping for long-term (corrupted) sample
        long_prompt_sample = self.get_prompt_sample(pref.long_term.sample_idx)
        if not long_prompt_sample:
            raise ValueError(f"Missing PromptSample for sample_idx {pref.long_term.sample_idx}")
        mapping_long = SamplePositionMapping.build(
            long_prompt_sample, self.runner, pref=pref.long_term
        )
        save_json(mapping_long.to_dict(), self.get_position_mapping_path(pair_idx, "long"))

        # Save the PairPositionMapping directly (the cross-sample alignment)
        if self.pairs and pair_idx < len(self.pairs):
            pair_pos_mapping = self.pairs[pair_idx].position_mapping
            save_json(pair_pos_mapping.to_dict(), pair_dir / "pair_position_mapping.json")

            # Generate alignment visualization showing src<->dst mapping
            visualize_pair_alignment(pair_pos_mapping, pair_dir / "pair_position_mapping.png")

        # Generate combined position mapping visualization showing both samples
        visualize_position_mapping_pair(mapping_short, mapping_long, pair_dir / "position_mapping.png")

        # Generate tokenization divergence visualization
        if self.pairs and pair_idx < len(self.pairs) and self.runner is not None:
            visualize_tokenization([self.pairs[pair_idx]], self.runner, pair_dir, max_pairs=1)

    def save_all_position_mappings(self) -> None:
        """Save position mappings for all pairs."""
        for pair_idx in range(len(self.pairs)):
            self.save_position_mapping(pair_idx)
        log(f"[ctx] Saved position mappings for {len(self.pairs)} pairs")

    def load_position_mapping(
        self, pair_idx: int, sample: str = "long"
    ) -> SamplePositionMapping | None:
        """Load position mapping from cached pair directory.

        Args:
            pair_idx: Index of the pair to load mapping for
            sample: "long" for long-term/corrupted, "short" for short-term/clean

        Returns:
            SamplePositionMapping if found, None otherwise
        """
        path = self.get_position_mapping_path(pair_idx, sample)
        if path.exists():
            data = json.loads(path.read_text())
            return SamplePositionMapping.from_dict(data)
        return None

    def get_position_mapping(
        self, pair_idx: int, sample: str = "long"
    ) -> SamplePositionMapping | None:
        """Get position mapping for a pair, loading from cache or building if needed.

        Args:
            pair_idx: Index of the pair
            sample: "long" for long-term/corrupted, "short" for short-term/clean

        Returns:
            SamplePositionMapping if available, None otherwise
        """
        # Try loading from cache first
        mapping = self.load_position_mapping(pair_idx, sample)
        if mapping is not None:
            return mapping

        # Build from PromptSample + PreferenceSample if available
        pref_pair = self.get_pref_pair(pair_idx)
        if pref_pair is not None and self._runner is not None:
            pref = pref_pair.long_term if sample == "long" else pref_pair.short_term
            prompt_sample = self.get_prompt_sample(pref.sample_idx)
            if prompt_sample:
                return SamplePositionMapping.build(prompt_sample, self._runner, pref=pref)

        return None

    def get_representative_position_mapping(self) -> SamplePositionMapping | None:
        """Get a representative position mapping for visualization labeling.

        This method provides a position mapping that can be used for converting
        absolute positions to semantic format_pos names in visualizations.
        All pairs have the same format_pos names (just different absolute positions),
        so the first pair's mapping is representative.

        Priority:
        1. Build from pref_pairs if runner is available (live run)
        2. Load from cached position mapping (regeneration from cache)

        Returns:
            SamplePositionMapping if available, None otherwise
        """
        # Try building from prompt_dataset + pref_pairs if available (live run)
        if self._pref_pairs and self._runner and self.prompt_dataset:
            pref = self._pref_pairs[0].long_term
            prompt_sample = self.get_prompt_sample(pref.sample_idx)
            if prompt_sample:
                return SamplePositionMapping.build(prompt_sample, self._runner, pref=pref)
        # Otherwise try loading from cache (regeneration)
        return self.load_position_mapping(0)

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

    def get_fine_pair_dir(self, pair_idx: int) -> Path:
        """Get directory for per-pair fine patching results."""
        return self.get_pair_dir(pair_idx) / "fine_patching"

    def get_fine_pair_path(self, pair_idx: int) -> Path:
        """Get path for per-pair fine patching results JSON."""
        return self.get_fine_pair_dir(pair_idx) / "fine_results.json"

    def save_fine_pair(self, pair_idx: int) -> None:
        """Save per-pair fine patching results."""
        if pair_idx not in self.fine_patching:
            return
        result = self.fine_patching[pair_idx]
        path = self.get_fine_pair_path(pair_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(result.to_dict(), path)

    def load_fine_pair(self, pair_idx: int) -> bool:
        """Load per-pair fine patching results."""
        path = self.get_fine_pair_path(pair_idx)
        if path.exists():
            self.fine_patching[pair_idx] = FinePatchingResults.from_json(path)
            return True
        return False

    def detect_cached_fine_pairs(self) -> list[int]:
        """Detect all pair indices that have cached fine patching results."""
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
        return self.output_dir / "aggregated" / "diffmeans"

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
        return self.output_dir / "aggregated" / "geo"

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

    def unload_attrib_agg(self) -> None:
        """Clear attribution aggregated results from memory."""
        self.attrib_agg = None
        self.attrib_patching.clear()

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

    def unload_geo_agg(self) -> None:
        """Clear geo aggregated results from memory."""
        self.geo_agg = None
        self.geo_patching.clear()

    def unload_mlp_agg(self) -> None:
        """Clear MLP aggregated results from memory."""
        self.mlp_agg = None
        self.mlp_analysis.clear()

    def unload_attn_agg(self) -> None:
        """Clear attention aggregated results from memory."""
        self.attn_agg = None
        self.attn_analysis.clear()

    def unload_fine_grained_agg(self) -> None:
        """Clear fine-grained patching results from memory."""
        self.fine_grained_patching.clear()

    def unload_all(self) -> None:
        """Clear all aggregated results from memory."""
        self.unload_attrib_agg()
        self.unload_coarse_agg()
        self.unload_fine_agg()
        self.unload_diffmeans_agg()
        self.unload_geo_agg()
        self.unload_mlp_agg()
        self.unload_attn_agg()
        self.unload_fine_grained_agg()

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
            results_path = pair_dir / "coarse" / f"sweep_{component}" / "coarse_results.json"
            if results_path.exists():
                cached.append(pair_idx)
            pair_idx += 1
        return cached

    def detect_cached_attrib_pairs(self) -> list[int]:
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
            if self.get_attrib_pair_path(pair_idx).exists():
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
            coarse_dir = pair_0 / "coarse"
            if coarse_dir.exists():
                for d in coarse_dir.iterdir():
                    if d.is_dir() and d.name.startswith("sweep_"):
                        comp = d.name.replace("sweep_", "")
                        if (d / "coarse_results.json").exists():
                            components.append(comp)
        return components

    # ─── Processed results save/load methods ───

    def get_analysis_dir(self) -> Path:
        """Get directory for analysis results (processed, horizon, pair)."""
        return self.output_dir / "aggregated" / "analysis"

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

    # ─── MLP Analysis save/load methods ───

    def get_mlp_analysis_pair_dir(self, pair_idx: int) -> Path:
        """Get directory for per-pair MLP analysis results."""
        return self.get_pair_dir(pair_idx) / "mlp_analysis"

    def get_mlp_analysis_pair_path(self, pair_idx: int) -> Path:
        """Get path for per-pair MLP analysis results JSON."""
        return self.get_mlp_analysis_pair_dir(pair_idx) / "mlp_analysis.json"

    def save_mlp_analysis_pair(self, pair_idx: int) -> None:
        """Save per-pair MLP analysis results."""
        if pair_idx not in self.mlp_analysis:
            return
        result = self.mlp_analysis[pair_idx]
        path = self.get_mlp_analysis_pair_path(pair_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(result.to_dict(), path)

    def load_mlp_analysis_pair(self, pair_idx: int) -> bool:
        """Load per-pair MLP analysis results."""
        path = self.get_mlp_analysis_pair_path(pair_idx)
        if path.exists():
            self.mlp_analysis[pair_idx] = MLPPairResult.from_json(path)
            return True
        return False

    def get_mlp_agg_dir(self) -> Path:
        """Get directory for aggregated MLP analysis results."""
        return self.output_dir / "aggregated" / "mlp"

    def save_mlp_agg(self) -> None:
        """Save aggregated MLP analysis results."""
        if not self.mlp_agg:
            return
        agg_dir = self.get_mlp_agg_dir()
        agg_dir.mkdir(parents=True, exist_ok=True)
        path = agg_dir / "mlp_analysis_agg.json"
        log(f"[mlp] Saving aggregated results to {path}...")
        save_json(self.mlp_agg.to_dict(), path)
        log("[mlp] Saved.")

    def load_mlp_agg(self) -> bool:
        """Load aggregated MLP analysis results."""
        path = self.get_mlp_agg_dir() / "mlp_analysis_agg.json"
        if path.exists():
            self.mlp_agg = MLPAggregatedResults.from_json(path)
            return True
        return False

    def detect_cached_mlp_analysis_pairs(self) -> list[int]:
        """Detect all pair indices that have cached MLP analysis results."""
        cached = []
        pair_idx = 0
        while True:
            pair_dir = self.get_pair_dir(pair_idx)
            if not pair_dir.exists():
                break
            if self.get_mlp_analysis_pair_path(pair_idx).exists():
                cached.append(pair_idx)
            pair_idx += 1
        return cached

    # ─── Attention Analysis save/load methods ───

    def get_attn_analysis_pair_dir(self, pair_idx: int) -> Path:
        """Get directory for per-pair attention analysis results."""
        return self.get_pair_dir(pair_idx) / "attn"

    def get_attn_analysis_pair_path(self, pair_idx: int) -> Path:
        """Get path for per-pair attention analysis results JSON."""
        return self.get_attn_analysis_pair_dir(pair_idx) / "attn_results.json"

    def save_attn_analysis_pair(self, pair_idx: int, store_patterns: bool = False) -> None:
        """Save per-pair attention analysis results.

        Args:
            pair_idx: Index of the pair
            store_patterns: If True, keep attention patterns in saved file (large)
        """
        if pair_idx not in self.attn_analysis:
            return
        result = self.attn_analysis[pair_idx]
        path = self.get_attn_analysis_pair_path(pair_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Pop heavy data before saving unless store_patterns=True
        if not store_patterns:
            result.pop_heavy()
        save_json(result.to_dict(), path)

    def load_attn_analysis_pair(self, pair_idx: int) -> bool:
        """Load per-pair attention analysis results."""
        path = self.get_attn_analysis_pair_path(pair_idx)
        if path.exists():
            self.attn_analysis[pair_idx] = AttnPairResult.from_json(path)
            return True
        return False

    def get_attn_agg_dir(self) -> Path:
        """Get directory for aggregated attention analysis results."""
        return self.output_dir / "aggregated" / "attn"

    def save_attn_agg(self) -> None:
        """Save aggregated attention analysis results."""
        if not self.attn_agg:
            return
        agg_dir = self.get_attn_agg_dir()
        agg_dir.mkdir(parents=True, exist_ok=True)
        path = agg_dir / "attn_agg.json"
        log(f"[attn] Saving aggregated results to {path}...")
        save_json(self.attn_agg.to_dict(), path)
        log("[attn] Saved.")

    def load_attn_agg(self) -> bool:
        """Load aggregated attention analysis results."""
        path = self.get_attn_agg_dir() / "attn_agg.json"
        if path.exists():
            self.attn_agg = AttnAggregatedResults.from_json(path)
            return True
        return False

    def detect_cached_attn_analysis_pairs(self) -> list[int]:
        """Detect all pair indices that have cached attention analysis results."""
        cached = []
        pair_idx = 0
        while True:
            pair_dir = self.get_pair_dir(pair_idx)
            if not pair_dir.exists():
                break
            if self.get_attn_analysis_pair_path(pair_idx).exists():
                cached.append(pair_idx)
            pair_idx += 1
        return cached

    # ---- Fine-grained patching save/load methods ----

    def get_fine_grained_pair_dir(self, pair_idx: int) -> Path:
        """Get directory for per-pair fine-grained patching results."""
        return self.get_pair_dir(pair_idx) / "fine_grained"

    def get_fine_grained_pair_path(self, pair_idx: int) -> Path:
        """Get path for per-pair fine-grained patching results JSON."""
        return self.get_fine_grained_pair_dir(pair_idx) / "fine_grained.json"

    def save_fine_grained_pair(self, pair_idx: int) -> None:
        """Save per-pair fine-grained patching results."""
        if pair_idx not in self.fine_grained_patching:
            return
        result = self.fine_grained_patching[pair_idx]
        path = self.get_fine_grained_pair_path(pair_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(result.to_dict(), path)

    def load_fine_grained_pair(self, pair_idx: int) -> bool:
        """Load per-pair fine-grained patching results."""
        path = self.get_fine_grained_pair_path(pair_idx)
        if path.exists():
            self.fine_grained_patching[pair_idx] = FineGrainedResults.from_json(path)
            return True
        return False

    def detect_cached_fine_grained_pairs(self) -> list[int]:
        """Detect all pair indices that have cached fine-grained results."""
        cached = []
        pair_idx = 0
        while True:
            pair_dir = self.get_pair_dir(pair_idx)
            if not pair_dir.exists():
                break
            if self.get_fine_grained_pair_path(pair_idx).exists():
                cached.append(pair_idx)
            pair_idx += 1
        return cached
