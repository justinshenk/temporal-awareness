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
from ...inference import get_recommended_backend_interventions
from ...inference.backends import ModelBackend
from ...binary_choice import BinaryChoiceRunner
from ...activation_patching.coarse import (
    CoarseActPatchResults,
    CoarseActPatchAggregatedResults,
)
from ...attribution_patching import AttrPatchPairResult, AttrPatchAggregatedResults
from .experiment_utils import ExperimentMixin
from .diffmeans import DiffMeansPairResult, DiffMeansAggregatedResults
from .mlp import MLPPairResult, MLPAggregatedResults
from .attn import AttnPairResult, AttnAggregatedResults
from .fine import FineResults, FineAggregatedResults
from .analysis import ProcessedResults
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
)
from .experiment_config import ExperimentConfig


@dataclass
class ExperimentContext(ExperimentMixin):
    """Shared experiment state."""

    cfg: ExperimentConfig
    pref_data: PreferenceDataset | None = None
    prompt_dataset: PromptDataset | None = None

    output_dir: Path | None = field(default=None)
    timestamp: str = field(default_factory=get_timestamp)
    backend: str | None = None

    def __post_init__(self) -> None:
        if self.output_dir is None:
            self.output_dir = get_experiment_dir() / self.cfg.get_id()

    _pairs: list | None = field(default=None, init=False)
    _runner: BinaryChoiceRunner | None = field(default=None, init=False)
    _pref_pairs: list[ContrastivePreferences] | None = field(default=None, init=False)
    _pair_to_pref_idx: dict[int, int] = field(default_factory=dict, init=False)
    _position_mappings: dict[
        int, tuple[SamplePositionMapping, SamplePositionMapping]
    ] = field(default_factory=dict, init=False)
    _use_cached_pairs: bool = field(default=False, init=False)

    # Results storage (ordered to match run_experiment steps)
    attrib_patching: dict[int, AttrPatchPairResult] = field(default_factory=dict)
    attrib_agg: AttrPatchAggregatedResults | None = None
    coarse_patching: dict[tuple[int, str], CoarseActPatchResults] = field(
        default_factory=dict
    )
    coarse_agg_by_component: dict[str, CoarseActPatchAggregatedResults] = field(
        default_factory=dict
    )
    diffmeans_patching: dict[int, DiffMeansPairResult] = field(default_factory=dict)
    diffmeans_agg: DiffMeansAggregatedResults | None = None
    mlp: dict[int, MLPPairResult] = field(default_factory=dict)
    mlp_agg: MLPAggregatedResults | None = None
    attn: dict[int, AttnPairResult] = field(default_factory=dict)
    attn_agg: AttnAggregatedResults | None = None
    fine: dict[int, FineResults] = field(default_factory=dict)
    fine_agg: FineAggregatedResults | None = None
    processed_results: ProcessedResults | None = None

    @property
    def viz_enabled(self) -> bool:
        return self.cfg.viz_cfg.get("enabled", True)

    @property
    def save_svg(self) -> bool:
        """Whether to save SVG files alongside PNGs (camera-ready mode)."""
        return self.cfg.viz_cfg.get("save_svg", False)

    @property
    def save_pdf(self) -> bool:
        """Whether to save PDF files alongside PNGs (camera-ready mode)."""
        return self.cfg.viz_cfg.get("save_pdf", False)

    @property
    def only_viz_agg(self) -> bool:
        """Check if ALL step configs have only_viz_agg=True (skip per-pair viz)."""
        return all(cfg.get("only_viz_agg", False) for cfg in self.cfg.step_cfgs)

    @property
    def runner(self) -> BinaryChoiceRunner:
        """Cached runner for this experiment."""
        if self._runner is None:
            backend = (
                ModelBackend(self.backend)
                if self.backend
                else get_recommended_backend_interventions()
            )
            self._runner = BinaryChoiceRunner(
                self.pref_data.model, device=get_device(), backend=backend
            )
        return self._runner

    @runner.setter
    def runner(self, value: BinaryChoiceRunner | None) -> None:
        """Set or clear the runner."""
        self._runner = value

    @property
    def pairs(self) -> list[ContrastivePair]:
        """Contrastive pairs (cached)."""
        if self._pairs is None:
            self._build_pairs()
        return self._pairs

    def get_cached_pair_count(self) -> int:
        """Count the number of cached pairs in the pairs directory."""
        pairs_dir = self.pairs_dir
        if not pairs_dir.exists():
            return 0
        count = 0
        while (pairs_dir / f"pair_{count}").exists():
            count += 1
        return count

    def enable_cached_pairs(self) -> bool:
        """Enable using cached pairs instead of building from config.

        Returns True if cached pairs exist and were enabled.
        """
        cached_count = self.get_cached_pair_count()
        if cached_count > 0:
            self._use_cached_pairs = True
            log(f"[ctx] Enabled cached pairs mode: {cached_count} pairs available")
            return True
        return False

    def _build_pairs(self) -> None:
        """Build contrastive pairs from preference data."""
        pair_req = (
            PrefPairRequirement.from_dict(self.cfg.pair_req_cfg)
            if self.cfg.pair_req_cfg
            else None
        )
        all_prefs = get_contrastive_preferences(self.pref_data, req=pair_req)

        # When using cached pairs, use the cached count instead of config
        if self._use_cached_pairs:
            cached_count = self.get_cached_pair_count()
            n_select = (
                cached_count
                if cached_count > 0
                else (self.cfg.n_pairs or len(all_prefs))
            )
            log(f"[ctx] Using cached pair count: {n_select}")
        else:
            n_select = self.cfg.n_pairs or len(all_prefs)

        selected = all_prefs[:n_select]
        log(f"[ctx] Found {len(all_prefs)} contrastive prefs, using {len(selected)}")

        self._pairs, self._pref_pairs, self._pair_to_pref_idx = [], [], {}

        for i, pref in enumerate(selected):
            log_progress(i + 1, len(selected), prefix="[ctx] Building pair ")
            short_prompt_sample = self.get_prompt_sample(pref.short_term.sample_idx)
            long_prompt_sample = self.get_prompt_sample(pref.long_term.sample_idx)

            if not short_prompt_sample or not long_prompt_sample:
                raise ValueError(
                    f"prompt_dataset required but missing PromptSample for sample_idx {pref.short_term.sample_idx} or {pref.long_term.sample_idx}"
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
    def position_mappings(
        self,
    ) -> dict[int, tuple[SamplePositionMapping, SamplePositionMapping]]:
        """Position mappings per pair: {pair_idx: (short_mapping, long_mapping)}."""
        if not self._position_mappings and self._pairs is None:
            _ = self.pairs
        return self._position_mappings

    @property
    def pref_pairs(self) -> list[ContrastivePreferences]:
        """ContrastivePreferences objects corresponding to pairs (cached)."""
        if self._pref_pairs is None:
            _ = self.pairs
        return self._pref_pairs or []

    def get_pref_pair(self, pair_idx: int) -> ContrastivePreferences | None:
        """Get the ContrastivePreferences object for a given pair index."""
        if self._pref_pairs is None:
            _ = self.pairs
        pref_idx = self._pair_to_pref_idx.get(pair_idx)
        return (
            self._pref_pairs[pref_idx]
            if pref_idx is not None and self._pref_pairs
            else None
        )

    @property
    def viz_dir(self) -> Path:
        d = self.output_dir / "viz"
        ensure_dir(d)
        return d

    @property
    def pairs_dir(self) -> Path:
        return self.output_dir / "pairs"

    def get_pair_dir(self, pair_idx: int) -> Path:
        return self.pairs_dir / f"pair_{pair_idx}"

    def get_runner(self) -> BinaryChoiceRunner:
        return self.runner

    @property
    def best_contrastive_pair(self):
        return self.pairs[0] if self.pairs else None

    def save_token_trees(
        self, pair_idx: int, pair: ContrastivePair, output_dir: Path
    ) -> None:
        """Save analyzed TokenTree for a contrastive pair."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tree = TokenTree.from_trajectories(
            [pair.clean_traj, pair.corrupted_traj],
            groups_per_traj=[[0], [1]],
            fork_arms=[(0, 1)],
        )
        tree.pop_heavy()
        with open(output_dir / "token_tree.json", "w") as f:
            json.dump(tree.to_dict(), f, indent=2)

    def get_contrastive_pref_path(self, pair_idx: int) -> Path:
        return self.get_pair_dir(pair_idx) / "contrastive_preference.json"

    def save_contrastive_pref(self, pair_idx: int) -> None:
        pref = self.get_pref_pair(pair_idx)
        if pref is None:
            return
        path = self.get_contrastive_pref_path(pair_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(pref.to_summary_dict(), path)

    def save_all_contrastive_prefs(self) -> None:
        for pair_idx in range(len(self.pairs)):
            self.save_contrastive_pref(pair_idx)
        log(f"[ctx] Saved contrastive preferences for {len(self.pairs)} pairs")

    def is_analysis_cached(self, pair_idx: int) -> bool:
        """Check if analysis artifacts exist for a pair."""
        return (
            self.get_contrastive_pref_path(pair_idx).exists()
            and self.get_position_mapping_path(pair_idx, "long").exists()
        )

    def get_position_mapping_path(self, pair_idx: int, sample: str = "long") -> Path:
        if sample == "short":
            return self.get_pair_dir(pair_idx) / "short_position_mapping.json"
        return self.get_pair_dir(pair_idx) / "long_position_mapping.json"

    def save_position_mapping_data(self, pair_idx: int) -> None:
        """Save position mapping JSON files for a pair (no visualization)."""
        pref = self.get_pref_pair(pair_idx)
        if pref is None:
            return

        pair_dir = self.get_pair_dir(pair_idx)
        pair_dir.mkdir(parents=True, exist_ok=True)

        short_prompt_sample = self.get_prompt_sample(pref.short_term.sample_idx)
        if not short_prompt_sample:
            raise ValueError(
                f"Missing PromptSample for sample_idx {pref.short_term.sample_idx}"
            )
        mapping_short = SamplePositionMapping.build(
            short_prompt_sample, self.runner, pref=pref.short_term
        )
        save_json(
            mapping_short.to_dict(), self.get_position_mapping_path(pair_idx, "short")
        )

        long_prompt_sample = self.get_prompt_sample(pref.long_term.sample_idx)
        if not long_prompt_sample:
            raise ValueError(
                f"Missing PromptSample for sample_idx {pref.long_term.sample_idx}"
            )
        mapping_long = SamplePositionMapping.build(
            long_prompt_sample, self.runner, pref=pref.long_term
        )
        save_json(
            mapping_long.to_dict(), self.get_position_mapping_path(pair_idx, "long")
        )

        if self.pairs and pair_idx < len(self.pairs):
            pair_pos_mapping = self.pairs[pair_idx].position_mapping
            save_json(
                pair_pos_mapping.to_dict(), pair_dir / "pair_position_mapping.json"
            )

    def save_position_mapping_viz(self, pair_idx: int) -> None:
        """Generate per-pair position mapping visualizations."""
        pref = self.get_pref_pair(pair_idx)
        if pref is None:
            return

        pair_dir = self.get_pair_dir(pair_idx)

        # Load mappings from saved data
        mapping_short = self.load_position_mapping(pair_idx, "short")
        mapping_long = self.load_position_mapping(pair_idx, "long")

        if self.pairs and pair_idx < len(self.pairs):
            pair_pos_mapping = self.pairs[pair_idx].position_mapping
            visualize_pair_alignment(
                pair_pos_mapping,
                pair_dir / "pair_position_mapping.png",
                pair_idx=pair_idx,
            )

        if mapping_short and mapping_long:
            visualize_position_mapping_pair(
                mapping_short,
                mapping_long,
                pair_dir / "position_mapping.png",
                pair_idx=pair_idx,
            )

        if self.pairs and pair_idx < len(self.pairs) and self.runner is not None:
            visualize_tokenization(
                [self.pairs[pair_idx]],
                self.runner,
                pair_dir,
                max_pairs=1,
                pair_idx=pair_idx,
            )

    def save_all_position_mapping_data(self) -> None:
        """Save position mapping JSON for all pairs."""
        for pair_idx in range(len(self.pairs)):
            self.save_position_mapping_data(pair_idx)
        log(f"[ctx] Saved position mapping data for {len(self.pairs)} pairs")

    def save_all_position_mapping_viz(self) -> None:
        """Generate per-pair position mapping visualizations for all pairs."""
        for pair_idx in range(len(self.pairs)):
            self.save_position_mapping_viz(pair_idx)
        log(f"[ctx] Saved position mapping viz for {len(self.pairs)} pairs")

    def save_all_position_mappings(self, skip_viz: bool = False) -> None:
        """Save position mappings (data + optional viz) for all pairs."""
        self.save_all_position_mapping_data()
        if not skip_viz:
            self.save_all_position_mapping_viz()

    def load_position_mapping(
        self, pair_idx: int, sample: str = "long"
    ) -> SamplePositionMapping | None:
        path = self.get_position_mapping_path(pair_idx, sample)
        if path.exists():
            return SamplePositionMapping.from_dict(json.loads(path.read_text()))
        return None

    def get_position_mapping(
        self, pair_idx: int, sample: str = "long"
    ) -> SamplePositionMapping | None:
        mapping = self.load_position_mapping(pair_idx, sample)
        if mapping is not None:
            return mapping
        pref_pair = self.get_pref_pair(pair_idx)
        if pref_pair is not None and self._runner is not None:
            pref = pref_pair.long_term if sample == "long" else pref_pair.short_term
            prompt_sample = self.get_prompt_sample(pref.sample_idx)
            if prompt_sample:
                return SamplePositionMapping.build(
                    prompt_sample, self._runner, pref=pref
                )
        return None

    def get_representative_position_mapping(self) -> SamplePositionMapping | None:
        if self._pref_pairs and self._runner and self.prompt_dataset:
            pref = self._pref_pairs[0].long_term
            prompt_sample = self.get_prompt_sample(pref.sample_idx)
            if prompt_sample:
                return SamplePositionMapping.build(
                    prompt_sample, self._runner, pref=pref
                )
        return self.load_position_mapping(0)
