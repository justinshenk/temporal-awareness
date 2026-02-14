"""Structure and System abstractions for tree analysis.

Key concepts from Structure-Aware Diversity (diversity.pdf):
- Structure αi(x): Compliance of string x to structure i, ∈ [0,1]
- System Λn(x): Vector of compliances across n structures
- Core ⟨Λn⟩(xp): Expected system compliance conditioned on xp
- Orientation θn(x|xp): Λn(x) - ⟨Λn⟩(xp)
- Deviance ∂n(x|xp): ||θn(x|xp)||

Structures/Systems are global. Normativity is per-node (different cores).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from ..base_schema import BaseSchema
from ..math import (
    core_diversity,
    core_entropy,
    deficit_deviance,
    deviance,
    deviance_variance,
    excess_deviance,
    expected_deviance,
    orientation,
)

if TYPE_CHECKING:
    from ..token_tree import BinaryFork, TokenTrajectory, TokenTree


def _traj_probability(traj: TokenTrajectory) -> float:
    """p(y) = exp(Σ logprobs)."""
    return math.exp(sum(traj.logprobs))


def _normalize_probs(probs: Sequence[float]) -> list[float]:
    """Normalize to sum to 1."""
    total = sum(probs)
    if total <= 0:
        n = len(probs)
        return [1.0 / n] * n if n > 0 else []
    return [p / total for p in probs]


def _excess_compliance_inf(compliance: Sequence[float], core: Sequence[float]) -> float:
    """∂⁺_∞(y) = max_i (Λ_norm_i / core_norm_i) - largest excess of compliance."""
    c_norm = _normalize_probs(compliance)
    core_norm = _normalize_probs(core)
    max_ratio = 0.0
    for c, k in zip(c_norm, core_norm):
        if c > 1e-10:
            if k < 1e-10:
                return float("inf")
            max_ratio = max(max_ratio, c / k)
    return max_ratio if max_ratio > 0 else 1.0


def _deficit_compliance_inf(compliance: Sequence[float], core: Sequence[float]) -> float:
    """∂⁻_∞(y) = max_i (core_norm_i / Λ_norm_i) - largest deficit of compliance."""
    c_norm = _normalize_probs(compliance)
    core_norm = _normalize_probs(core)
    max_ratio = 0.0
    for c, k in zip(c_norm, core_norm):
        if k > 1e-10:
            if c < 1e-10:
                return float("inf")
            max_ratio = max(max_ratio, k / c)
    return max_ratio if max_ratio > 0 else 1.0


@dataclass
class Structure(BaseSchema):
    """Group with binary compliance per trajectory."""

    idx: int
    compliances: tuple[float, ...]

    @classmethod
    def from_trajectories(cls, idx: int, trajs: Sequence[TokenTrajectory]) -> Structure:
        return cls(
            idx=idx,
            compliances=tuple(
                1.0 if (t.group_idx and idx in t.group_idx) else 0.0 for t in trajs
            ),
        )


@dataclass
class System(BaseSchema):
    """A system comparing n structures."""

    idx: int
    structure_idx: tuple[int, ...]  # Which structures this system compares

    @classmethod
    def from_fork(cls, idx: int, fork: BinaryFork) -> System:
        if fork.group_idx is None:
            raise ValueError(f"Fork {idx} has no group_idx")
        return cls(idx=idx, structure_idx=fork.group_idx)

    @classmethod
    def from_groups(cls, idx: int, groups: Sequence[int]) -> System:
        return cls(idx=idx, structure_idx=tuple(groups))


@dataclass
class SystemCore(BaseSchema):
    """⟨Λn⟩(xp) = Σ p(y|xp) Λn(y) at a branch node."""

    sys_idx: int
    struct_idx: tuple[int, ...]
    struct_cores: tuple[float, ...]  # Core value per structure
    entropy: float
    diversity: float


@dataclass
class SystemOrientation(BaseSchema):
    """θn(y|xp) and ∂n(y|xp) at a branch node."""

    sys_idx: int
    traj_orientations: tuple[
        tuple[float, ...] | None, ...
    ]  # Per-traj orientation vector
    traj_deviances: tuple[float | None, ...]
    expected_deviance: float
    var_deviance: float
    max_deviance: float
    min_deviance: float

    # Per-trajectory effective over/under-compliance (4 values per trajectory)
    # q=1 (KL divergence based)
    traj_excess_deviance: tuple[float | None, ...]  # ∂⁺_1(y) = e^{D_1(Λ||core)}
    traj_deficit_deviance: tuple[float | None, ...]  # ∂⁻_1(y) = e^{D_1(core||Λ)}
    # q=∞ (max ratio)
    traj_excess_compliance: tuple[float | None, ...]  # ∂⁺_∞(y) = max_i(Λ_i/core_i)
    traj_deficit_compliance: tuple[float | None, ...]  # ∂⁻_∞(y) = max_i(core_i/Λ_i)


@dataclass
class Normativity(BaseSchema):
    """Cores and orientations for all systems at a branch node."""

    node_idx: int
    token_position: int | None  # Position in sequence where this branching occurs
    traj_indices: tuple[int, ...]
    traj_probs: tuple[float, ...]
    cores: tuple[SystemCore, ...]
    orientations: tuple[SystemOrientation, ...]

    @property
    def n_systems(self) -> int:
        return len(self.cores)


def _traj_prob_from_position(traj: TokenTrajectory, position: int | None) -> float:
    """p(y|position) = exp(Σ logprobs[position:])."""
    if position is None:
        return math.exp(sum(traj.logprobs))
    return math.exp(sum(traj.logprobs[position:]))


def calculate_normativity(
    node_idx: int,
    token_position: int | None,
    systems: Sequence[System],
    structures: dict[int, Structure],
    all_trajs: Sequence[TokenTrajectory],
    branch_traj_indices: Sequence[int],
) -> Normativity:
    """Calculate normativity for a branch node.

    Args:
        token_position: If None (root), use full trajectory probs.
                       Otherwise, use conditional probs from this position.
    """
    n_trajs = len(all_trajs)
    branch_indices = list(branch_traj_indices)
    branch_set = set(branch_indices)

    # Compute conditional probabilities from this node's position
    traj_probs = [_traj_prob_from_position(t, token_position) for t in all_trajs]
    branch_probs = _normalize_probs([traj_probs[i] for i in branch_indices])
    branch_prob_map = dict(zip(branch_indices, branch_probs))

    cores: list[SystemCore] = []
    orientations_list: list[SystemOrientation] = []

    for system in systems:
        # Get structures for this system (can be any number)
        sys_structures = [structures[g] for g in system.structure_idx]

        # Core: ⟨α_i⟩ = Σ p(y|branch) α_i(y) for each structure
        core = tuple(
            sum(branch_prob_map[i] * s.compliances[i] for i in branch_indices)
            for s in sys_structures
        )

        cores.append(
            SystemCore(
                sys_idx=system.idx,
                struct_idx=system.structure_idx,
                struct_cores=core,
                entropy=float(core_entropy(core)),
                diversity=float(core_diversity(core)),
            )
        )

        # Per-trajectory orientation/deviance
        traj_orientations: list[tuple[float, ...] | None] = []
        traj_deviances_list: list[float | None] = []
        traj_excess_dev_list: list[float | None] = []
        traj_deficit_dev_list: list[float | None] = []
        traj_excess_comp_list: list[float | None] = []
        traj_deficit_comp_list: list[float | None] = []
        branch_compliances: list[tuple[float, ...]] = []

        for traj_idx in range(n_trajs):
            if traj_idx not in branch_set:
                traj_orientations.append(None)
                traj_deviances_list.append(None)
                traj_excess_dev_list.append(None)
                traj_deficit_dev_list.append(None)
                traj_excess_comp_list.append(None)
                traj_deficit_comp_list.append(None)
            else:
                compliance = tuple(s.compliances[traj_idx] for s in sys_structures)
                theta = orientation(compliance, core)
                dev = deviance(compliance, core, norm="l2")
                # q=1 (KL divergence)
                exc_dev = excess_deviance(compliance, core, alpha=1.0)
                def_dev = deficit_deviance(compliance, core, alpha=1.0)
                # q=∞ (max ratio)
                exc_comp = _excess_compliance_inf(compliance, core)
                def_comp = _deficit_compliance_inf(compliance, core)

                traj_orientations.append(tuple(float(t) for t in theta))
                traj_deviances_list.append(float(dev))
                traj_excess_dev_list.append(float(exc_dev))
                traj_deficit_dev_list.append(float(def_dev))
                traj_excess_comp_list.append(float(exc_comp))
                traj_deficit_comp_list.append(float(def_comp))
                branch_compliances.append(compliance)

        active = [d for d in traj_deviances_list if d is not None]

        orientations_list.append(
            SystemOrientation(
                sys_idx=system.idx,
                traj_orientations=tuple(traj_orientations),
                traj_deviances=tuple(traj_deviances_list),
                expected_deviance=expected_deviance(
                    branch_compliances, core, branch_probs, norm="l2"
                ),
                var_deviance=deviance_variance(
                    branch_compliances, core, branch_probs, norm="l2"
                ),
                max_deviance=max(active) if active else 0.0,
                min_deviance=min(active) if active else 0.0,
                traj_excess_deviance=tuple(traj_excess_dev_list),
                traj_deficit_deviance=tuple(traj_deficit_dev_list),
                traj_excess_compliance=tuple(traj_excess_comp_list),
                traj_deficit_compliance=tuple(traj_deficit_comp_list),
            )
        )

    return Normativity(
        node_idx=node_idx,
        token_position=token_position,
        traj_indices=tuple(branch_indices),
        traj_probs=tuple(branch_probs),
        cores=tuple(cores),
        orientations=tuple(orientations_list),
    )


@dataclass
class StructureSystemAnalysis(BaseSchema):
    """Complete analysis: structures, systems, and per-node normativity."""

    structures: tuple[Structure, ...]
    systems: tuple[System, ...]
    normativities: dict[int, Normativity]

    @property
    def n_structures(self) -> int:
        return len(self.structures)

    @property
    def n_systems(self) -> int:
        return len(self.systems)

    def get_structure(self, group_idx: int) -> Structure | None:
        return next((s for s in self.structures if s.idx == group_idx), None)

    def get_system(self, fork_idx: int) -> System | None:
        return next((s for s in self.systems if s.idx == fork_idx), None)

    def get_normativity(self, node_idx: int) -> Normativity | None:
        return self.normativities.get(node_idx)

    def _to_dict_hook(self, d: dict) -> dict:
        """Present all information with clear, descriptive keys."""

        def r(x: float) -> float | str:
            """Round to 3 decimal places, handle inf/nan."""
            if math.isnan(x):
                return "NaN"
            if math.isinf(x):
                return "Inf" if x > 0 else "-Inf"
            return round(x, 3)

        def norm_to_dict(norm: Normativity) -> dict:
            """Convert normativity to dict, handling multiple systems."""
            if not norm.cores:
                return {}

            def orient_to_dict(orient: SystemOrientation) -> dict:
                """Convert orientation to dict with per-trajectory metrics."""
                # Compute deviance statistics
                active = [d for d in orient.traj_deviances if d is not None]
                std_dev = math.sqrt(orient.var_deviance) if orient.var_deviance > 0 else 0.0
                median_dev = 0.0
                if active:
                    sorted_dev = sorted(active)
                    n = len(sorted_dev)
                    if n % 2 == 1:
                        median_dev = sorted_dev[n // 2]
                    else:
                        median_dev = (sorted_dev[n // 2 - 1] + sorted_dev[n // 2]) / 2

                return {
                    "traj_deviances": [
                        r(dv) if dv is not None else None for dv in orient.traj_deviances
                    ],
                    "expected_deviance": r(orient.expected_deviance),
                    "min_deviance": r(orient.min_deviance),
                    "max_deviance": r(orient.max_deviance),
                    "std_deviance": r(std_dev),
                    "median_deviance": r(median_dev),
                    "generalizations": {
                        # q=1 (KL divergence based)
                        "traj_excess_deviance": [
                            r(v) if v is not None else None
                            for v in orient.traj_excess_deviance
                        ],
                        "traj_deficit_deviance": [
                            r(v) if v is not None else None
                            for v in orient.traj_deficit_deviance
                        ],
                        # q=∞ (max ratio)
                        "traj_excess_compliance": [
                            r(v) if v is not None else None
                            for v in orient.traj_excess_compliance
                        ],
                        "traj_deficit_compliance": [
                            r(v) if v is not None else None
                            for v in orient.traj_deficit_compliance
                        ],
                    },
                }

            # If single system, show flat structure
            if len(norm.cores) == 1:
                core = norm.cores[0]
                orient = norm.orientations[0]
                result = {
                    "groups": list(core.struct_idx),
                    "core": [r(c) for c in core.struct_cores],
                    "core_entropy": r(core.entropy),
                    "core_diversity": r(core.diversity),
                }
                result.update(orient_to_dict(orient))
                return result

            # Multiple systems: key by groups being compared
            systems = {}
            for core, orient in zip(norm.cores, norm.orientations):
                key = str(list(core.struct_idx))  # e.g., "[0, 1]"
                sys_dict = {
                    "core": [r(c) for c in core.struct_cores],
                    "core_entropy": r(core.entropy),
                    "core_diversity": r(core.diversity),
                }
                sys_dict.update(orient_to_dict(orient))
                systems[key] = sys_dict
            return {"systems": systems}

        # Root normativity (full trajectory probs, before branching)
        root = self.normativities.get(0)
        if not root or not root.cores:
            return {"n_groups": self.n_structures, "n_forks": self.n_systems}

        result: dict = {
            "root": norm_to_dict(root),
        }

        # All branch normativities keyed by token position
        branches = {}
        for node_idx, norm in self.normativities.items():
            if node_idx == 0:  # Skip root
                continue
            if norm.token_position is not None and norm.cores:
                branches[norm.token_position] = norm_to_dict(norm)

        if branches:
            result["branches"] = branches

        return result

    @classmethod
    def from_dict(cls, d: dict):
        """Reconstruct from the simplified dict format."""
        root_data = d.get("root", {})

        # Handle single vs multiple systems format
        if "systems" in root_data:
            # Multiple systems: get first for basic info
            first_sys = next(iter(root_data["systems"].values()), {})
            core_values = tuple(first_sys.get("core", []))
            traj_deviances = tuple(first_sys.get("traj_deviances", []))
        else:
            # Single system (flat format)
            core_values = tuple(root_data.get("core", []))
            traj_deviances = tuple(root_data.get("traj_deviances", []))

        n_groups = len(core_values)
        n_trajs = len(traj_deviances)

        # Reconstruct structures (binary compliance)
        structures = tuple(
            Structure(
                idx=g,
                compliances=tuple(1.0 if i == g else 0.0 for i in range(n_trajs)),
            )
            for g in range(n_groups)
        )

        # Reconstruct systems (comparing all groups)
        struct_idx = tuple(root_data.get("groups", list(range(n_groups))))
        systems = (System(idx=0, structure_idx=struct_idx),) if n_groups >= 2 else ()

        def make_normativity(
            node_idx: int, token_position: int | None, data: dict
        ) -> Normativity:
            # Handle single vs multiple systems format
            if "systems" in data:
                first_sys = next(iter(data["systems"].values()), {})
                core_vals = tuple(first_sys.get("core", []))
                deviances = tuple(first_sys.get("traj_deviances", []))
                gen = first_sys.get("generalizations", {})
                excess_dev = tuple(gen.get("traj_excess_deviance", []))
                deficit_dev = tuple(gen.get("traj_deficit_deviance", []))
                excess_comp = tuple(gen.get("traj_excess_compliance", []))
                deficit_comp = tuple(gen.get("traj_deficit_compliance", []))
                entropy = first_sys.get("core_entropy", 0.0)
                diversity = first_sys.get("core_diversity", 1.0)
                exp_dev = first_sys.get("expected_deviance", 0.0)
            else:
                core_vals = tuple(data.get("core", []))
                deviances = tuple(data.get("traj_deviances", []))
                gen = data.get("generalizations", {})
                excess_dev = tuple(gen.get("traj_excess_deviance", []))
                deficit_dev = tuple(gen.get("traj_deficit_deviance", []))
                excess_comp = tuple(gen.get("traj_excess_compliance", []))
                deficit_comp = tuple(gen.get("traj_deficit_compliance", []))
                entropy = data.get("core_entropy", 0.0)
                diversity = data.get("core_diversity", 1.0)
                exp_dev = data.get("expected_deviance", 0.0)

            core = SystemCore(
                sys_idx=0,
                struct_idx=struct_idx,
                struct_cores=core_vals,
                entropy=entropy,
                diversity=diversity,
            )
            orient = SystemOrientation(
                sys_idx=0,
                traj_orientations=(),
                traj_deviances=deviances,
                expected_deviance=exp_dev,
                var_deviance=0.0,
                max_deviance=max((x for x in deviances if x is not None), default=0.0),
                min_deviance=min((x for x in deviances if x is not None), default=0.0),
                traj_excess_deviance=excess_dev,
                traj_deficit_deviance=deficit_dev,
                traj_excess_compliance=excess_comp,
                traj_deficit_compliance=deficit_comp,
            )
            return Normativity(
                node_idx=node_idx,
                token_position=token_position,
                traj_indices=tuple(range(n_trajs)),
                traj_probs=core_vals,
                cores=(core,),
                orientations=(orient,),
            )

        # Root normativity (token_position=None means full trajectory probs)
        normativities: dict[int, Normativity] = {
            0: make_normativity(0, None, root_data)
        }

        # Branch normativities keyed by token position
        branches = d.get("branches", {})
        for i, (token_pos, branch_data) in enumerate(branches.items(), start=1):
            pos = int(token_pos) if isinstance(token_pos, str) else token_pos
            normativities[i] = make_normativity(i, pos, branch_data)

        return cls(
            structures=structures,
            systems=systems,
            normativities=normativities,
        )


def build_tree_as_structures_system(tree: TokenTree) -> StructureSystemAnalysis:
    """Build structure/system analysis for a token tree."""
    if not tree.forks or not tree.groups:
        return StructureSystemAnalysis(structures=(), systems=(), normativities={})

    structures = {g: Structure.from_trajectories(g, tree.trajs) for g in tree.groups}
    systems = [System.from_fork(i, f) for i, f in enumerate(tree.forks) if f.group_idx]

    normativities: dict[int, Normativity] = {}

    # Root (all trajectories, full trajectory probs)
    normativities[0] = calculate_normativity(
        0,
        None,  # Full trajectory probs
        systems,
        structures,
        tree.trajs,
        list(range(len(tree.trajs))),
    )

    # Branching nodes (conditional probs from each node's position)
    if tree.nodes:
        for i, node in enumerate(tree.nodes):
            if node.traj_idx:
                normativities[i + 1] = calculate_normativity(
                    i + 1,
                    node.branching_token_position,
                    systems,
                    structures,
                    tree.trajs,
                    list(node.traj_idx),
                )

    return StructureSystemAnalysis(
        structures=tuple(structures.values()),
        systems=tuple(systems),
        normativities=normativities,
    )
