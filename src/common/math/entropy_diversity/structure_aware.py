"""Structure-aware diversity metrics.

Implements concepts from "Structure-Aware Diversity Pursuit" for reasoning
about diversity relative to context-specific structures.

Key concepts:
- Structure: A specification of organization among tokens (e.g., "mentions women")
- StructureCompliance α_i(x) ∈ [0,1]: How much string x satisfies structure i
- System: A collection of structures of interest
- SystemCompliance Λ_n(x): Vector of compliances across n structures
- SystemCore ⟨Λ_n⟩: Expected system compliance under a distribution
- Orientation θ_n(x): Deviation from the core, Λ_n(x) - ⟨Λ_n⟩
- Deviance ∂_n(x): Scalar measure of non-normativity, ||θ_n(x)||

The core insight: diversity is always relative to a context (system of structures).
What counts as "diverse" depends on which structures we care about.

Reference: https://www.unrulyabstractions.com/pdfs/diversity.pdf
"""

from __future__ import annotations

import math
from typing import Literal, Sequence, Union

import torch

from ..num_types import Num, Nums, is_tensor
from .core import _EPS, probs_to_logprobs
from .divergence import renyi_divergence
from .entropy import shannon_entropy
from .escort_distribution import escort_probs
from .power_mean import weighted_power_mean


# ── Type aliases ─────────────────────────────────────────────────────────────

StructureCompliance = float  # α_i(x) ∈ [0, 1]
SystemCompliance = Union[Sequence[StructureCompliance], torch.Tensor]
SystemCore = Union[Sequence[StructureCompliance], torch.Tensor]
NormType = Literal["l1", "l2", "linf"]


# ══════════════════════════════════════════════════════════════════════════════
# ORIENTATION: θ_n(x) = Λ_n(x) - ⟨Λ_n⟩
# ══════════════════════════════════════════════════════════════════════════════


def _orientation_native(
    compliance: Sequence[float], core: Sequence[float]
) -> list[float]:
    """Orientation vector (pure Python)."""
    return [c - m for c, m in zip(compliance, core)]


def _orientation_torch(compliance: torch.Tensor, core: torch.Tensor) -> torch.Tensor:
    """Orientation vector (PyTorch)."""
    return compliance - core


def orientation(compliance: SystemCompliance, core: SystemCore) -> Nums:
    """Orientation: deviation of compliance from the system core.

    θ_n(x) = Λ_n(x) - ⟨Λ_n⟩

    The orientation tells us in what ways a string is non-normative.
    Positive = over-compliance, negative = under-compliance.

    Args:
        compliance: System compliance Λ_n(x) for a specific string
        core: System core ⟨Λ_n⟩ (expected compliance)

    Returns:
        Orientation vector θ_n(x) with same length as inputs
    """
    if is_tensor(compliance) and is_tensor(core):
        return _orientation_torch(compliance, core)
    if is_tensor(compliance) or is_tensor(core):
        raise TypeError("compliance and core must both be tensors or both be sequences")
    compliance_seq = list(compliance)
    core_seq = list(core)
    if len(compliance_seq) != len(core_seq):
        raise ValueError("compliance and core must have same length")
    return _orientation_native(compliance_seq, core_seq)


# ══════════════════════════════════════════════════════════════════════════════
# DEVIANCE: ∂_n(x) = ||θ_n(x)||
# ══════════════════════════════════════════════════════════════════════════════


def _deviance_native(theta: Sequence[float], norm: NormType) -> float:
    """Deviance from orientation vector (pure Python)."""
    if not theta:
        return 0.0
    if norm == "l2":
        return math.sqrt(sum(t * t for t in theta))
    if norm == "l1":
        return sum(abs(t) for t in theta)
    if norm == "linf":
        return max(abs(t) for t in theta)
    raise ValueError(f"Unknown norm: {norm}")


def _deviance_torch(theta: torch.Tensor, norm: NormType) -> torch.Tensor:
    """Deviance from orientation vector (PyTorch)."""
    if theta.numel() == 0:
        return torch.tensor(0.0, device=theta.device)
    if norm == "l2":
        return theta.norm(p=2)
    if norm == "l1":
        return theta.abs().sum()
    if norm == "linf":
        return theta.abs().max()
    raise ValueError(f"Unknown norm: {norm}")


def deviance(
    compliance: SystemCompliance,
    core: SystemCore,
    norm: NormType = "l2",
) -> Num:
    """Deviance: scalar measure of non-normativity.

    ∂_n(x) = ||θ_n(x)|| = ||Λ_n(x) - ⟨Λ_n⟩||

    Summarizes how far a string deviates from the normative core.
    Higher deviance = more "queer" / non-normative.

    Args:
        compliance: System compliance Λ_n(x) for a specific string
        core: System core ⟨Λ_n⟩ (expected compliance)
        norm: "l2" (Euclidean), "l1" (Manhattan), or "linf" (max)

    Returns:
        Deviance ∈ [0, √n] for l2, [0, n] for l1, [0, 1] for linf
    """
    theta = orientation(compliance, core)
    if is_tensor(theta):
        return _deviance_torch(theta, norm)
    return _deviance_native(theta, norm)


def normalized_deviance(
    compliance: SystemCompliance,
    core: SystemCore,
    norm: NormType = "l2",
) -> Num:
    """Normalized deviance in [0, 1].

    Scales deviance by its theoretical maximum for n structures.
    """
    if is_tensor(compliance):
        n = compliance.numel()
    else:
        n = len(compliance)

    if n == 0:
        return 0.0

    raw = deviance(compliance, core, norm)

    # Maximum deviance when all components differ by 1
    if norm == "l2":
        max_dev = math.sqrt(n)
    elif norm == "l1":
        max_dev = float(n)
    else:  # linf
        max_dev = 1.0

    if is_tensor(raw):
        return raw / max_dev
    return raw / max_dev if max_dev > _EPS else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# CORE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════


def _normalize_core_native(core: Sequence[float]) -> list[float]:
    """Normalize core to probability distribution (pure Python)."""
    total = sum(core)
    if total < _EPS:
        n = len(core)
        return [1.0 / n] * n if n > 0 else []
    return [c / total for c in core]


def _normalize_core_torch(core: torch.Tensor) -> torch.Tensor:
    """Normalize core to probability distribution (PyTorch)."""
    total = core.sum()
    if total < _EPS:
        n = core.numel()
        return torch.full_like(core, 1.0 / n) if n > 0 else core
    return core / total


def normalize_core(core: SystemCore) -> Nums:
    """Normalize system core to form a probability distribution.

    ⟨α_norm_i⟩ = ⟨α_i⟩ / Σ_j ⟨α_j⟩

    Used for computing core entropy.
    """
    if is_tensor(core):
        return _normalize_core_torch(core)
    return _normalize_core_native(list(core))


def _core_entropy_native(core: Sequence[float]) -> float:
    """Entropy of normalized core (pure Python).

    Re-uses shannon_entropy by converting to logprobs.
    """
    normalized = _normalize_core_native(core)
    if not normalized:
        return 0.0
    logprobs = probs_to_logprobs(normalized)
    return shannon_entropy(logprobs)


def _core_entropy_torch(core: torch.Tensor) -> torch.Tensor:
    """Entropy of normalized core (PyTorch)."""
    normalized = _normalize_core_torch(core)
    if normalized.numel() == 0:
        return torch.tensor(0.0, device=core.device)
    logprobs = torch.log(normalized.clamp(min=_EPS))
    return shannon_entropy(logprobs)


def core_entropy(core: SystemCore) -> Num:
    """Entropy of the normalized system core.

    H(⟨Λ_n⟩) = -Σ ⟨α_norm_i⟩ log(⟨α_norm_i⟩)

    Measures how evenly distributed compliance is across structures.
    Low entropy = fewer structures dominate (homogenization).
    High entropy = balanced compliance (diversity).

    Range: [0, log(n)] where n = number of structures.
    """
    if is_tensor(core):
        return _core_entropy_torch(core)
    return _core_entropy_native(list(core))


def core_diversity(core: SystemCore) -> Num:
    """Effective number of structures (Hill number D_1).

    D_1 = exp(H(⟨Λ_n⟩))

    Interpretation: "How many structures effectively contribute?"
    Range: [1, n] where n = number of structures.
    """
    h = core_entropy(core)
    if is_tensor(h):
        return h.exp()
    return math.exp(h)


# ══════════════════════════════════════════════════════════════════════════════
# GENERALIZED CORES (Escort Power Mean)
# ══════════════════════════════════════════════════════════════════════════════


def _generalized_structure_core_native(
    compliances: Sequence[float],
    probs: Sequence[float],
    q: float,
    r: float,
) -> float:
    """Generalized structure core (pure Python).

    Uses escort_probs for r-weighting and weighted_power_mean for q-averaging.

    ⟨Λ_n⟩_{q,r} = M_q(compliances, escort_r(probs))
    """
    if len(compliances) != len(probs):
        raise ValueError("compliances and probs must have same length")
    if not compliances:
        return 0.0

    # Convert probs to logprobs for escort_probs
    logprobs = probs_to_logprobs(list(probs))

    # Compute r-escort weights using escort_probs
    escort_weights = escort_probs(logprobs, r)

    # Apply weighted power mean with exponent q
    return weighted_power_mean(list(compliances), escort_weights, q)


def _generalized_structure_core_torch(
    compliances: torch.Tensor,
    probs: torch.Tensor,
    q: float,
    r: float,
) -> torch.Tensor:
    """Generalized structure core (PyTorch).

    Uses escort_probs for r-weighting and weighted_power_mean for q-averaging.

    ⟨Λ_n⟩_{q,r} = M_q(compliances, escort_r(probs))
    """
    if compliances.shape != probs.shape:
        raise ValueError("compliances and probs must have same shape")
    if compliances.numel() == 0:
        return torch.tensor(0.0, device=compliances.device)

    # Convert probs to logprobs for escort_probs
    logprobs = probs.clamp(min=_EPS).log()

    # Compute r-escort weights using escort_probs
    escort_weights = escort_probs(logprobs, r)

    # Apply weighted power mean with exponent q
    return weighted_power_mean(compliances, escort_weights, q)


def generalized_structure_core(
    compliances: Nums,
    probs: Nums,
    q: float = 1.0,
    r: float = 1.0,
) -> Num:
    """Generalized structure core via escort power mean.

    ⟨α_i^(q,r)⟩ = (Σ_y p(y)^r α_i(y)^q / Σ_y p(y)^r)^(1/q)

    Special cases:
        q=1, r=1: Standard expected compliance ⟨α_i⟩
        q=1, r=0: Uniform average over support
        q=1, r=∞: Compliance of the mode
        q=∞, r=1: Maximum compliance in support
        q=-∞, r=∞: Minimum compliance among modes

    Args:
        compliances: Compliance values α_i(y) for each trajectory
        probs: Probability p(y) for each trajectory
        q: Power mean order for compliances
        r: Escort order for probabilities

    Returns:
        Generalized structure core value
    """
    if is_tensor(compliances) and is_tensor(probs):
        return _generalized_structure_core_torch(compliances, probs, q, r)
    return _generalized_structure_core_native(list(compliances), list(probs), q, r)


def generalized_system_core(
    system_compliances: Sequence[SystemCompliance],
    probs: Nums,
    q: float = 1.0,
    r: float = 1.0,
) -> list[float]:
    """Generalized system core via escort power mean.

    Computes generalized_structure_core for each structure.

    Args:
        system_compliances: List of Λ_n(y) for each trajectory y
        probs: Probability p(y) for each trajectory y
        q: Power mean order
        r: Escort order

    Returns:
        Generalized system core ⟨Λ_n^(q,r)⟩
    """
    if not system_compliances:
        return []

    n_structures = len(system_compliances[0])
    probs_list = list(probs) if not is_tensor(probs) else probs.tolist()

    core = []
    for i in range(n_structures):
        structure_compliances = [sc[i] for sc in system_compliances]
        val = generalized_structure_core(structure_compliances, probs_list, q, r)
        core.append(float(val) if is_tensor(val) else val)

    return core


# ══════════════════════════════════════════════════════════════════════════════
# RELATIVE ENTROPY DEVIANCE
# ══════════════════════════════════════════════════════════════════════════════


def excess_deviance(
    compliance: SystemCompliance,
    core: SystemCore,
    alpha: float = 1.0,
) -> float:
    """Excess deviance: effective over-compliance.

    ∂⁺_α = exp(D_α(Λ_norm(y) || ⟨Λ_norm⟩))

    Measures how much the string over-complies with certain structures.
    Both inputs are normalized to probability distributions before computing divergence.

    Args:
        compliance: System compliance Λ_n(y) (will be normalized)
        core: System core ⟨Λ_n⟩ (will be normalized)
        alpha: Divergence order (1.0 = KL divergence)

    Returns:
        Excess deviance ∈ [1, ∞). Value of 1 means compliance matches core.
    """
    c_list = list(compliance) if not is_tensor(compliance) else compliance.tolist()
    core_list = list(core) if not is_tensor(core) else core.tolist()
    # renyi_divergence normalizes both inputs by default
    h = renyi_divergence(c_list, core_list, alpha=alpha, normalize=True)
    return math.exp(h) if math.isfinite(h) else float("inf")


def deficit_deviance(
    compliance: SystemCompliance,
    core: SystemCore,
    alpha: float = 1.0,
) -> float:
    """Deficit deviance: effective under-compliance.

    ∂⁻_α = exp(D_α(⟨Λ_norm⟩ || Λ_norm(y)))

    Measures how much the string under-complies with certain structures.
    Note the reversed argument order compared to excess_deviance.
    Both inputs are normalized to probability distributions before computing divergence.

    Args:
        compliance: System compliance Λ_n(y) (will be normalized)
        core: System core ⟨Λ_n⟩ (will be normalized)
        alpha: Divergence order (1.0 = KL divergence)

    Returns:
        Deficit deviance ∈ [1, ∞). Value of 1 means compliance matches core.
    """
    c_list = list(compliance) if not is_tensor(compliance) else compliance.tolist()
    core_list = list(core) if not is_tensor(core) else core.tolist()
    # renyi_divergence normalizes both inputs by default
    # Note: arguments reversed - core is p, compliance is q
    h = renyi_divergence(core_list, c_list, alpha=alpha, normalize=True)
    return math.exp(h) if math.isfinite(h) else float("inf")


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE DEVIANCE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════


def expected_deviance(
    compliances: Sequence[SystemCompliance],
    core: SystemCore,
    weights: Sequence[float] | None = None,
    norm: NormType = "l2",
) -> float:
    """Expected deviance E[∂_n] over a set of samples.

    Computes probability-weighted mean of deviances.

    Args:
        compliances: List of system compliances Λ_n(y) for each sample
        core: System core ⟨Λ_n⟩
        weights: Optional probability weights (uniform if None)
        norm: Norm type for deviance computation

    Returns:
        Expected deviance E[∂_n]
    """
    if not compliances:
        return 0.0

    n = len(compliances)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        # Normalize weights
        total = sum(weights)
        if total < _EPS:
            weights = [1.0 / n] * n
        else:
            weights = [w / total for w in weights]

    total_deviance = 0.0
    for compliance, w in zip(compliances, weights):
        d = deviance(compliance, core, norm)
        total_deviance += w * (float(d) if is_tensor(d) else d)

    return total_deviance


def deviance_variance(
    compliances: Sequence[SystemCompliance],
    core: SystemCore,
    weights: Sequence[float] | None = None,
    norm: NormType = "l2",
) -> float:
    """Variance of deviance Var[∂_n] over a set of samples.

    Computes probability-weighted variance of deviances.

    Args:
        compliances: List of system compliances Λ_n(y) for each sample
        core: System core ⟨Λ_n⟩
        weights: Optional probability weights (uniform if None)
        norm: Norm type for deviance computation

    Returns:
        Variance of deviance Var[∂_n]
    """
    if not compliances:
        return 0.0

    n = len(compliances)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        if total < _EPS:
            weights = [1.0 / n] * n
        else:
            weights = [w / total for w in weights]

    # Compute all deviances
    deviances = [
        float(deviance(c, core, norm))
        if is_tensor(deviance(c, core, norm))
        else deviance(c, core, norm)
        for c in compliances
    ]

    # E[∂]
    mean = sum(w * d for w, d in zip(weights, deviances))

    # E[(∂ - E[∂])²] = E[∂²] - E[∂]²
    mean_sq = sum(w * d * d for w, d in zip(weights, deviances))

    return mean_sq - mean * mean


def expected_orientation(
    compliances: Sequence[SystemCompliance],
    core: SystemCore,
    weights: Sequence[float] | None = None,
) -> list[float]:
    """Expected orientation E[θ_n] over a set of samples.

    Computes probability-weighted mean of orientation vectors.

    Args:
        compliances: List of system compliances Λ_n(y) for each sample
        core: System core ⟨Λ_n⟩
        weights: Optional probability weights (uniform if None)

    Returns:
        Expected orientation E[θ_n] as a list
    """
    if not compliances:
        return []

    n = len(compliances)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        if total < _EPS:
            weights = [1.0 / n] * n
        else:
            weights = [w / total for w in weights]

    # Get dimensionality from first compliance
    core_list = list(core) if not is_tensor(core) else core.tolist()
    dim = len(core_list)
    result = [0.0] * dim

    for compliance, w in zip(compliances, weights):
        theta = orientation(compliance, core)
        theta_list = list(theta) if not is_tensor(theta) else theta.tolist()
        for i in range(dim):
            result[i] += w * theta_list[i]

    return result


