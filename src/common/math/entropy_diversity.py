"""Core entropy and diversity theory (unified framework).

Mathematical foundations based on Hill numbers and power means.
See: https://arxiv.org/pdf/2012.02113

Key insight: All standard diversity measures (Shannon, Simpson, richness)
are special cases of a single parametric family indexed by order q.

All functions use log-space arithmetic internally for numerical stability.

Hierarchy of concepts:
- surprise / rarity: pointwise measures at single positions
- power_mean: generalized mean of order α
- q_diversity: Hill number D_q (effective number of categories)
- q_abundance: power mean of probabilities from logprobs (for trajectories)

The parameter q controls sensitivity to rare vs common categories:
    q → -∞: dominated by rarest category
    q = 0:  richness (count of non-zero categories)
    q = 1:  Shannon diversity exp(H)
    q = 2:  Simpson diversity 1/Σpᵢ²
    q → +∞: dominated by most common category

Type handling:
- All public functions accept both Sequence[float] and torch.Tensor
- Dispatch to optimized implementations based on input type
- Use `Nums` type alias for brevity
"""

from __future__ import annotations

import math
from typing import Sequence, Union

import torch
import torch.nn.functional as F

# ── Type aliases ────────────────────────────────────────────────────────────────

Nums = Union[Sequence[float], torch.Tensor]
"""Union type for numeric sequences: Sequence[float] | torch.Tensor"""

_EPS = 1e-12


def _is_tensor(x: Nums) -> bool:
    return isinstance(x, torch.Tensor)


# ══════════════════════════════════════════════════════════════════════════════
# Conversion helpers
# ══════════════════════════════════════════════════════════════════════════════


def probs_to_logprobs(probs: Nums) -> Nums:
    """Convert probabilities to log-probabilities."""
    if _is_tensor(probs):
        return torch.log(probs.clamp(min=_EPS))
    return [math.log(p) if p > _EPS else float("-inf") for p in probs]


def logprobs_to_probs(logprobs: Nums) -> Nums:
    """Convert log-probabilities to probabilities."""
    if _is_tensor(logprobs):
        return logprobs.exp()
    return [math.exp(lp) if math.isfinite(lp) else 0.0 for lp in logprobs]


# ══════════════════════════════════════════════════════════════════════════════
# Log-sum-exp (numerical stability primitive)
# ══════════════════════════════════════════════════════════════════════════════


def _log_sum_exp_native(values: Sequence[float]) -> float:
    """Compute log(Σ exp(xᵢ)) in a numerically stable way (pure Python)."""
    if not values:
        return float("-inf")
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("-inf")
    max_val = max(finite)
    return max_val + math.log(sum(math.exp(v - max_val) for v in finite))


def _log_sum_exp_torch(values: torch.Tensor) -> torch.Tensor:
    """Compute log(Σ exp(xᵢ)) using torch.logsumexp."""
    return torch.logsumexp(values, dim=-1)


def log_sum_exp(values: Nums) -> float | torch.Tensor:
    """Compute log(Σ exp(xᵢ)) in a numerically stable way."""
    if _is_tensor(values):
        return _log_sum_exp_torch(values)
    return _log_sum_exp_native(values)


# ══════════════════════════════════════════════════════════════════════════════
# Pointwise measures
# ══════════════════════════════════════════════════════════════════════════════


def surprise(logprob: float | torch.Tensor) -> float | torch.Tensor:
    """Information content (self-information) at a single position.

    s = -log p = log(1/p)

    Also known as: surprisal, Shannon information.
    Range: [0, ∞). Lower = less surprising (more expected).
    """
    return -logprob


def rarity(logprob: float | torch.Tensor) -> float | torch.Tensor:
    """Rarity of an outcome (inverse probability).

    r = 1/p = exp(-log p)

    Interpretation: "effective number of equiprobable alternatives."
    Range: [1, ∞). Lower = more common (model expected this).
    """
    if isinstance(logprob, torch.Tensor):
        return (-logprob).exp()
    return math.exp(-logprob)


# ══════════════════════════════════════════════════════════════════════════════
# Power means (foundation for Hill numbers)
# ══════════════════════════════════════════════════════════════════════════════


def _power_mean_native(values: Sequence[float], alpha: float) -> float:
    """Generalized (power) mean of order α (pure Python)."""
    if not values:
        return 0.0

    n = len(values)
    active = [v for v in values if v > _EPS]
    if not active:
        return 0.0

    # Limiting cases
    if alpha == float("-inf"):
        return min(active)
    if alpha == float("inf"):
        return max(active)
    if abs(alpha) < _EPS:
        # Geometric mean: exp(mean(log(x)))
        return math.exp(sum(math.log(v) for v in active) / n)

    # General case
    powered = sum(v**alpha for v in active)
    return (powered / n) ** (1.0 / alpha)


def _power_mean_torch(values: torch.Tensor, alpha: float) -> torch.Tensor:
    """Generalized (power) mean of order α (PyTorch)."""
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device)

    active = values[values > _EPS]
    if active.numel() == 0:
        return torch.tensor(0.0, device=values.device)

    n = values.numel()

    # Limiting cases
    if alpha == float("-inf"):
        return active.min()
    if alpha == float("inf"):
        return active.max()
    if abs(alpha) < _EPS:
        # Geometric mean: exp(mean(log(x)))
        return (active.log().sum() / n).exp()

    # General case
    powered = (active**alpha).sum()
    return (powered / n) ** (1.0 / alpha)


def power_mean(values: Nums, alpha: float) -> float | torch.Tensor:
    """Generalized (power) mean of order α.

    M_α(x₁, ..., xₙ) = (Σ xᵢ^α / n)^(1/α)

    Special cases:
        α → -∞: minimum
        α = -1: harmonic mean
        α → 0:  geometric mean
        α = 1:  arithmetic mean
        α = 2:  quadratic mean (RMS)
        α → +∞: maximum

    The power mean is monotonic in α: M_α ≤ M_β for α < β.
    """
    if _is_tensor(values):
        return _power_mean_torch(values, alpha)
    return _power_mean_native(values, alpha)


# ══════════════════════════════════════════════════════════════════════════════
# Rényi entropy (takes logprobs for stability)
# ══════════════════════════════════════════════════════════════════════════════


def _renyi_entropy_native(logprobs: Sequence[float], q: float) -> float:
    """Rényi entropy of order q (pure Python, takes logprobs)."""
    finite_lps = [lp for lp in logprobs if math.isfinite(lp)]
    if not finite_lps:
        return float("inf")

    # q = 0: Hartley entropy = log(count of non-zero)
    if q == 0:
        return math.log(len(finite_lps))

    # q = 1: Shannon entropy H = -Σ pᵢ log pᵢ = -Σ exp(lp) * lp
    if abs(q - 1.0) < _EPS:
        return -sum(math.exp(lp) * lp for lp in finite_lps)

    # q → ∞: min-entropy = -log(max p) = -max(lp)
    if q == float("inf"):
        return -max(finite_lps)

    # q → -∞: max-entropy = -log(min p) = -min(lp)
    if q == float("-inf"):
        return -min(finite_lps)

    # General case: use log-sum-exp for stability
    # H_q = (1/(1-q)) * log(Σ pᵢ^q) = (1/(1-q)) * log(Σ exp(q * lp))
    log_sum = _log_sum_exp_native([q * lp for lp in finite_lps])
    return log_sum / (1.0 - q)


def _renyi_entropy_torch(logprobs: torch.Tensor, q: float) -> torch.Tensor:
    """Rényi entropy of order q (PyTorch, takes logprobs)."""
    finite_mask = torch.isfinite(logprobs)
    if not finite_mask.any():
        return torch.tensor(float("inf"), device=logprobs.device)

    finite_lps = logprobs[finite_mask]

    # q = 0: Hartley entropy = log(count of non-zero)
    if q == 0:
        return torch.log(torch.tensor(finite_lps.numel(), dtype=logprobs.dtype, device=logprobs.device))

    # q = 1: Shannon entropy H = -Σ pᵢ log pᵢ = -Σ exp(lp) * lp
    if abs(q - 1.0) < _EPS:
        probs = finite_lps.exp()
        return -(probs * finite_lps).sum()

    # q → ∞: min-entropy = -log(max p) = -max(lp)
    if q == float("inf"):
        return -finite_lps.max()

    # q → -∞: max-entropy = -log(min p) = -min(lp)
    if q == float("-inf"):
        return -finite_lps.min()

    # General case: use log-sum-exp for stability
    log_sum = torch.logsumexp(q * finite_lps, dim=-1)
    return log_sum / (1.0 - q)


def renyi_entropy(logprobs: Nums, q: float) -> float | torch.Tensor:
    """Rényi entropy of order q (numerically stable, takes logprobs).

    H_q = (1/(1-q)) · log(Σ pᵢ^q)

    Special cases:
        q = 0:  log(S)      (Hartley entropy, log of richness)
        q = 1:  H           (Shannon entropy, via L'Hôpital)
        q = 2:  -log(Σpᵢ²)  (collision entropy)
        q → ∞:  -log(max pᵢ) (min-entropy)

    Connection to Hill numbers: D_q = exp(H_q)

    Args:
        logprobs: Log-probabilities (Sequence[float] or torch.Tensor)
        q: Order parameter
    """
    if _is_tensor(logprobs):
        return _renyi_entropy_torch(logprobs, q)
    return _renyi_entropy_native(logprobs, q)


def shannon_entropy(logprobs: Nums) -> float | torch.Tensor:
    """Shannon entropy (= renyi_entropy with q=1).

    H = -Σ pᵢ log pᵢ = -Σ exp(lpᵢ) * lpᵢ

    Takes logprobs for numerical stability.
    """
    return renyi_entropy(logprobs, q=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# q-Diversity (Hill numbers)
# ══════════════════════════════════════════════════════════════════════════════


def _q_diversity_native(logprobs: Sequence[float], q: float) -> float:
    """Hill number D_q (pure Python)."""
    finite_lps = [lp for lp in logprobs if math.isfinite(lp)]
    if not finite_lps:
        return 0.0

    # q = 0: richness = count of non-zero
    if q == 0:
        return float(len(finite_lps))

    # D_q = exp(H_q) for all other cases
    H_q = _renyi_entropy_native(finite_lps, q)
    return math.exp(H_q) if math.isfinite(H_q) else float("inf")


def _q_diversity_torch(logprobs: torch.Tensor, q: float) -> torch.Tensor:
    """Hill number D_q (PyTorch)."""
    finite_mask = torch.isfinite(logprobs)
    if not finite_mask.any():
        return torch.tensor(0.0, device=logprobs.device)

    finite_lps = logprobs[finite_mask]

    # q = 0: richness = count of non-zero
    if q == 0:
        return torch.tensor(finite_lps.numel(), dtype=logprobs.dtype, device=logprobs.device)

    # D_q = exp(H_q) for all other cases
    H_q = _renyi_entropy_torch(finite_lps, q)
    return H_q.exp()


def q_diversity(logprobs: Nums, q: float) -> float | torch.Tensor:
    """Hill number D_q: effective number of categories of order q.

    D_q = exp(H_q) where H_q is Rényi entropy.

    This is THE unified diversity measure. All standard indices are special cases:
        q → -∞: 1 / min pᵢ  (maximum rarity)
        q = 0:  richness S  (count of categories with p > 0)
        q = 1:  exp(H)      (Shannon diversity, via L'Hôpital)
        q = 2:  1 / Σpᵢ²    (Simpson diversity)
        q → +∞: 1 / max pᵢ  (Berger-Parker index)

    Range: [1, n] where n = number of categories.
    Higher = more diverse. Monotonically decreasing in q.

    Args:
        logprobs: Log-probabilities (Sequence[float] or torch.Tensor)
        q: Order parameter
    """
    if _is_tensor(logprobs):
        return _q_diversity_torch(logprobs, q)
    return _q_diversity_native(logprobs, q)


def _q_concentration_native(logprobs: Sequence[float], q: float) -> float:
    """Concentration of order q (pure Python)."""
    d = _q_diversity_native(logprobs, q)
    return 1.0 / d if d > _EPS else float("inf")


def _q_concentration_torch(logprobs: torch.Tensor, q: float) -> torch.Tensor:
    """Concentration of order q (PyTorch)."""
    d = _q_diversity_torch(logprobs, q)
    return 1.0 / d


def q_concentration(logprobs: Nums, q: float) -> float | torch.Tensor:
    """Concentration of order q (= 1/D_q).

    The "inverse diversity" - how concentrated is the distribution?
    Range: [1/n, 1]. Higher = more concentrated.
    """
    if _is_tensor(logprobs):
        return _q_concentration_torch(logprobs, q)
    return _q_concentration_native(logprobs, q)


# ══════════════════════════════════════════════════════════════════════════════
# q-Abundance (for trajectories)
# ══════════════════════════════════════════════════════════════════════════════


def _q_abundance_native(logprobs: Sequence[float], q: float) -> float:
    """Generalized abundance of order q (pure Python)."""
    if not logprobs:
        return 0.0

    m = len(logprobs)
    finite_lps = [lp for lp in logprobs if math.isfinite(lp)]
    if not finite_lps:
        return 0.0

    # Limiting cases
    if q == float("-inf"):
        return math.exp(max(finite_lps))
    if q == float("inf"):
        return math.exp(min(finite_lps))

    # q = 1: geometric mean = exp(mean(logprobs))
    if abs(q - 1.0) < _EPS:
        return math.exp(sum(finite_lps) / m)

    alpha = q - 1.0
    if abs(alpha) < _EPS:
        return math.exp(sum(finite_lps) / m)

    # General case: use log-sum-exp
    # A_q = (Σ pᵢ^q / n)^(1/(q-1)) = (Σ exp(q*lp) / n)^(1/(q-1))
    # = exp((log(Σ exp(q*lp)) - log(n)) / (q-1))
    log_sum = _log_sum_exp_native([q * lp for lp in finite_lps])
    return math.exp((log_sum - math.log(m)) / alpha)


def _q_abundance_torch(logprobs: torch.Tensor, q: float) -> torch.Tensor:
    """Generalized abundance of order q (PyTorch)."""
    if logprobs.numel() == 0:
        return torch.tensor(0.0, device=logprobs.device)

    m = logprobs.numel()
    finite_mask = torch.isfinite(logprobs)
    if not finite_mask.any():
        return torch.tensor(0.0, device=logprobs.device)

    finite_lps = logprobs[finite_mask]

    # Limiting cases
    if q == float("-inf"):
        return finite_lps.max().exp()
    if q == float("inf"):
        return finite_lps.min().exp()

    # q = 1: geometric mean = exp(mean(logprobs))
    if abs(q - 1.0) < _EPS:
        return (finite_lps.sum() / m).exp()

    alpha = q - 1.0
    if abs(alpha) < _EPS:
        return (finite_lps.sum() / m).exp()

    # General case: use log-sum-exp
    log_sum = torch.logsumexp(q * finite_lps, dim=-1)
    return ((log_sum - math.log(m)) / alpha).exp()


def q_abundance(logprobs: Nums, q: float) -> float | torch.Tensor:
    """Generalized abundance of order q: power mean of probabilities.

    A_q = M_{q-1}(p₁, ..., pₙ) = (Σ pᵢ^q / n)^(1/(q-1))

    For trajectory analysis, this represents the "typical probability"
    of observing the sequence, weighted by order q:
        q → -∞: max pᵢ  (best-case, most likely token)
        q = 0:  harmonic mean (pessimistic)
        q = 1:  geometric mean (standard inv-perplexity)
        q = 2:  arithmetic mean (optimistic)
        q → +∞: min pᵢ  (worst-case, least likely token)

    Range: (0, 1]. Higher = better (more confident predictions).

    Uses log-space arithmetic for numerical stability.
    """
    if _is_tensor(logprobs):
        return _q_abundance_torch(logprobs, q)
    return _q_abundance_native(logprobs, q)


# ══════════════════════════════════════════════════════════════════════════════
# Convenience wrappers for common orders
# ══════════════════════════════════════════════════════════════════════════════


def shannon_diversity(logprobs: Nums) -> float | torch.Tensor:
    """Shannon diversity D₁ = exp(H). Wraps q_diversity(q=1)."""
    return q_diversity(logprobs, q=1.0)


def simpson_diversity(logprobs: Nums) -> float | torch.Tensor:
    """Simpson diversity D₂ = 1/Σpᵢ². Wraps q_diversity(q=2)."""
    return q_diversity(logprobs, q=2.0)


def richness(logprobs: Nums) -> float | torch.Tensor:
    """Richness D₀ = count of non-zero categories. Wraps q_diversity(q=0)."""
    return q_diversity(logprobs, q=0.0)
