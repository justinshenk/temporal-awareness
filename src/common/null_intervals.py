"""Exact binomial bounds for results reported as zero.

Every rung of the procedure ladder is reported as ``0.00``. A bare zero is not a claim: it does
not say whether the run could have detected a recovery of 0.05, or of 0.30. The reader cannot
tell an established null from an underpowered one, and "you were underpowered" is the standard
and unanswerable objection to an interpretability negative result.

The bootstrap in :mod:`src.common.bootstrap_stats` cannot supply the missing bound. Resampling a
sample that is entirely zeros yields only zeros, so its interval collapses to ``[0, 0]``
regardless of ``n`` — it reports the *observed* spread, and an unobserved success has no spread
to report. That is tested explicitly in ``tests/common/test_null_intervals.py``.

What is wanted instead is the exact (Clopper-Pearson) binomial interval, which inverts the
binomial test and so bounds the rates that remain *consistent* with having seen no successes.
For ``0`` hits in ``n`` trials the two-sided upper limit has the closed form
``1 - (alpha/2) ** (1/n)`` — about 0.12 at ``n = 30`` and 0.17 at ``n = 20``. Those are the
honest bounds on the ladder's nulls, and they are looser than the bare ``0.00`` suggests.

Reported on the recovery scale, ``(acc - base) / (lora - base)``, the interval treats the base
and donor references as **known constants**. It therefore covers sampling error in the steered
run only, not in the two reference measurements, and is an underestimate of total uncertainty to
that extent. Say so wherever these numbers appear.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import beta

from src.common.base_schema import BaseSchema


def clopper_pearson(hits: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided ``1 - alpha`` binomial interval for ``hits`` successes in ``n`` trials.

    Clamped at the ends: ``hits == 0`` gives a lower limit of exactly 0 (and a positive upper
    limit), ``hits == n`` gives an upper limit of exactly 1.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= hits <= n:
        raise ValueError(f"hits must lie in [0, {n}], got {hits}")

    lo = 0.0 if hits == 0 else float(beta.ppf(alpha / 2, hits, n - hits + 1))
    hi = 1.0 if hits == n else float(beta.ppf(1 - alpha / 2, hits + 1, n - hits))
    return lo, hi


@dataclass
class BoundedNull(BaseSchema):
    """A measured rate with an exact binomial interval, optionally rescaled to recovery.

    ``recovery*`` fields are populated only when base and donor references are supplied; the
    recovery scale is ``(rate - base_acc) / (lora_acc - base_acc)``.
    """

    hits: int
    n: int
    rate: float
    rate_lo: float
    rate_hi: float
    recovery: float | None = None
    recovery_lo: float | None = None
    recovery_hi: float | None = None
    budget: float | None = None

    def render(self, digits: int = 2) -> str:
        """One-line form for a results table, on the recovery scale where one exists."""
        if self.recovery is None:
            return (f"{self.rate:.{digits}f} "
                    f"[{self.rate_lo:.{digits}f}, {self.rate_hi:.{digits}f}] (n={self.n})")
        return (f"{self.recovery:.{digits}f} "
                f"[{self.recovery_lo:.{digits}f}, {self.recovery_hi:.{digits}f}] (n={self.n})")


def bounded_null(hits: int, n: int, base_acc: float | None = None,
                 lora_acc: float | None = None, alpha: float = 0.05) -> BoundedNull:
    """Bound a rate, and rescale the bound to recovery when references are given."""
    lo, hi = clopper_pearson(hits, n, alpha)
    rate = hits / n
    result = BoundedNull(hits=hits, n=n, rate=rate, rate_lo=lo, rate_hi=hi)

    if base_acc is None or lora_acc is None:
        return result

    budget = lora_acc - base_acc
    if budget == 0:
        raise ValueError("zero budget: base and donor accuracies are equal, so recovery is undefined")
    if budget < 0:
        raise ValueError(f"negative budget: donor scores below base ({lora_acc} < {base_acc}), so "
                         f"there is nothing to recover and dividing by it would invert the "
                         f"interval. Report this arm on the accuracy scale instead.")

    result.budget = budget
    result.recovery = (rate - base_acc) / budget
    result.recovery_lo = (lo - base_acc) / budget
    result.recovery_hi = (hi - base_acc) / budget
    return result


def bounded_null_from_rate(rate: float, n: int, base_acc: float | None = None,
                           lora_acc: float | None = None, alpha: float = 0.05) -> BoundedNull:
    """Bound a rate stored as an accuracy rather than a hit count.

    The committed artifacts record ``steer_acc`` and ``n_eval``, not the underlying successes, so
    the count is recovered as ``rate * n`` and must be integral to within floating-point noise —
    a non-integral count means the rate and n do not belong to the same run.
    """
    exact = rate * n
    hits = round(exact)
    if abs(exact - hits) > 1e-6:
        raise ValueError(f"rate {rate} x n {n} = {exact} is not a whole number of successes; "
                         f"the rate and n do not come from the same run")
    return bounded_null(hits, n, base_acc, lora_acc, alpha)
