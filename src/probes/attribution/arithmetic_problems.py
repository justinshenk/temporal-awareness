"""Single-operation arithmetic problems for the short-output steering probe.

The GSM8K finding is that static activation injection recovers 0.00 of the LoRA budget over
long (512-token) self-maintained chains of thought. This module builds the *short-output*
control: single-operation problems whose correct answer is reachable in a few tokens, phrased
in natural language (in-distribution for both the LoRA and the GSM8K-fit maps). Holding the
prompt/steering mechanism fixed and shrinking the output isolates "the long CoT trajectory
drifts" from "the compute does not transfer at all".

Tiers are chosen so the weak base fails but the LoRA succeeds while the answer stays short:
``mul2x2`` (two 2-digit factors), ``add3`` / ``sub3`` (two 3-digit terms with carries/borrows).
"""

from __future__ import annotations

import random

_PHRASING = {
    "mul2x2": ("times", lambda a, b: a * b),
    "add3": ("plus", lambda a, b: a + b),
    "sub3": ("minus", lambda a, b: a - b),
}
DEFAULT_TIERS = ("mul2x2", "add3", "sub3")


def _operands(tier: str, rng: random.Random) -> tuple[int, int]:
    """Sample an (a, b) operand pair for ``tier`` (both non-trivial; sub3 kept positive)."""
    if tier == "mul2x2":
        return rng.randint(12, 99), rng.randint(12, 99)
    a, b = rng.randint(100, 999), rng.randint(100, 999)
    if tier == "sub3" and a < b:
        a, b = b, a
    return a, b


def generate_arithmetic_problems(
    n: int, seed: int, tiers: tuple[str, ...] = DEFAULT_TIERS
) -> list[tuple[str, float, str]]:
    """Return ``n`` deterministic ``(question, gold_float, tier)`` problems, round-robin over tiers.

    ``question`` is natural language ("What is 23 times 47?"); ``gold_float`` is the exact answer
    as a float (so it feeds ``numeric_match`` directly); ``tier`` tags it for per-tier breakdown.
    """
    rng = random.Random(seed)
    out: list[tuple[str, float, str]] = []
    for i in range(n):
        tier = tiers[i % len(tiers)]
        word, op = _PHRASING[tier]
        a, b = _operands(tier, rng)
        out.append((f"What is {a} {word} {b}?", float(op(a, b)), tier))
    return out
