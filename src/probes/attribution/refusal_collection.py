"""Token-slice for the refusal-transfer fit (mirrors ``cot_collection`` for the safety axis).

Where the GSM8K fit took ``(a_t, δ_t)`` over the whole generated CoT (``[prompt_len:]``),
the refusal fit takes them over the **refusal-decision window**: the last prompt token
(the position whose residual decides the first response token) plus the early-decode
positions. The generation budget ``k_decode`` (``max_new_tokens`` at collection time)
bounds how far into the response we reach; this slice simply spans from the decision
token to the end of whatever was teacher-forced.

Fitting on a mixed harmful+benign prompt set over this window lets the ridge learn the
harm-conditionality for free — δ is large where the expert diverges (harmful handling)
and ~0 where it does not (benign) — with no hand-built gating.
"""

from __future__ import annotations


def refusal_token_slice(prompt_len: int, full_len: int) -> slice:
    """Positions of the refusal-decision window: last prompt token + all decoded tokens.

    ``prompt_len-1`` is the last prompt position (its residual predicts the first response
    token — the refusal decision); the window runs to ``full_len``. Requires ``prompt_len >= 1``
    so a decision position exists, and ``prompt_len <= full_len``.
    """
    if not 1 <= prompt_len <= full_len:
        raise ValueError(f"need 1 <= prompt_len <= full_len, got {prompt_len}, {full_len}")
    return slice(prompt_len - 1, full_len)
