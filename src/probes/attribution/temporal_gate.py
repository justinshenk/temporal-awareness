"""Temporal gating of the lockstep oracle: patch only a *subset* of decode steps.

Phase 1 of the temporal decomposition (``tasks/temporal_oracle_decomposition.md``). The full-δ L20
oracle (``lockstep_oracle.lockstep_generate``) overwrites base's residual with the LoRA's at **every**
decode step. Here a per-step **gate** decides whether the step gets the overwrite
(``inject.set_values(caps)``) or passes through as plain base (``inject.set_values({})`` — the hook is
a no-op when a layer is absent). Skipping also skips the LoRA capture forward, so sparser gates are
strictly cheaper than the full oracle.

A gate is a callable ``gate(decoded, step) -> bool``: ``decoded`` is the list of decoded strings of
the tokens emitted *so far in the generation* (prompt excluded; the current step's token is not yet
known, so there is no look-ahead leak) and ``step`` is the 0-based decode index. ``True`` patches.

Two families:
  * **periodic(k)** — patch when ``step % k == 0`` (``k=1`` is the full oracle); the recovery-vs-
    fraction-patched headline curve;
  * **structural** — gate on the role of the token via the post-'=' result-span state machine shared
    with ``gold_token_lens_gsm8k.computed_flags``: ``result_only`` patches inside the result span
    (the computed digits after '='), ``planning_only`` is its exact complement (operators, step text,
    boundaries), ``step_boundary`` patches only the first token of each line (the thinnest scaffold).
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import torch

from src.probes.attribution.lockstep_oracle import OverwriteResidualHook

Gate = Callable[[List[str], int], bool]


def always(decoded: List[str], step: int) -> bool:
    return True


def never(decoded: List[str], step: int) -> bool:
    return False


def periodic(k: int) -> Gate:
    """Patch every ``k``-th step (steps 0, k, 2k, …); ``k=1`` is the full oracle."""
    if k < 1:
        raise ValueError(f"periodic period must be >= 1, got {k}")

    def gate(decoded: List[str], step: int) -> bool:
        return step % k == 0

    return gate


def in_result_span(decoded: List[str]) -> bool:
    """Whether, after consuming ``decoded``, we are inside an open post-'=' result span.

    Mirrors the state machine in ``gold_token_lens_gsm8k.computed_flags``: a span opens on any token
    containing '=', stays open across (non-newline) space and digit tokens, and closes on the first
    other token. The Llama tokenizer emits the inter-token space as its own piece and splits
    multi-digit numbers per digit, so '= 48' is '=', ' ', '4', '8' — all inside one span. Refinement
    over ``computed_flags`` for *gating*: a newline closes the span (a result does not cross a line
    boundary; the newline is itself a structural/planning token).
    """
    open_span = False
    for d in decoded:
        if "=" in d:
            open_span = True
        elif open_span:
            if "\n" in d:
                open_span = False                 # line boundary: close the span
            elif d.strip() == "":
                continue                          # space token: stay in span
            elif any(c.isdigit() for c in d):
                continue                          # result digit: stay in span
            else:
                open_span = False                 # any other token closes the span
    return open_span


def result_only(decoded: List[str], step: int) -> bool:
    """Patch only steps that emit a computed-result token (inside an open post-'=' span)."""
    return in_result_span(decoded)


def planning_only(decoded: List[str], step: int) -> bool:
    """Patch everything except computed-result tokens — the exact complement of ``result_only``."""
    return not in_result_span(decoded)


def step_boundary(decoded: List[str], step: int) -> bool:
    """Patch the first token of the generation and the first token after each newline."""
    return step == 0 or (len(decoded) > 0 and "\n" in decoded[-1])


@torch.no_grad()
def gated_lockstep_generate(
    input_ids: torch.Tensor,
    capture_residuals: Callable[[torch.Tensor], dict],
    base_logits: Callable[[torch.Tensor], torch.Tensor],
    inject: OverwriteResidualHook,
    inject_layers: Iterable[int],
    gate: Gate,
    decode_token: Callable[[int], str],
    max_new: int,
    eos_id: int | None,
    device,
) -> Tuple[torch.Tensor, List[bool]]:
    """Greedy lockstep dual-forward decode that patches only the steps the ``gate`` selects.

    Identical to ``lockstep_oracle.lockstep_generate`` except: before each step ``gate(decoded, step)``
    decides whether to inject. On a patched step the LoRA capture supplies the overwrite values; on a
    pass-through step the capture forward is skipped and the hook is fed ``{}`` (a no-op), so the base
    forward runs unperturbed. Returns the full ``(1, prompt+gen)`` ids and the per-step patched-mask.
    """
    inject_layers = list(inject_layers)
    S = input_ids
    decoded: List[str] = []
    mask: List[bool] = []
    for step in range(max_new):
        patch = bool(gate(decoded, step))
        if patch:
            caps = capture_residuals(S)
            inject.set_values({li: caps[li] for li in inject_layers})
        else:
            inject.set_values({})
        nxt = int(base_logits(S)[0, -1].argmax())
        S = torch.cat([S, torch.tensor([[nxt]], device=device)], dim=1)
        decoded.append(decode_token(nxt))
        mask.append(patch)
        if eos_id is not None and nxt == eos_id:
            break
    return S, mask
