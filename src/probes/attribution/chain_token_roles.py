"""Per-token *role* labels for a gold reasoning chain — the seam of the plan-vs-execute lens (P4).

The gold-token teacher-forced lens (``scripts/attribution/gold_token_lens_gsm8k.py``) asks: with the
context pinned to a correct chain, which token roles can base predict? That needs a per-token role,
and the two procedures label roles by different means:

* **GSM8K** has a lexical delimiter — a result span opens on ``=`` and runs over whitespace/digit
  tokens — so roles come from an online state machine over decoded tokens (``computed`` /
  ``copied_digit`` / ``other``). Ported here verbatim from the ``computed_flags`` it replaces,
  whitespace semantics included (see :func:`gsm8k_token_roles`).
* **MuSiQue** has none: ``format_multihop_solution`` renders ``Step i: <sub-question> <answer>.``
  with only a space between plan and answer. Roles are therefore built by *construction* — rendered
  in lockstep with the chain itself (:func:`multihop_chain_spans`) — and mapped onto tokens through
  the fast tokenizer's character offsets (:func:`roles_from_offsets`). Nothing is searched for, so an
  answer that also appears inside its own sub-question cannot be mislabeled.

Pure string/offset logic (no torch, no ``datasets``), CPU/offline-testable like ``multihop_prompts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

from src.probes.attribution.multihop_prompts import ANSWER_MARKER, resolve_decomposition

ROLE_PROMPT = "prompt"                 # not scored (before the chain)
ROLE_SCAFFOLD = "scaffold"             # 'Step i:', '.', newlines, the answer marker, eos
ROLE_SUB_QUESTION = "sub_question"     # the plan: which question to ask at this hop
ROLE_HOP_ANSWER = "hop_answer"         # the execution: this hop's answer
ROLE_FINAL_ANSWER = "final_answer"     # the restatement after 'The answer is:' (a copy)

ROLE_COMPUTED = "computed"             # GSM8K: a digit inside an open post-'=' span
ROLE_COPIED_DIGIT = "copied_digit"     # GSM8K: any other digit
ROLE_OTHER = "other"


@dataclass(frozen=True)
class RoleSpan:
    """A half-open character range ``[start, end)`` of the chain carrying one role."""

    start: int
    end: int
    role: str
    hop: int | None = None


def multihop_chain_spans(decomposition: Sequence[dict]) -> Tuple[str, List[RoleSpan]]:
    """Render the gold chain **and** its role spans in one pass.

    Returns ``(chain_text, spans)`` where ``chain_text`` is the supervised target verbatim — the
    leading ``"\\n"`` join separator plus ``format_multihop_solution(decomposition)`` — and ``spans``
    tile it exactly (contiguous, non-overlapping, covering every character).

    Role boundaries always fall on the **space that opens the new role**, so a token carrying its
    leading space (Llama's ``▁Danny``) lies wholly inside one role rather than straddling two.
    """
    steps = resolve_decomposition(decomposition)
    if not steps:
        raise ValueError("empty decomposition: no gold chain to label")

    parts: List[Tuple[str, str, int | None]] = [("\n", ROLE_SCAFFOLD, None)]
    for i, (question, answer) in enumerate(steps):
        hop = i + 1
        if i > 0:
            parts.append(("\n", ROLE_SCAFFOLD, None))
        parts.append((f"Step {hop}:", ROLE_SCAFFOLD, hop))
        parts.append((f" {question}", ROLE_SUB_QUESTION, hop))
        parts.append((f" {answer}", ROLE_HOP_ANSWER, hop))
        parts.append((".", ROLE_SCAFFOLD, hop))
    parts.append(("\n", ROLE_SCAFFOLD, None))
    parts.append((ANSWER_MARKER, ROLE_SCAFFOLD, None))
    parts.append((f" {steps[-1][1]}", ROLE_FINAL_ANSWER, None))

    text, spans, cursor = "", [], 0
    for chunk, role, hop in parts:
        spans.append(RoleSpan(cursor, cursor + len(chunk), role, hop))
        text += chunk
        cursor += len(chunk)
    return text, spans


def roles_from_offsets(offsets: Sequence[Tuple[int, int]], spans: Sequence[RoleSpan]) -> List[dict]:
    """Map token character-offsets onto role spans; a token takes the role of its first character.

    A token that straddles a boundary therefore scores as the *earlier* role — the conservative
    choice, since it keeps a partially-planned token out of the ``hop_answer`` class.
    """
    out = []
    for start, _end in offsets:
        for s in spans:
            if s.start <= start < s.end:
                out.append({"role": s.role, "hop": s.hop})
                break
        else:
            raise ValueError(f"token offset {start} falls outside the chain spans")
    return out


def multihop_token_roles(tok, ids: Sequence[int], prompt_len: int, gold: Any) -> List[dict]:
    """Role per token of a teacher-forced ``prompt + gold chain [+ eos]`` sequence.

    Re-tokenizes the gold chain to obtain character offsets and **asserts the ids round-trip**, so a
    drift between this driver's join and the trained one (``multihop_data.encode_multihop``) fails
    loudly instead of silently mislabeling every role.
    """
    if not getattr(tok, "is_fast", False):
        raise ValueError("multi-hop roles need a fast tokenizer (character offset mapping)")

    text, spans = multihop_chain_spans(gold["decomposition"])
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    chain_ids, offsets = list(enc.input_ids), list(enc.offset_mapping)
    if list(ids[prompt_len:prompt_len + len(chain_ids)]) != chain_ids:
        raise ValueError("teacher-forced ids do not match the gold chain tokenization")

    roles: List[dict] = [{"role": ROLE_PROMPT, "hop": None} for _ in range(prompt_len)]
    roles += roles_from_offsets(offsets, spans)
    # anything past the chain is the appended eos
    roles += [{"role": ROLE_SCAFFOLD, "hop": None} for _ in range(len(ids) - len(roles))]
    return roles


def gsm8k_token_roles(tok, ids: Sequence[int], prompt_len: int, gold: Any) -> List[dict]:
    """Role per token for a MetaMath CoT: ``computed`` / ``copied_digit`` / ``other``.

    The post-``=`` result-span state machine, ported unchanged from the E1b driver: a span opens on
    any token containing ``=``, stays open across **whitespace** tokens (newlines included — this is
    what the committed GSM8K numbers were computed with, and deliberately differs from
    ``temporal_gate.in_result_span``, which closes on a newline for *gating*) and across digit tokens
    (flagging them), and closes on the first other token. The machine runs over the whole sequence,
    prompt included, so a ``=`` in the prompt carries over exactly as before; prompt positions are
    then overwritten with :data:`ROLE_PROMPT` because they are never scored.
    """
    decoded = [tok.decode([int(i)]) for i in ids]
    roles: List[dict] = []
    in_result = False
    for k, d in enumerate(decoded):
        role = ROLE_OTHER
        if "=" in d:
            in_result = True
        elif in_result:
            if d.strip() == "":
                pass                                       # whitespace token: stay in the span
            elif any(c.isdigit() for c in d):
                role = ROLE_COMPUTED
            else:
                in_result = False
        if role == ROLE_OTHER and any(c.isdigit() for c in d):
            role = ROLE_COPIED_DIGIT
        roles.append({"role": ROLE_PROMPT if k < prompt_len else role, "hop": None})
    return roles
