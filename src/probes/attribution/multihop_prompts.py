"""MuSiQue multi-hop QA prompt format, gold-chain linearizer, answer scoring, and answer-span gate.

The four arithmetic-specific seams of the GSM8K procedure apparatus, ported to open-book multi-hop QA
for the generality test (``tasks/multihop_generality.md``). Pure string/logic only (no ``datasets``,
no torch), so the contract is CPU/offline-testable exactly like :mod:`gsm8k_prompts`.

The prompt mirrors MetaMath's Alpaca structure so the CoT continues from a fixed prefix (``Let's
think step by step.``) — the same fixed-continuation-point assumption the lockstep oracle / logit-lens
infra relies on. Open-book: the gold supporting passages are placed in the instruction; the LoRA
learns the multi-hop *composition* over given facts (the analogue of GSM8K's in-problem numbers),
not parametric fact recall.
"""

from __future__ import annotations

import re
from typing import Callable, List, Sequence, Tuple

INSTRUCTION_PREAMBLE = "Read the following passages and answer the question with step-by-step reasoning."

# Alpaca wrapper isomorphic to ``gsm8k_prompts.METAMATH_TEMPLATE``: a single ``{instruction}`` slot
# ending at the fixed CoT continuation point. This lets the attribution drivers treat a precomposed
# multi-hop instruction exactly like a one-arg MetaMath question, while ``multihop_prompt(q, passages)``
# stays the convenience builder used to format training/eval data.
MULTIHOP_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
)

ANSWER_MARKER = "The answer is:"
_REF_RE = re.compile(r"#(\d+)")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def render_passages(passages: Sequence[Tuple[str, str]]) -> str:
    """Render ``(title, text)`` passages as a numbered block for the prompt."""
    return "\n".join(f"Passage {i + 1} ({title}): {text}"
                     for i, (title, text) in enumerate(passages))


def build_instruction(question: str, passages: Sequence[Tuple[str, str]]) -> str:
    """The instruction body (preamble + numbered passages + question) that fills the template.

    Drivers store this string as the problem's "question" and wrap it with
    :func:`multihop_prompt_from_instruction`, mirroring how GSM8K passes a bare question to
    ``metamath_prompt`` — the passages ride inside the instruction (open-book).
    """
    return f"{INSTRUCTION_PREAMBLE}\n\n{render_passages(passages)}\n\nQuestion: {question}"


def multihop_prompt_from_instruction(instruction: str) -> str:
    """Wrap a precomposed instruction in the Alpaca template (one-arg, driver-facing prompt)."""
    return MULTIHOP_TEMPLATE.format(instruction=instruction)


def multihop_prompt(question: str, passages: Sequence[Tuple[str, str]]) -> str:
    """The open-book multi-hop prompt; CoT tokens follow the fixed ``Let's think step by step.``."""
    return multihop_prompt_from_instruction(build_instruction(question, passages))


def _resolve_ref(match: "re.Match", answers: List[str]) -> str:
    """Substitute ``#k`` with hop-k's answer when in range; else keep the literal.

    Almost all ``#k`` are 1-indexed back-references to an earlier hop. A few MuSiQue questions contain
    a literal ``#k`` that is part of a title (e.g. the song "#9 Dream"), not a reference — these point
    out of range, so we leave them verbatim rather than crash or mis-substitute.
    """
    k = int(match.group(1)) - 1
    return answers[k] if 0 <= k < len(answers) else match.group(0)


def resolve_decomposition(decomposition: Sequence[dict]) -> List[Tuple[str, str]]:
    """Linearize a MuSiQue ``question_decomposition`` into resolved ``(sub_question, answer)`` hops.

    Each hop's ``question`` may reference an earlier hop's answer as ``#k`` (1-indexed); substitute it
    so the chain reads as natural worked steps. Out-of-range ``#k`` (title literals, not references)
    are preserved verbatim.
    """
    steps: List[Tuple[str, str]] = []
    answers: List[str] = []
    for hop in decomposition:
        q = _REF_RE.sub(lambda m: _resolve_ref(m, answers), hop["question"])
        steps.append((q, hop["answer"]))
        answers.append(hop["answer"])
    return steps


def format_multihop_solution(decomposition: Sequence[dict]) -> str:
    """Render the gold decomposition as a worked CoT ending in ``The answer is: <final>``."""
    steps = resolve_decomposition(decomposition)
    lines = [f"Step {i + 1}: {q} {a}." for i, (q, a) in enumerate(steps)]
    final = steps[-1][1] if steps else ""
    return "\n".join(lines + [f"{ANSWER_MARKER} {final}"])


def extract_pred_answer(completion: str) -> str | None:
    """Parse the model's predicted answer from the ``The answer is: `` tail (last occurrence)."""
    parts = completion.split(ANSWER_MARKER)
    if len(parts) <= 1:
        return None
    # first line after the final marker (the answer span ends at a newline / restatement)
    tail = parts[-1].strip()
    return tail.split("\n")[0].strip() or None


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles, collapse whitespace."""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _ARTICLES_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def answer_match(pred: str | None, gold: str, aliases: Sequence[str] = ()) -> bool:
    """True iff a prediction was parsed and normalized-exact-matches gold or any alias."""
    if pred is None:
        return False
    p = normalize_answer(pred)
    return any(p == normalize_answer(g) for g in (gold, *aliases))


def answer_span_gate(marker: str = ANSWER_MARKER, complement: bool = False) -> Callable[[List[str], int], bool]:
    """A ``gated_lockstep_generate`` gate firing inside the final-answer span.

    The span opens once ``marker`` is present in the already-emitted tokens (no look-ahead on the
    current step). ``complement=True`` gives the reasoning-span gate; the two are exact complements,
    the multi-hop analogue of GSM8K's ``result_only`` / ``planning_only`` partition.
    """
    def gate(decoded: List[str], step: int) -> bool:  # noqa: ARG001 (step unused; signature parity)
        in_answer = marker in "".join(decoded)
        return (not in_answer) if complement else in_answer

    return gate
