"""Transcript construction for the E1 distance sweep and the E3 competition sweep.

Both experiments vary *one* property of an accumulated conversation while holding everything else
fixed, so the construction has to be a parameter rather than a forked driver — a fork drifts, and
a drifted arm is a confound that looks like a result.

Three invariants make the arms comparable, and each is pinned by a test:

- **The evidence always occupies its own user turn**, `local` included. Inlining the evidence into
  the question turn for `local` while splitting it out for `back_k` would vary turn structure
  alongside distance; here `local` is simply distance 0, so turn count and total text are
  identical across the whole ladder and only position moves.
- **The question text is byte-identical across arms.** E1 moves the evidence, never the question.
- **The overflow guard skips, never truncates, and records every skip.** §5 of the brief warns
  that truncating long items manufactures exactly the late-window dip the paper attributes to
  accumulation; an unlogged skip is indistinguishable from an item that never existed, which is
  why :class:`OverflowGuard` carries its own report.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema


@dataclass
class SkippedItem(BaseSchema):
    """One item the overflow guard refused, with enough detail to state the bias direction."""

    index: int
    reason: str
    n_tokens: int
    budget: int


@dataclass
class AssembledTranscript(BaseSchema):
    """A built conversation plus where the evidence and question landed in it."""

    turns: list[dict]
    question_text: str
    evidence_turn_indices: tuple[int, ...]
    question_turn_index: int


class ArmSpec:
    """The arm names from the brief's E1 and E3 tables."""

    DISTANCES = {"local": 0, "back_2": 2, "back_5": 5, "back_10": 10, "back_20": 20}
    SPLIT = "split"
    COMPETITION = ["unrelated", "same_subject", "near_dup"]

    @classmethod
    def distance_arms(cls) -> list[str]:
        return [*cls.DISTANCES, cls.SPLIT]

    @classmethod
    def competition_arms(cls) -> list[str]:
        return list(cls.COMPETITION)


class OverflowGuard:
    """Decide whether an item fits whole, and remember every one that did not.

    Near-full context a long item would be right-truncated by the tokenizer, losing its own
    options and scoring as a spurious error in exactly the top fill bin. Skipping is safe;
    truncating manufactures the effect under study.
    """

    def __init__(self, count_tokens, max_ctx: int, max_new: int, headroom: int = 16):
        self.count_tokens = count_tokens
        self.max_ctx = max_ctx
        self.max_new = max_new
        self.headroom = headroom
        self.skipped: list[SkippedItem] = []

    def fits(self, text: str, used: int, index: int) -> bool:
        budget = self.max_ctx - used - self.max_new - self.headroom
        n_tokens = self.count_tokens(text)
        if n_tokens > budget:
            self.skipped.append(SkippedItem(index=index, reason="overflow",
                                            n_tokens=n_tokens, budget=budget))
            return False
        return True

    @property
    def n_skipped(self) -> int:
        return len(self.skipped)

    def skip_rate(self, n_seen: int) -> float:
        return self.n_skipped / n_seen if n_seen else 0.0

    def report(self, n_seen: int) -> dict:
        """The per-arm skip summary every report must quote (brief §9)."""
        return {
            "n_seen": n_seen,
            "n_skipped": self.n_skipped,
            "skip_rate": self.skip_rate(n_seen),
            "skipped": [s.to_dict() for s in self.skipped],
        }


def _user_turn_positions(turns) -> list[int]:
    return [i for i, t in enumerate(turns) if t["role"] == "user"]


def assemble_transcript(prior_turns, evidence, question, distance, ack: str = "Noted.",
                        role: str = "user") -> AssembledTranscript:
    """Build a transcript with the evidence placed ``distance`` user turns before the question.

    ``evidence`` is one block (E1's ``local``/``back_k``) or several (``split``), and ``distance``
    is correspondingly one integer or one per block. Distance counts the user turns standing
    *between* the evidence turn and the question turn, so ``distance=0`` puts the evidence turn
    immediately before the question.
    """
    blocks = [evidence] if isinstance(evidence, str) else list(evidence)
    depths = [distance] if isinstance(distance, int) else list(distance)
    if len(blocks) != len(depths):
        raise ValueError(f"{len(blocks)} evidence block(s) but {len(depths)} distance(s)")

    turns = [dict(t) for t in prior_turns]
    available = len(_user_turn_positions(turns))
    deepest = max(depths)
    if deepest > available:
        raise ValueError(
            f"insufficient depth: need {deepest} prior user turns to place the evidence, "
            f"transcript has {available}. Skip the item rather than clamping the distance, "
            f"which would silently mislabel the arm.")

    # Insert shallowest first. Depth is counted back from where the question will go, so an
    # insertion *closer* to the question adds a user turn behind any block already placed and
    # pushes it one deeper; going shallow-to-deep means each later insertion lands farther back
    # and leaves the earlier ones at the depth they were asked for.
    marker = object()
    for block, depth in sorted(zip(blocks, depths), key=lambda bd: bd[1]):
        users = _user_turn_positions(turns)
        at = len(turns) if depth == 0 else users[len(users) - depth]
        turns[at:at] = [{"role": role, "content": block, "_evidence": marker},
                        {"role": "assistant", "content": ack}]

    turns.append({"role": role, "content": question})
    question_turn_index = len(turns) - 1

    evidence_turn_indices = tuple(i for i, t in enumerate(turns) if t.get("_evidence") is marker)
    for t in turns:
        t.pop("_evidence", None)

    return AssembledTranscript(turns=turns, question_text=question,
                               evidence_turn_indices=evidence_turn_indices,
                               question_turn_index=question_turn_index)


def _answer_identity(item) -> str:
    return item["choices"][item["gold_index"]]


def select_competitors(pool, current, arm: str, n: int, seed: int) -> list[dict]:
    """Pick ``n`` accumulated-context items of the requested confusability.

    Every arm excludes the current item and any item whose *correct option* matches the current
    item's, so no arm — least of all ``near_dup``, which is built to look like the current
    question — can leak the answer into the context it is supposed to compete with.
    """
    if arm not in ArmSpec.COMPETITION:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ArmSpec.COMPETITION}")

    current_answer = _answer_identity(current)
    candidates = [it for it in pool
                  if it["question"] != current["question"]
                  and _answer_identity(it) != current_answer]

    if arm in ("same_subject", "near_dup"):
        candidates = [it for it in candidates if it["subject"] == current["subject"]]

    rng = random.Random(seed)
    if arm == "near_dup":
        # Most confusable first: share the most option text with the current item.
        current_options = set(current["choices"])
        rng.shuffle(candidates)  # seeded tie-break, so ordering is reproducible
        candidates.sort(key=lambda it: -len(current_options & set(it["choices"])))
        return candidates[:n]

    rng.shuffle(candidates)
    return candidates[:n]


@dataclass
class AssemblyReport(BaseSchema):
    """Per-arm provenance a driver writes alongside its results (brief §9)."""

    arm: str
    n_seen: int = 0
    n_used: int = 0
    overflow: dict = field(default_factory=dict)
