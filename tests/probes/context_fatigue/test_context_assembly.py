"""Tests for E1/E3 transcript construction.

E1 varies *where the answer-bearing evidence lives* while holding items, fill and metric fixed;
E3 varies *how confusable the accumulated context is* while holding distance and fill fixed. Both
therefore need transcript construction to be a parameter rather than a forked driver, and both are
only interpretable if the arms differ in exactly the intended dimension — hence the byte-equality
and no-leak assertions here.

The overflow guard is tested hardest, because §5 warns that truncating long items manufactures
exactly the late-window dip the paper attributes to accumulation. Skipping is safe; truncating is
not; and a skip nobody logged is indistinguishable from an item that never existed.

A word-count stand-in stands in for a tokenizer so these run instantly with no model.
"""

import pytest

from src.probes.context_fatigue.context_assembly import (
    ArmSpec,
    OverflowGuard,
    assemble_transcript,
    select_competitors,
)

ACK = "Noted."


def word_tokens(text: str) -> int:
    """Deterministic stand-in for a tokenizer: one token per whitespace-delimited word."""
    return len(text.split())


def _prior_turns(n_exchanges: int):
    turns = []
    for i in range(n_exchanges):
        turns.append({"role": "user", "content": f"prior case {i}"})
        turns.append({"role": "assistant", "content": f"answer {i}"})
    return turns


def _user_indices(turns):
    return [i for i, t in enumerate(turns) if t["role"] == "user"]


def _depth(out, evidence_index):
    """User turns standing between the evidence turn and the question turn."""
    users = _user_indices(out.turns)
    return users.index(out.question_turn_index) - users.index(evidence_index) - 1


# ── evidence placement (E1) ─────────────────────────────────────────────
#
# Every arm gives the evidence its own user turn, `local` included: if `local` inlined the
# evidence into the question turn while `back_k` split it out, the arms would differ in turn
# structure as well as in distance, and E1's whole point is that distance is the only thing
# that moves. `local` is therefore distance 0 — the evidence turn immediately precedes the
# question turn — which keeps turn count and total text identical across the ladder.

@pytest.mark.parametrize("k", [0, 2, 5, 10, 20])
def test_back_k_places_evidence_exactly_k_user_turns_back(k):
    """§8: assert the exact turn index of the evidence block for each k."""
    prior = _prior_turns(30)
    out = assemble_transcript(prior, evidence="EVIDENCE BLOCK", question="QUESTION?",
                              distance=k, ack=ACK)

    assert _depth(out, out.evidence_turn_indices[0]) == k
    assert out.turns[out.evidence_turn_indices[0]]["content"] == "EVIDENCE BLOCK"


def test_local_arm_is_distance_zero():
    prior = _prior_turns(10)
    out = assemble_transcript(prior, evidence="E", question="Q", distance=0, ack=ACK)
    assert _depth(out, out.evidence_turn_indices[0]) == 0
    assert out.evidence_turn_indices[0] < out.question_turn_index


@pytest.mark.parametrize("k", [0, 2, 5, 10, 20])
def test_question_text_is_byte_identical_across_distances(k):
    """§8: the question span must be byte-equal to the `local` arm, or E1 varies two things."""
    prior = _prior_turns(30)
    local = assemble_transcript(prior, evidence="E", question="QUESTION?", distance=0, ack=ACK)
    moved = assemble_transcript(prior, evidence="E", question="QUESTION?", distance=k, ack=ACK)
    assert moved.question_text == local.question_text
    assert moved.turns[moved.question_turn_index]["content"] == \
        local.turns[local.question_turn_index]["content"]


@pytest.mark.parametrize("k", [0, 2, 5, 10, 20])
def test_turn_count_is_matched_across_distances(k):
    """Distance must be the only thing that moves: same number of turns, same total text."""
    prior = _prior_turns(30)
    local = assemble_transcript(prior, evidence="E", question="Q", distance=0, ack=ACK)
    moved = assemble_transcript(prior, evidence="E", question="Q", distance=k, ack=ACK)
    assert len(moved.turns) == len(local.turns)
    assert sorted(t["content"] for t in moved.turns) == sorted(t["content"] for t in local.turns)


def test_evidence_appears_exactly_once():
    prior = _prior_turns(20)
    out = assemble_transcript(prior, evidence="UNIQUE-EVIDENCE", question="Q", distance=7, ack=ACK)
    assert sum(t["content"].count("UNIQUE-EVIDENCE") for t in out.turns) == 1


def test_split_arm_places_evidence_at_two_depths():
    prior = _prior_turns(30)
    out = assemble_transcript(prior, evidence=["PART-A", "PART-B"], question="Q",
                              distance=(4, 12), ack=ACK)
    assert sorted(_depth(out, i) for i in out.evidence_turn_indices) == [4, 12]


def test_insufficient_depth_is_reported_not_silently_clamped():
    """A transcript too short for k must be refused; clamping would mislabel the arm."""
    prior = _prior_turns(3)
    with pytest.raises(ValueError, match="depth"):
        assemble_transcript(prior, evidence="E", question="Q", distance=20, ack=ACK)


# ── overflow guard ──────────────────────────────────────────────────────

def test_guard_skips_rather_than_truncates_and_logs_it():
    """§8: over-long items absent from results, present in the skip log, nothing truncated."""
    guard = OverflowGuard(count_tokens=word_tokens, max_ctx=50, max_new=5, headroom=2)
    items = [("short one", 0), ("w " * 80, 1), ("also short", 2), ("x " * 100, 3)]

    kept = []
    for text, idx in items:
        if guard.fits(text, used=10, index=idx):
            kept.append(idx)

    assert kept == [0, 2]
    assert [s.index for s in guard.skipped] == [1, 3]
    assert all(s.reason == "overflow" for s in guard.skipped)
    for skip in guard.skipped:
        assert skip.n_tokens > skip.budget  # the log says *why*, not just *that*


def test_guard_skip_rate_is_reportable():
    """§5 requires reporting which items were skipped and the direction of the bias."""
    guard = OverflowGuard(count_tokens=word_tokens, max_ctx=50, max_new=5, headroom=2)
    for i in range(10):
        guard.fits("w " * (100 if i % 5 == 0 else 2), used=10, index=i)
    assert guard.n_skipped == 2
    assert guard.skip_rate(n_seen=10) == pytest.approx(0.2)


def test_guard_never_reports_a_fit_it_cannot_honour():
    """The budget must account for generation headroom, not just the prompt."""
    guard = OverflowGuard(count_tokens=word_tokens, max_ctx=20, max_new=5, headroom=2)
    # 12 used + 6 prompt + 5 gen + 2 headroom = 25 > 20
    assert not guard.fits("w " * 6, used=12, index=0)


def test_guard_with_no_skips_is_still_reportable():
    guard = OverflowGuard(count_tokens=word_tokens, max_ctx=1000, max_new=5, headroom=2)
    assert guard.fits("tiny", used=0, index=0)
    assert guard.n_skipped == 0
    assert guard.skip_rate(n_seen=1) == 0.0
    assert guard.skipped == []


# ── competitor selection (E3) ───────────────────────────────────────────

def _pool():
    return [
        {"question": f"q{i}", "choices": [f"opt{i}a", f"opt{i}b", "shared", f"opt{i}d"],
         "gold_index": i % 4, "subject": "physics" if i % 2 else "history"}
        for i in range(40)
    ]


def test_near_dup_never_leaks_the_current_answer():
    """§8: zero overlap between any context item's correct option and the current item's."""
    pool = _pool()
    current = pool[0]
    picked = select_competitors(pool, current, arm="near_dup", n=8, seed=42)
    current_answer = current["choices"][current["gold_index"]]
    assert len(picked) == 8
    for item in picked:
        assert item["choices"][item["gold_index"]] != current_answer


def test_same_subject_arm_holds_subject_fixed():
    pool = _pool()
    current = pool[1]
    picked = select_competitors(pool, current, arm="same_subject", n=6, seed=42)
    assert picked and all(p["subject"] == current["subject"] for p in picked)


def test_unrelated_arm_is_not_restricted_to_subject():
    pool = _pool()
    current = pool[1]
    picked = select_competitors(pool, current, arm="unrelated", n=20, seed=42)
    assert len({p["subject"] for p in picked}) > 1


def test_competitor_selection_is_seeded_and_reproducible():
    pool = _pool()
    a = select_competitors(pool, pool[0], arm="unrelated", n=10, seed=7)
    b = select_competitors(pool, pool[0], arm="unrelated", n=10, seed=7)
    c = select_competitors(pool, pool[0], arm="unrelated", n=10, seed=8)
    assert [x["question"] for x in a] == [x["question"] for x in b]
    assert [x["question"] for x in a] != [x["question"] for x in c]


def test_current_item_is_never_its_own_competitor():
    pool = _pool()
    current = pool[3]
    for arm in ("unrelated", "same_subject", "near_dup"):
        picked = select_competitors(pool, current, arm=arm, n=10, seed=1)
        assert all(p["question"] != current["question"] for p in picked)


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="arm"):
        select_competitors(_pool(), _pool()[0], arm="teleportation", n=4, seed=1)


def test_arm_spec_enumerates_the_briefs_arms():
    assert ArmSpec.distance_arms() == ["local", "back_2", "back_5", "back_10", "back_20", "split"]
    assert ArmSpec.competition_arms() == ["unrelated", "same_subject", "near_dup"]


@pytest.mark.parametrize("depths", [(4, 12), (12, 4), (0, 3, 9), (9, 3, 0), (1, 2, 3, 20)])
def test_multi_block_depths_hold_regardless_of_argument_order(depths):
    """Regression: inserting a shallower block must not push a deeper one further back.

    Placing deepest-first looks natural and is wrong — each later, shallower insertion adds a user
    turn between the question and every block already placed, so a requested (4, 12) came out as
    (4, 13). The depths must be exactly what was asked for, in whatever order they arrive.
    """
    prior = _prior_turns(40)
    blocks = [f"BLOCK-{d}" for d in depths]
    out = assemble_transcript(prior, evidence=blocks, question="Q", distance=depths, ack=ACK)

    got = {out.turns[i]["content"]: _depth(out, i) for i in out.evidence_turn_indices}
    assert got == {f"BLOCK-{d}": d for d in depths}
