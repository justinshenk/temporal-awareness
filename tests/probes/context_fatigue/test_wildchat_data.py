"""Tests for WildChat conversation handling and attention-segment math (GPU-free)."""

import pytest

from src.probes.context_fatigue.wildchat_data import (
    assistant_boundary_indices,
    attention_segments,
    count_user_turns,
    normalize_conversation,
    passes_language_turn_filter,
    segment_fractions,
)


def _conv(n_pairs):
    out = []
    for i in range(n_pairs):
        out += [{"role": "user", "content": f"u{i}", "extra": 1},
                {"role": "assistant", "content": f"a{i}"}]
    return out


def test_normalize_drops_extra_fields_and_nonchat_roles():
    raw = [{"role": "system", "content": "s"},
           {"role": "user", "content": "u", "lang": "en"},
           {"role": "assistant", "content": "a"}]
    msgs = normalize_conversation(raw)
    assert msgs == [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]


def test_normalize_starts_on_user():
    raw = [{"role": "assistant", "content": "a0"}, {"role": "user", "content": "u0"},
           {"role": "assistant", "content": "a1"}]
    assert normalize_conversation(raw)[0]["role"] == "user"


def test_boundary_indices_are_assistant_answers_to_users():
    msgs = _conv(3)  # u0 a0 u1 a1 u2 a2 -> boundaries at 1,3,5
    assert assistant_boundary_indices(msgs) == [1, 3, 5]
    assert count_user_turns(msgs) == 3


def test_filter_requires_language_and_min_turns():
    assert passes_language_turn_filter({"language": "English", "turn": 12}, 10, "English")
    assert not passes_language_turn_filter({"language": "English", "turn": 4}, 10, "English")
    assert not passes_language_turn_filter({"language": "French", "turn": 12}, 10, "English")
    assert not passes_language_turn_filter({"turn": None}, 1, "English")


def test_segments_partition_the_sequence():
    seg = attention_segments(seq_len=100, first_end=10, recent_start=60, current_start=80)
    assert seg == {"first": (0, 10), "middle": (10, 60), "recent": (60, 80), "current": (80, 100)}
    # contiguous, non-overlapping, covering [0, seq_len)
    flat = [seg[k] for k in ("first", "middle", "recent", "current")]
    assert flat[0][0] == 0 and flat[-1][1] == 100
    assert all(flat[i][1] == flat[i + 1][0] for i in range(3))


def test_segments_clamp_degenerate_early_turn():
    # turn-0-ish: recent_start past current_start should clamp, not invert
    seg = attention_segments(seq_len=20, first_end=8, recent_start=50, current_start=15)
    assert seg["middle"] == (8, 15) and seg["recent"] == (15, 15)  # empty recent
    assert seg["current"] == (15, 20)


def test_segment_fractions_sum_to_one_for_normalized_attention():
    attn = [0.25] * 4  # uniform over 4 positions
    seg = attention_segments(seq_len=4, first_end=1, recent_start=2, current_start=3)
    fr = segment_fractions(attn, seg)
    assert fr["frac_first"] == pytest.approx(0.25)
    assert fr["frac_current"] == pytest.approx(0.25)
    assert sum(fr.values()) == pytest.approx(1.0)


def test_segment_fractions_locate_mass():
    # all mass on the last (current) position
    attn = [0.0, 0.0, 0.0, 1.0]
    seg = attention_segments(seq_len=4, first_end=1, recent_start=2, current_start=3)
    fr = segment_fractions(attn, seg)
    assert fr["frac_current"] == pytest.approx(1.0)
    assert fr["frac_first"] == pytest.approx(0.0)
