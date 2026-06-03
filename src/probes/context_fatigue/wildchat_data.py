"""WildChat-1M conversation handling for entropy/attention depth dynamics.

Pure, model-free helpers (streaming filter, turn normalization, attention-segment
math) so the experiment's bookkeeping is unit-testable offline; the runner adds the
model forward pass and the actual token offsets on top.

The attention decomposition mirrors the DDXPlus context-rot analysis but without a
system prompt: at each user->assistant boundary the last-query-token attention is
split into

    first   — the opening tokens (BOS / chat-template prefix; the attention sink),
    middle  — the bulk of accumulated earlier turns,
    recent  — the immediately preceding user+assistant pair (recency),
    current — the current user turn being responded to.
"""

from __future__ import annotations


def normalize_conversation(conv: list[dict]) -> list[dict]:
    """Strip to ``{role, content}``, keep only user/assistant, start on a user turn."""
    msgs = [{"role": m["role"], "content": m["content"]}
            for m in conv if m.get("role") in ("user", "assistant")]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


def count_user_turns(msgs: list[dict]) -> int:
    return sum(1 for m in msgs if m["role"] == "user")


def assistant_boundary_indices(msgs: list[dict]) -> list[int]:
    """Indices ``i`` where ``msgs[i]`` is an assistant turn answering ``msgs[i-1]`` (a user turn).

    These are the points at which the model would generate — where we probe entropy
    and capture attention (using ``msgs[:i]`` as the prompt).
    """
    return [i for i in range(1, len(msgs))
            if msgs[i]["role"] == "assistant" and msgs[i - 1]["role"] == "user"]


def passes_language_turn_filter(row: dict, min_turns: int, language: str) -> bool:
    """Stream filter: right language and at least ``min_turns`` user turns."""
    return row.get("language") == language and (row.get("turn") or 0) >= min_turns


def attention_segments(seq_len: int, first_end: int, recent_start: int,
                       current_start: int) -> dict[str, tuple[int, int]]:
    """Half-open key-position ranges for the four attention buckets.

    Offsets must satisfy ``0 <= first_end <= recent_start <= current_start <= seq_len``;
    they are clamped to that order so degenerate early turns (no real "middle"/"recent")
    just yield empty ranges rather than overlaps.
    """
    first_end = max(0, min(first_end, seq_len))
    current_start = max(0, min(current_start, seq_len))
    recent_start = max(first_end, min(recent_start, current_start))
    return {
        "first": (0, first_end),
        "middle": (first_end, recent_start),
        "recent": (recent_start, current_start),
        "current": (current_start, seq_len),
    }


def segment_fractions(attn_vec, segments: dict[str, tuple[int, int]]) -> dict[str, float]:
    """Fraction of last-token attention mass landing in each segment.

    ``attn_vec`` is a 1-D sequence (list / numpy / tensor slice-able) of per-key-position
    attention weights for one head or a head-mean, summing to ~1 over ``[0, seq_len)``.
    """
    return {f"frac_{name}": float(sum(attn_vec[lo:hi]))
            for name, (lo, hi) in segments.items()}
