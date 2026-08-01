"""Chat boundaries must be exact for every model we run, not just Qwen.

These tests load real tokenizers. They are marked `slow` because the first run
may download tokenizer files.
"""

from __future__ import annotations

import pytest

from src.intertemporal.common.chat_template_boundaries import (
    ROLE_ASSISTANT,
    ROLE_TURN_END,
    ROLE_TURN_START,
    get_chat_boundaries,
    resolve_role_positions,
    suffix_special_tokens,
)

# Three families with three different turn conventions. Gemma is the reason the
# resolver may not key on the literal "assistant" or on the <|...|> bracket
# style: it uses "model" and <start_of_turn>.
MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
]

EXPECTED_ROLE_TOKEN = {
    "Qwen/Qwen3-4B-Instruct-2507": "assistant",
    "meta-llama/Llama-3.1-8B-Instruct": "assistant",
    "google/gemma-2-9b-it": "model",
}


def _suffix_tokens(tok):
    bounds = get_chat_boundaries(tok)
    ids = tok.encode(bounds.suffix, add_special_tokens=False)
    return [tok.decode([i]) for i in ids]


@pytest.fixture(scope="module")
def tokenizers():
    from transformers import AutoTokenizer

    out = {}
    for name in MODELS:
        try:
            out[name] = AutoTokenizer.from_pretrained(name)
        except Exception as exc:  # noqa: BLE001 - reported as a skip, not a failure
            pytest.skip(f"tokenizer {name} unavailable: {exc}")
    return out


@pytest.mark.slow
@pytest.mark.parametrize("model", MODELS)
def test_boundaries_reconstruct_the_template(model, tokenizers):
    """prefix + body + suffix must equal what the template actually produces."""
    tok = tokenizers[model]
    body = "SITUATION: Plan for the future.\nTASK: choose."

    bounds = get_chat_boundaries(tok)
    expected = tok.apply_chat_template(
        [{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True
    )

    # Llama's template applies `| trim` to content, so compare against the
    # trimmed body — the boundaries themselves must still be exact.
    assert bounds.prefix + body.strip() + bounds.suffix == expected


@pytest.mark.slow
@pytest.mark.parametrize("model", MODELS)
def test_suffix_is_nonempty_and_tokenizes(model, tokenizers):
    tok = tokenizers[model]
    bounds = get_chat_boundaries(tok)
    assert bounds.has_suffix
    assert len(tok.encode(bounds.suffix, add_special_tokens=False)) >= 3


@pytest.mark.slow
@pytest.mark.parametrize("model", MODELS)
def test_role_aliases_resolve(model, tokenizers):
    """Every model must expose the assistant turn, wherever it sits."""
    tok = tokenizers[model]
    decoded = _suffix_tokens(tok)

    roles = resolve_role_positions(decoded, suffix_special_tokens(tok))
    assert ROLE_ASSISTANT in roles, f"no role token found in {decoded}"
    assert ROLE_TURN_END in roles
    assert ROLE_TURN_START in roles
    assert roles[ROLE_TURN_END] < roles[ROLE_ASSISTANT]
    assert decoded[roles[ROLE_ASSISTANT]].strip() == EXPECTED_ROLE_TOKEN[model]


@pytest.mark.slow
@pytest.mark.parametrize("model", MODELS)
def test_turn_positions_are_exactly_the_geometry_targets(model, tokenizers):
    """The change-of-turn window is what the geometry run extracts.

    It must be a small, contiguous, fully-resolved set for every family, since
    the whole disk budget depends on extracting only these positions.
    """
    tok = tokenizers[model]
    decoded = _suffix_tokens(tok)
    roles = resolve_role_positions(decoded, suffix_special_tokens(tok))

    assert 3 <= len(decoded) <= 6, f"unexpected suffix width {len(decoded)}: {decoded}"
    ordered = sorted(roles.values())
    assert ordered == sorted(set(ordered)), "role aliases collided on one token"
    assert max(ordered) < len(decoded)


@pytest.mark.slow
def test_assistant_index_actually_differs_across_families(tokenizers):
    """Guards the reason role aliases exist at all.

    If this ever stops holding, a raw rel_pos comparison would be safe and the
    alias layer could be simplified. Until then, it is load-bearing.
    """
    idx = {}
    for model in MODELS:
        tok = tokenizers[model]
        bounds = get_chat_boundaries(tok)
        ids = tok.encode(bounds.suffix, add_special_tokens=False)
        decoded = [tok.decode([i]) for i in ids]
        idx[model] = resolve_role_positions(decoded)[ROLE_ASSISTANT]

    assert len(set(idx.values())) > 1, (
        f"expected the assistant token to sit at different offsets per family, got {idx}"
    )


def test_missing_template_is_an_error():
    class NoTemplate:
        chat_template = None

    with pytest.raises(ValueError, match="no chat_template"):
        get_chat_boundaries(NoTemplate())
