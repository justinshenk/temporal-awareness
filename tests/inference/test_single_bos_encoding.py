"""Templated prompts must reach the model with exactly one BOS.

Mistral/Gemma chat templates embed the BOS string in the templated text.
The HF backend used to encode that text with add_special_tokens=True, which
prepended a second BOS id ("<s><s>[INST]"). These tests run the backend's
encode path with real tokenizers (tokenizer-only, no model) and assert a
single BOS at position 0. Qwen's template has no BOS, so the guard must be
a no-op there.
"""

from __future__ import annotations

import pytest

from src.inference.backends.backend_huggingface import encode_without_duplicate_bos

BOS_TEMPLATE_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
]

NO_BOS_TEMPLATE_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
]


@pytest.fixture(scope="module")
def tokenizers():
    from transformers import AutoTokenizer

    out = {}
    for name in BOS_TEMPLATE_MODELS + NO_BOS_TEMPLATE_MODELS:
        try:
            out[name] = AutoTokenizer.from_pretrained(name)
        except Exception as exc:  # noqa: BLE001 - reported as a skip, not a failure
            pytest.skip(f"tokenizer {name} unavailable: {exc}")
    return out


def _templated_prompt(tok) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": "Pick a) or b)."}],
        tokenize=False,
        add_generation_prompt=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize("model", BOS_TEMPLATE_MODELS)
def test_bos_prefixed_template_encodes_single_bos(model, tokenizers):
    tok = tokenizers[model]
    text = _templated_prompt(tok)
    assert text.startswith(tok.bos_token), "template must embed BOS for this test"

    ids = encode_without_duplicate_bos(tok, text, add_special_tokens=True)
    ids = ids.flatten().tolist()

    assert ids[0] == tok.bos_token_id, "first token must be BOS"
    assert ids[1] != tok.bos_token_id, "duplicate BOS must not follow"
    assert ids.count(tok.bos_token_id) == 1, "exactly one BOS in the prompt"


@pytest.mark.slow
@pytest.mark.parametrize("model", NO_BOS_TEMPLATE_MODELS)
def test_no_bos_template_is_unchanged(model, tokenizers):
    tok = tokenizers[model]
    text = _templated_prompt(tok)

    guarded = encode_without_duplicate_bos(tok, text, add_special_tokens=True)
    direct = tok(text, return_tensors="pt", add_special_tokens=True).input_ids

    assert guarded.flatten().tolist() == direct.flatten().tolist()


@pytest.mark.slow
@pytest.mark.parametrize("model", BOS_TEMPLATE_MODELS)
def test_plain_text_still_gets_bos(model, tokenizers):
    """Non-templated text keeps the default add_special_tokens behavior."""
    tok = tokenizers[model]
    ids = encode_without_duplicate_bos(tok, "Pick a) or b).", add_special_tokens=True)
    ids = ids.flatten().tolist()
    assert ids[0] == tok.bos_token_id
    assert ids.count(tok.bos_token_id) == 1
