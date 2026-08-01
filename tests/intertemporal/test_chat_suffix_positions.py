"""The turn window must be labelled for every chat family, not just Qwen.

A geometry run scoped with --turn-only extracts only chat_suffix and
chat_suffix_tail. If the position mapper fails to label them the run finishes
clean and writes nothing, which costs a full GPU rental to discover. Gemma-2 is
the case that breaks a hardcoded token list: it ends a turn with <end_of_turn>
and names the assistant "model".

These tests load real tokenizers and are marked `slow`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.intertemporal.common.chat_template_boundaries import get_chat_boundaries
from src.intertemporal.common.sample_position_mapping import (
    _build_named_positions,
    _find_chat_suffix_start,
)
from src.intertemporal.common.semantic_positions import TURN_POSITIONS
from src.intertemporal.prompt import PromptDatasetConfig
from src.intertemporal.prompt.prompt_dataset_generator import PromptDatasetGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# One model per turn convention, paired with the domain we run it on.
MODEL_DOMAINS = [
    ("Qwen/Qwen3-4B-Instruct-2507", "investment"),
    ("meta-llama/Llama-3.1-8B-Instruct", "health"),
    ("google/gemma-2-9b-it", "climate"),
]


def _tokenizer(name):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # noqa: BLE001 - a missing gated token is a skip
        pytest.skip(f"tokenizer {name} unavailable: {exc}")


def _first_sample(domain: str):
    cfg_path = PROJECT_ROOT / "data" / "intertemporal" / domain / f"{domain}_geometry.json"
    config = PromptDatasetConfig.from_dict(json.loads(cfg_path.read_text()))
    return PromptDatasetGenerator(config).generate().samples[0]


def _templated_tokens(tok, sample):
    body = sample.text
    templated = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
    ids = tok.encode(templated, add_special_tokens=False)
    return [tok.decode([i]) for i in ids]


@pytest.mark.slow
@pytest.mark.parametrize("model,domain", MODEL_DOMAINS)
def test_turn_window_covers_the_whole_template_suffix(model, domain):
    """chat_suffix + chat_suffix_tail must hold every generation-suffix token."""
    tok = _tokenizer(model)
    sample = _first_sample(domain)
    decoded = _templated_tokens(tok, sample)
    prompt_len = len(decoded)

    named = _build_named_positions(sample, None, decoded, prompt_len, prompt_len, tok)

    window = named.get("chat_suffix", []) + named.get("chat_suffix_tail", [])
    suffix = get_chat_boundaries(tok).suffix
    n_suffix = len(tok.encode(suffix, add_special_tokens=False))

    assert len(window) == n_suffix
    assert "".join(decoded[p] for p in sorted(window)) == suffix


@pytest.mark.slow
@pytest.mark.parametrize("model,domain", MODEL_DOMAINS)
def test_turn_positions_are_all_populated(model, domain):
    """Every name in TURN_POSITIONS must be non-empty, or --turn-only extracts nothing."""
    tok = _tokenizer(model)
    sample = _first_sample(domain)
    decoded = _templated_tokens(tok, sample)
    prompt_len = len(decoded)

    named = _build_named_positions(sample, None, decoded, prompt_len, prompt_len, tok)

    for position in TURN_POSITIONS:
        assert named.get(position), f"{position} is empty for {model}"


@pytest.mark.slow
@pytest.mark.parametrize("model,domain", MODEL_DOMAINS)
def test_suffix_tail_is_the_last_prompt_token(model, domain):
    tok = _tokenizer(model)
    sample = _first_sample(domain)
    decoded = _templated_tokens(tok, sample)
    prompt_len = len(decoded)

    named = _build_named_positions(sample, None, decoded, prompt_len, prompt_len, tok)

    assert named["chat_suffix_tail"] == [prompt_len - 1]


def test_suffix_start_is_none_without_a_chat_template():
    """A tokenizer with no template has no turn window, and must say so."""

    class NoTemplate:
        chat_template = None

    assert _find_chat_suffix_start(NoTemplate(), ["a", "b"]) is None
    assert _find_chat_suffix_start(None, ["a", "b"]) is None
