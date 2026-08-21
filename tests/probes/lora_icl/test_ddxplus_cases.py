"""Tests for DDXPlus case/ICL-context assembly."""

from src.probes.lora_icl.ddxplus_cases import MCQCase, icl_messages


class _BatchEncodingLike(dict):
    """Qwen-style apply_chat_template return: a mapping, whose len() counts keys."""


class _FakeTokenizer:
    """Ten tokens per message; returns a BatchEncoding-like mapping, not a list."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        ids = list(range(10 * len(messages)))
        return _BatchEncodingLike(input_ids=ids, attention_mask=[1] * len(ids))


def test_icl_budget_counts_tokens_not_mapping_keys():
    """A tokenizer returning a BatchEncoding must not defeat the fill budget.

    len(BatchEncoding) is the number of KEYS (2), which is always under budget, so every
    filler would be appended and a 3.5k-token context silently becomes 68k tokens.
    """
    fillers = [MCQCase(source_index=i, prompt_text=f"case {i}", gold_letter="A",
                       option_names=[]) for i in range(50)]
    final = [{"role": "user", "content": "probe"}]
    # budget of 200 tokens at 10/message => at most ~20 messages (10 filler Q&A pairs)
    msgs = icl_messages(_FakeTokenizer(), fillers, final, max_ctx=200, fill_target=1.0)
    assert len(msgs) < 25, f"budget ignored: {len(msgs)} messages"
