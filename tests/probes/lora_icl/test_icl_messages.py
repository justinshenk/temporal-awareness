"""Tests for the shared ICL-context builder (model-free, fake tokenizer)."""

from src.probes.lora_icl.ddxplus_cases import MCQCase, icl_messages


class _FakeTok:
    """apply_chat_template returns a token list sized by total content length."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        n = sum(len(m["content"]) for m in messages)
        return list(range(n))


def _fillers(k):
    return [MCQCase(source_index=i, prompt_text=f"case-{i}", gold_letter="A") for i in range(k)]


def test_appends_final_turn_and_some_fillers():
    final = [{"role": "user", "content": "HARMFUL"}]
    msgs = icl_messages(_FakeTok(), _fillers(5), final, max_ctx=10000, fill_target=0.85)
    assert msgs[-1] == {"role": "user", "content": "HARMFUL"}
    # filler user turns present (each is a 'user' role before the final)
    assert sum(m["role"] == "user" for m in msgs) >= 2


def test_tiny_budget_yields_only_final():
    final = [{"role": "user", "content": "HARMFUL"}]
    msgs = icl_messages(_FakeTok(), _fillers(5), final, max_ctx=1, fill_target=0.85)
    assert msgs == final
