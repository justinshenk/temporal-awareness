"""Tests for the MetaMath prompt format and GSM8K answer scoring (CPU, fast)."""

from __future__ import annotations

import pytest

from src.probes.attribution.gsm8k_prompts import (
    extract_pred_number,
    format_gsm8k_solution,
    gsm8k_gold_answer,
    metamath_direct_prompt,
    metamath_fewshot_prompt,
    metamath_prompt,
    numeric_match,
)


def test_metamath_direct_prompt():
    p = metamath_direct_prompt("What is 23 times 47?")
    assert "### Instruction:\nWhat is 23 times 47?" in p
    assert p.endswith("### Response: The answer is: ")   # no CoT; answer comes next
    assert "Let's think step by step" not in p
    for tok in ("<|im_start|>", "[INST]", "<<SYS>>", "<|user|>"):
        assert tok not in p


def test_format_gsm8k_solution():
    sol = format_gsm8k_solution("She has 16-3 = <<16-3=13>>13 left.\n#### 13")
    assert sol == "She has 16-3 = <<16-3=13>>13 left.\nThe answer is: 13"


def test_metamath_fewshot_prompt_structure():
    shots = [("Q1?", "work one\n#### 5"), ("Q2?", "work two\n#### 9")]
    p = metamath_fewshot_prompt("TargetQ?", shots)
    assert p.count("### Instruction:") == 3                 # 2 shots + target
    assert "The answer is: 5" in p and "The answer is: 9" in p
    assert "### Instruction:\nTargetQ?" in p
    assert p.endswith("### Response: Let's think step by step.")  # target left open
    # the target's own answer is NOT leaked into the prompt
    assert p.rstrip().count("The answer is:") == 2


def test_metamath_prompt_format():
    p = metamath_prompt("What is 2+2?")
    assert "### Instruction:\nWhat is 2+2?" in p
    assert p.endswith("### Response: Let's think step by step.")
    # plain Alpaca text — NOT a chat template
    for tok in ("<|im_start|>", "[INST]", "<<SYS>>", "<|user|>"):
        assert tok not in p


def test_gsm8k_gold_extraction():
    assert gsm8k_gold_answer("Janet sells ...\n#### 18") == 18.0
    assert gsm8k_gold_answer("steps ...\n#### 1,000") == 1000.0
    assert gsm8k_gold_answer("...\n#### -5") == -5.0


@pytest.mark.parametrize("text,expected", [
    ("First ... The answer is: 42", 42.0),
    ("The answer is: 1,000", 1000.0),
    ("The answer is: -4.5", -4.5),
    ("The answer is: 3/4", 0.75),
    ("The answer is: 18.", 18.0),
    ("no answer marker here at all", None),
    ("The answer is: not-a-number", None),
])
def test_extract_pred_number(text, expected):
    assert extract_pred_number(text) == expected


def test_numeric_match():
    assert numeric_match(18.0, 18.0)
    assert numeric_match(18.0, 18)           # gold int-valued
    assert numeric_match(0.75, 0.75)
    assert not numeric_match(17.0, 18.0)
    assert not numeric_match(None, 18.0)
