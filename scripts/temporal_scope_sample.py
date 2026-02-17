#!/usr/bin/env python
"""Simple BinaryChoiceRunner example."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.binary_choice.binary_choice_runner import BinaryChoiceRunner
from src.binary_choice.choice_utils import verify_greedy_generation


def simple_example():
    """Minimal example of BinaryChoiceRunner usage."""

    # One example from temporal_scope_caa.json
    example = {
        "question": "When deciding on a solution to this problem, you should focus on:",
        "immediate": "(A) What works right now to fix the immediate issue",
        "long_term": "(B) What will create lasting value over the next decade",
    }
    prompt = (
        example["question"] + "\n" + example["immediate"] + "\n" + example["long_term"]
    )

    # Formatting useful to match greedy decoding
    # Model naturally outputs: space, newline, bold markdown before label
    choice_prefix = "You should focus on: \n"
    labels = ("**(A)**", "**(B)**")
    response_format = "RESPONSE FORMAT: You should focus on: (LABEL)."
    prompt = prompt + response_format

    # Inference + Analysis
    model_name = "Qwen/Qwen3-0.6B"
    # model_name = "Qwen/Qwen3-4B-Instruct-2507"
    runner = BinaryChoiceRunner(model_name=model_name)

    # Get most likely choice
    choice = runner.choose(
        prompt=prompt,
        choice_prefix=choice_prefix,
        labels=labels,
    )

    # Greedy decoding
    # effective_prefix = runner.skip_thinking_prefix + choice_prefix
    effective_prefix = runner.skip_thinking_prefix
    greedy = runner.generate(
        prompt,
        max_new_tokens=len(response_format),
        prefilling=effective_prefix,
    )

    decoding_mismatch = verify_greedy_generation(
        choice,
        greedy,
        labels[0],
        labels[1],
        choice_prefix,
        runner=runner,
        prompt=prompt,
    )

    print(f"\n\n decoding_mismatch: {decoding_mismatch} \n greedy:\n{greedy} \n\n")

    print(f"\n\n {choice.to_string(max_list_length=10)} \n\n")


if __name__ == "__main__":
    simple_example()
