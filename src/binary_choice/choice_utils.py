"""Binary choice runner for preference experiments.

Extends ModelRunner with specialized binary choice methods.
"""

from __future__ import annotations
import bisect
import re
from typing import Any


def encode_into_trajectory_ids(
    runner: Any, prompt: str, response_text: str, debug: bool = False
) -> list[int]:
    """Encode prompt + response into token IDs with correct BOS handling.

    Encodes the full concatenated string (not separately) to preserve
    BPE merges at the prompt-response boundary. Resolves BOS ambiguity
    since some chat templates embed it in text, others rely on the tokenizer.
    """
    formatted_prompt = runner._apply_chat_template(prompt)
    full_text = formatted_prompt + response_text

    # Encode with and without special tokens to detect BOS handling
    trajectory_token_ids = runner.tokenizer.encode(full_text, add_special_tokens=True)
    ids_without = runner.tokenizer.encode(full_text, add_special_tokens=False)

    encoding_matches = trajectory_token_ids == ids_without

    if not encoding_matches or debug:
        encode_debug(
            runner,
            formatted_prompt,
            response_text,
            trajectory_token_ids,
            ids_without,
        )

    if encoding_matches:
        return trajectory_token_ids

    # Encodings differ — template may have already embedded BOS.
    # If so, use ids_without to avoid double-BOS.
    template_already_has_bos = (
        runner.tokenizer.bos_token_id is not None
        and ids_without[0] == runner.tokenizer.bos_token_id
    )
    if template_already_has_bos:
        return ids_without

    return trajectory_token_ids


def parse_choice_from_generated_response(
    response: str,
    short_label: str,
    long_label: str,
    choice_prefix: str,
) -> str:
    """
    Parse choice from model response.

    Looks for pattern: "<choice_prefix> <label>"
    Returns: 0, 1 or -1
    """
    response_lower = response.lower().strip()
    prefix_lower = choice_prefix.lower()

    labels = [short_label, long_label]
    labels_stripped = [label.rstrip(".)") for label in labels]
    all_variants = set(label.lower() for label in labels + labels_stripped)
    labels_pattern = "|".join(
        re.escape(label) for label in sorted(all_variants, key=len, reverse=True)
    )

    pattern = rf"{re.escape(prefix_lower)}\s*({labels_pattern})"
    match = re.search(pattern, response_lower)

    if match:
        matched = match.group(1)
        if matched in (short_label.lower(), short_label.rstrip(".)").lower()):
            return 1
        elif matched in (long_label.lower(), long_label.rstrip(".)").lower()):
            return 0

    return -1


def get_label_start_end_pos(
    runner: Any,
    token_ids: list[int],
    choice_prefix: str,
    label: str,
) -> tuple[int, int]:
    """Find token position range [start, end) of label in token sequence.

    Builds a char→token map via incremental decoding, then binary searches
    for the label's character span. Works correctly across BPE boundaries.

    Example:
        token_ids encodes "...prompt...I select: a)"
        choice_prefix = "I select: ", label = "a)"
        → returns token positions spanning "a)"
    """
    # Cumulative character count after decoding tokens [0..i]
    char_ends = [len(runner.decode(token_ids[: i + 1])) for i in range(len(token_ids))]

    # Find label in the fully decoded text (rfind to skip any prompt echo)
    full_text = runner.decode(token_ids)
    target = choice_prefix + label
    target_pos = full_text.rfind(target)
    if target_pos == -1:
        raise ValueError(f"{target!r} not found in decoded text")

    # Character span of just the label, after the prefix
    label_char_start = target_pos + len(choice_prefix)
    label_char_end = label_char_start + len(label)

    # Map character span → token span via binary search
    start = bisect.bisect_right(char_ends, label_char_start)
    end = bisect.bisect_left(char_ends, label_char_end) + 1

    return start, end


def get_divergent_token_id_position(ids1: list[int], ids2: list[int]) -> int:
    """Find first position where two token ID lists diverge."""
    for i, (a, b) in enumerate(zip(ids1, ids2)):
        if a != b:
            return i
    return min(len(ids1), len(ids2))


def encode_debug(
    runner, formatted_prompt, response_text, response_text_token_ids, ids_without
) -> None:
    """Debug encoding by comparing three strategies and printing diagnostics.

    Strategies compared:
      1. ids_with:     encode(full_text, add_special_tokens=True)   — default
      2. ids_without:  encode(full_text, add_special_tokens=False)  — no auto-BOS
      3. ids_isolated:  encode(prompt) + encode(response)            — split encoding

    If all three match, encoding is unambiguous. If they differ, this helps
    identify whether the issue is BOS duplication or boundary token merging.
    """
    # --- Encode prompt and response separately for comparison ---
    isolated_prompt_ids = runner.tokenizer.encode(
        formatted_prompt, add_special_tokens=True
    )
    isolated_response_ids = runner.tokenizer.encode(
        response_text, add_special_tokens=False
    )
    ids_isolated = isolated_prompt_ids + isolated_response_ids

    # --- Equality checks ---
    is_with_without_equal = response_text_token_ids == ids_without
    is_with_isolated_equal = response_text_token_ids == ids_isolated
    is_without_isolated_equal = ids_without == ids_isolated

    # --- BOS token inspection ---
    bos_id = runner.tokenizer.bos_token_id
    bos_token = runner.tokenizer.bos_token
    has_bos = bos_id is not None

    with_starts_with_bos = (
        has_bos
        and len(response_text_token_ids) > 0
        and response_text_token_ids[0] == bos_id
    )
    without_starts_with_bos = (
        has_bos and len(ids_without) > 0 and ids_without[0] == bos_id
    )
    isolated_starts_with_bos = (
        has_bos and len(ids_isolated) > 0 and ids_isolated[0] == bos_id
    )

    # Double BOS = both the template text and add_special_tokens added one
    has_double_bos = (
        has_bos
        and len(response_text_token_ids) >= 2
        and response_text_token_ids[0] == bos_id
        and response_text_token_ids[1] == bos_id
    )

    # --- Boundary token merge check ---
    # If isolated != without, tokens merged differently at the prompt-response boundary
    has_boundary_merge_issue = not is_without_isolated_equal

    # --- Decoded text for visual inspection ---
    text_with = runner.tokenizer.decode(response_text_token_ids)
    text_without = runner.tokenizer.decode(ids_without)
    text_isolated = runner.tokenizer.decode(ids_isolated)

    # --- Length comparison ---
    len_with = len(response_text_token_ids)
    len_without = len(ids_without)
    len_isolated = len(ids_isolated)

    # --- Print everything ---
    print("\n" + "=" * 60)
    print("_encode: DEBUG")
    print("=" * 60)

    print("\n--- Model/Tokenizer ---")
    print(f"  BOS token: {bos_token!r} (id={bos_id})")
    print(
        f"  EOS token: {runner.tokenizer.eos_token!r} (id={runner.tokenizer.eos_token_id})"
    )

    print("\n--- Input ---")
    print(f"  formatted_prompt length: {len(formatted_prompt)} chars")
    print(f"  response_text: {response_text!r}")

    print("\n--- Equality checks ---")
    print(f"  ids_with == ids_without:  {is_with_without_equal}")
    print(f"  ids_with == ids_isolated: {is_with_isolated_equal}")
    print(f"  ids_without == ids_isolated: {is_without_isolated_equal}")

    print("\n--- BOS analysis ---")
    print(f"  ids_with starts with BOS:     {with_starts_with_bos}")
    print(f"  ids_without starts with BOS:  {without_starts_with_bos}")
    print(f"  ids_isolated starts with BOS: {isolated_starts_with_bos}")
    print(f"  Double BOS detected:          {has_double_bos}")

    print("\n--- Boundary merge issue ---")
    print(f"  Split encode differs from joint encode: {has_boundary_merge_issue}")
    if has_boundary_merge_issue:
        # Show exactly where they diverge
        div = get_divergent_token_id_position(ids_without, ids_isolated)
        print(f"  First divergence at position: {div}")
        if div < len(ids_without):
            print(
                f"    joint:    id={ids_without[div]} -> {runner.tokenizer.decode([ids_without[div]])!r}"
            )
        if div < len(ids_isolated):
            print(
                f"    isolated: id={ids_isolated[div]} -> {runner.tokenizer.decode([ids_isolated[div]])!r}"
            )

    print("\n--- Token counts ---")
    print(f"  ids_with:     {len_with} tokens")
    print(f"  ids_without:  {len_without} tokens")
    print(f"  ids_isolated: {len_isolated} tokens")

    print("\n--- First/last 10 token IDs ---")
    for name, ids in [
        ("ids_with", response_text_token_ids),
        ("ids_without", ids_without),
        ("ids_isolated", ids_isolated),
    ]:
        print(f"  {name} first 10: {ids[:10]}")
        print(f"  {name}  last 10: {ids[-10:]}")

    print("\n--- Decoded text (ids_with) ---")
    print(text_with)
    print("\n--- Decoded text (ids_without) ---")
    print(text_without)
    print("\n--- Decoded text (ids_isolated) ---")
    print(text_isolated)

    print("\n" + "=" * 60 + "\n")
