"""Binary choice runner for preference experiments.

Extends ModelRunner with specialized binary choice methods.
"""

from __future__ import annotations
import bisect
import re
from typing import Any, Sequence


def encode_into_trajectory_ids(
    runner: Any, prompt: str, response_text: str, debug: bool = False
) -> list[int]:
    """Encode prompt + response into token IDs with correct BOS handling.

    Encodes the full concatenated string (not separately) to preserve
    BPE merges at the prompt-response boundary. Resolves BOS ambiguity
    since some chat templates embed it in text, others rely on the tokenizer.
    """
    full_text = prompt + response_text

    # Encode with and without special tokens to detect BOS handling
    trajectory_token_ids = runner.encode_ids(full_text, add_special_tokens=True)
    ids_without = runner.encode_ids(full_text, add_special_tokens=False)

    encoding_matches = trajectory_token_ids == ids_without

    if not encoding_matches or debug:
        encode_debug(
            runner,
            prompt,
            response_text,
            trajectory_token_ids,
            ids_without,
        )

    if encoding_matches:
        return trajectory_token_ids

    # Encodings differ — template may have already embedded BOS.
    # If so, use ids_without to avoid double-BOS.
    template_already_has_bos = (
        runner.bos_token_id is not None and ids_without[0] == runner.bos_token_id
    )
    if template_already_has_bos:
        return ids_without

    return trajectory_token_ids


def parse_choice_from_generated_response(
    response: str,
    short_label: str,
    long_label: str,
    choice_prefix: str,
) -> int:
    """Parse choice from model response.

    Looks for pattern: "<choice_prefix> <label>"
    Returns: 0 (short_label), 1 (long_label), or -1 (not found)
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
        # short_label is labels[0], long_label is labels[1]
        if matched in (short_label.lower(), short_label.rstrip(".)").lower()):
            return 0
        elif matched in (long_label.lower(), long_label.rstrip(".)").lower()):
            return 1

    return -1


def verify_greedy_generation(
    choice: Any,
    generated_response: str,
    short_label: str,
    long_label: str,
    choice_prefix: str,
    runner: Any = None,
    prompt: str = None,
) -> bool:
    """Verify that greedy generation matches probability-based choice.

    Returns True if there's a mismatch (decoding_mismatch=True).

    Checks (in order):
    1. Token IDs: Compare all encoded tokens between generated and chosen trajectory
    2. Label: Parse label from generated text and compare to choice.choice_idx

    Args:
        choice: SimpleBinaryChoice or LabeledSimpleBinaryChoice
        generated_response: The freely generated response text
        short_label: Label for option 0 (e.g., "a)")
        long_label: Label for option 1 (e.g., "b)")
        choice_prefix: Prefix before label (e.g., "I select:")
        runner: Optional ModelRunner for token-level comparison
        prompt: Optional prompt text for token-level comparison
    """
    labels = (short_label, long_label)
    chosen_traj = getattr(choice, "chosen_traj", None)

    # Only skip_thinking_prefix is used as prefilling in generate()
    # The model must generate choice_prefix itself as part of following the format
    prefilling = runner.skip_thinking_prefix if runner else ""

    # Check 1: Token ID comparison (primary check)
    # Compare generated tokens to chosen trajectory tokens
    if runner and prompt and chosen_traj:
        # Apply chat template to match how choose() encodes the trajectory
        templated_prompt = runner.apply_chat_template(prompt)
        # Prepend only what was actually prefilled (skip_thinking_prefix)
        # The model should have generated choice_prefix + label naturally
        full_response = prefilling + generated_response
        # Encode generated response
        generated_ids = encode_into_trajectory_ids(
            runner, templated_prompt, full_response
        )
        expected_ids = chosen_traj.token_ids

        # Find divergent position from the choice (where A vs B differs)
        div_pos = getattr(choice, "divergent_position", None)

        # Find first difference in token sequences
        first_diff_pos = get_divergent_token_id_position(generated_ids, expected_ids)
        min_len = min(len(generated_ids), len(expected_ids))

        # Only flag mismatch if divergence is AT or BEFORE the choice position
        # Post-choice differences are expected (generated text continues beyond label)
        if (
            div_pos is not None
            and first_diff_pos <= div_pos
            and first_diff_pos < min_len
        ):
            expected_token_id = expected_ids[first_diff_pos]
            actual_token_id = generated_ids[first_diff_pos]
            expected_token = runner.decode_ids([expected_token_id])
            actual_token = runner.decode_ids([actual_token_id])

            # Determine mismatch type
            if first_diff_pos < div_pos:
                # Divergence BEFORE the choice point (e.g., inserted \n)
                actual_is_whitespace = actual_token.strip() == ""
                if actual_is_whitespace:
                    mismatch_type = f"Whitespace insertion ({actual_token!r})"
                else:
                    mismatch_type = "Pre-choice token insertion"
            else:
                mismatch_type = "Choice token mismatch"

            # Show context around divergence
            context_start = max(0, first_diff_pos - 3)
            context_end = min(min_len, first_diff_pos + 4)
            expected_context = runner.decode(expected_ids[context_start:context_end])
            actual_context = runner.decode(generated_ids[context_start:context_end])

            print(
                f"\n{'=' * 60}\n"
                f"DECODING MISMATCH: {mismatch_type}\n"
                f"{'=' * 60}\n"
                f"  Divergence position: {first_diff_pos}"
                + (f" (choice at {div_pos})" if div_pos else "")
                + "\n"
                f"  Expected token: {expected_token_id} ({expected_token!r})\n"
                f"  Actual token:   {actual_token_id} ({actual_token!r})\n"
                f"\n"
                f"  Expected: {expected_context!r}\n"
                f"  Actual:   {actual_context!r}\n"
                f"\n"
                f"  Choice idx: {choice.choice_idx} ({labels[choice.choice_idx]})\n"
                f"  Sequence lengths: generated={len(generated_ids)}, expected={len(expected_ids)}\n"
                f"{'=' * 60}\n"
            )
            return True

        # Tokens match up to and including choice position - no mismatch
        return False

    # Check 2: Label-level comparison (fallback if no runner/prompt/chosen_traj)
    # Look for choice_prefix + label in the generated text
    generated_choice_idx = parse_choice_from_generated_response(
        generated_response, short_label, long_label, choice_prefix
    )

    # If probability-based choice was tied (-1), don't flag as mismatch if
    # generation picked a valid option (this is expected behavior for 50/50 cases)
    if choice.choice_idx == -1 and generated_choice_idx in (0, 1):
        # Not a mismatch - generation had to pick one when probabilities were tied
        return False

    if generated_choice_idx != choice.choice_idx:
        expected_label = labels[choice.choice_idx] if choice.choice_idx >= 0 else "?"
        actual_label = (
            labels[generated_choice_idx] if generated_choice_idx >= 0 else "?"
        )
        print(
            f"\n{'=' * 60}\n"
            f"DECODING MISMATCH: Label mismatch\n"
            f"{'=' * 60}\n"
            f"  Probability-based choice: {choice.choice_idx} ({expected_label})\n"
            f"  Generated text choice:    {generated_choice_idx} ({actual_label})\n"
            f"  Response preview: {generated_response[:100]}...\n"
            f"{'=' * 60}\n"
        )
        return True

    return False  # No mismatch


def get_label_start_end_pos(
    runner: Any,
    token_ids: Sequence[int],
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


def get_divergent_token_id_position(ids1: Sequence[int], ids2: Sequence[int]) -> int:
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
    isolated_prompt_ids = runner.encode_ids(formatted_prompt, add_special_tokens=True)
    isolated_response_ids = runner.encode_ids(response_text, add_special_tokens=False)
    ids_isolated = isolated_prompt_ids + isolated_response_ids

    # --- Equality checks ---
    is_with_without_equal = response_text_token_ids == ids_without
    is_with_isolated_equal = response_text_token_ids == ids_isolated
    is_without_isolated_equal = ids_without == ids_isolated

    # --- BOS token inspection ---
    bos_id = runner.bos_token_id
    bos_token = runner.bos_token
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
    text_with = runner.decode_ids(response_text_token_ids)
    text_without = runner.decode_ids(ids_without)
    text_isolated = runner.decode_ids(ids_isolated)

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
    print(f"  EOS token: {runner.eos_token!r} (id={runner.eos_token_id})")

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
                f"    joint:    id={ids_without[div]} -> {runner.decode_ids([ids_without[div]])!r}"
            )
        if div < len(ids_isolated):
            print(
                f"    isolated: id={ids_isolated[div]} -> {runner.decode_ids([ids_isolated[div]])!r}"
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
