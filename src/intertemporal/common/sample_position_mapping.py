"""Sample position mapping for token-level analysis.

Maps each absolute token position to semantic position information.

This module contains the domain-specific SamplePositionMapping class with
build methods. The generic base classes are in src.common.position_mapping_base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.common import (
    DatasetPositionMappingBase,
    SamplePositionMappingBase,
    TokenPositionInfo,
)
from ..data.default_datasets import BASE_CONTEXT
from ..formatting.configs.default_prompt_format import DefaultPromptFormat
from ..prompt.prompt_dataset_config import ContextConfig

if TYPE_CHECKING:
    from .preference_types import PreferenceSample, PromptSample


def _assert_no_overlapping_positions(named_positions: dict[str, list[int]]) -> None:
    """Assert that no absolute position appears in multiple named_positions.

    Raises:
        AssertionError: If any position is assigned to multiple names.
    """
    pos_to_names: dict[int, list[str]] = {}
    for name, positions in named_positions.items():
        for pos in positions:
            if pos not in pos_to_names:
                pos_to_names[pos] = []
            pos_to_names[pos].append(name)

    # Find overlaps
    overlaps = {pos: names for pos, names in pos_to_names.items() if len(names) > 1}
    if overlaps:
        overlap_strs = [f"pos {pos}: {names}" for pos, names in sorted(overlaps.items())]
        raise AssertionError(
            f"Overlapping positions in named_positions:\n  " + "\n  ".join(overlap_strs)
        )


@dataclass
class SamplePositionMapping(SamplePositionMappingBase):
    """Position mapping for a single sample with domain-specific build methods.

    Inherits generic position mapping functionality from SamplePositionMappingBase.
    Adds build methods specific to intertemporal preference experiments.
    """

    @classmethod
    def build(
        cls,
        sample: "PromptSample",
        runner: Any,
        pref: "PreferenceSample | None" = None,
    ) -> "SamplePositionMapping":
        """Build position mapping from PromptSample with optional PreferenceSample.

        Uses:
        - PromptSample.prompt for structured data (preference_pair, time_horizon)
        - PreferenceSample.chosen_traj for ACTUAL tokenized trajectory (when available)

        Args:
            sample: The PromptSample with prompt structure info
            runner: Model runner with tokenizer
            pref: Optional PreferenceSample with actual trajectory tokens

        Returns:
            SamplePositionMapping with all positions mapped
        """
        tokenizer = runner._tokenizer

        # Use actual trajectory tokens from PreferenceSample if available
        if pref is not None and pref.chosen_traj is not None:
            token_ids = pref.chosen_traj.token_ids
            full_len = len(token_ids)
            prompt_len = pref.prompt_token_count
        else:
            # Fallback: re-tokenize from text
            full_text = sample.text
            token_ids = tokenizer.encode(full_text)
            full_len = len(token_ids)
            prompt_len = full_len  # No response when re-tokenizing

        decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]

        # Build named positions using BOTH PromptSample (structured) and PreferenceSample (actual tokens)
        named_positions = _build_named_positions(
            sample, pref, decoded_tokens, prompt_len, full_len
        )

        # Build reverse mapping: abs_pos -> (format_pos, rel_pos)
        pos_to_format: dict[int, tuple[str, int]] = {}
        for format_name, abs_positions in named_positions.items():
            for rel_idx, abs_pos in enumerate(abs_positions):
                if abs_pos in pos_to_format:
                    existing_name = pos_to_format[abs_pos][0]
                    if len(format_name) < len(existing_name):
                        pos_to_format[abs_pos] = (format_name, rel_idx)
                else:
                    pos_to_format[abs_pos] = (format_name, rel_idx)

        # Build position list
        positions = []
        for abs_pos in range(full_len):
            decoded = decoded_tokens[abs_pos]
            traj_section = "prompt" if abs_pos < prompt_len else "response"

            if abs_pos in pos_to_format:
                format_pos, rel_pos = pos_to_format[abs_pos]
            else:
                format_pos, rel_pos = None, -1

            positions.append(
                TokenPositionInfo(
                    abs_pos=abs_pos,
                    decoded_token=decoded,
                    traj_section=traj_section,
                    format_pos=format_pos,
                    rel_pos=rel_pos,
                )
            )

        _assert_no_overlapping_positions(named_positions)

        return cls(
            sample_idx=sample.sample_idx,
            prompt_len=prompt_len,
            full_len=full_len,
            positions=positions,
            named_positions=named_positions,
        )


@dataclass
class DatasetPositionMapping(DatasetPositionMappingBase):
    """Position mappings for all samples in a dataset.

    Extends DatasetPositionMappingBase with domain-specific type hints.
    """

    mappings: list[SamplePositionMapping] = field(default_factory=list)

    def add(self, mapping: SamplePositionMapping) -> None:
        """Add a sample mapping."""
        self.mappings.append(mapping)

    def get(self, sample_idx: int) -> SamplePositionMapping | None:
        """Get mapping by sample index."""
        for m in self.mappings:
            if m.sample_idx == sample_idx:
                return m
        return None


# =============================================================================
# Position Building (factored from geometry_data.py)
# =============================================================================


def _find_substring_token_range(
    tokens: list[str], text: str, substring: str, verify: bool = True
) -> list[int]:
    """Find token positions spanning a substring in text.

    Includes all tokens that overlap with the substring. For boundary tokens
    that extend past the substring:
    - INCLUDE if inside part has non-whitespace content (e.g., '.\n' where '.' is inside)
    - EXCLUDE if inside part is pure whitespace AND outside part has invalid chars

    Args:
        tokens: List of decoded tokens
        text: Full text (joined tokens)
        substring: The substring to find
        verify: If True, assert that joined tokens match substring content

    Returns:
        List of token indices that span the substring
    """
    char_idx = text.find(substring)
    if char_idx == -1:
        return []

    char_end = char_idx + len(substring)
    substring_chars = set(substring)  # Characters valid in the substring
    positions = []

    char_count = 0
    for i, tok in enumerate(tokens):
        tok_start = char_count
        tok_end = char_count + len(tok)
        char_count = tok_end

        # Token must have some overlap with substring range
        if tok_end <= char_idx or tok_start >= char_end:
            if tok_end >= char_end:
                break
            continue

        # Check if token extends past substring end
        extends_past = tok_end > char_end

        if extends_past:
            # Get the part of the token that's INSIDE the substring range
            inside_part = tok[: char_end - tok_start]

            # If inside part has non-whitespace content, ALWAYS include
            # This ensures we capture essential chars like '.' at end of substring
            if inside_part.strip():
                positions.append(i)
                break  # Token ends past substring, so we're done

            # Inside part is pure whitespace - check outside part
            outside_part = tok[char_end - tok_start :]
            outside_chars = set(outside_part)
            invalid_chars = outside_chars - substring_chars
            if invalid_chars:
                # Outside part has invalid chars and inside is just whitespace
                # Exclude this token to avoid bleedover into next region
                break

            # Outside part is OK (all chars are in substring), include the whitespace
            positions.append(i)
            break

        positions.append(i)

        if tok_end >= char_end:
            break

    # VERIFY: The joined tokens at these positions must contain the substring content
    # Allow for boundary whitespace differences when boundary tokens are excluded
    if verify and positions:
        joined = "".join(tokens[i] for i in positions)
        # Check multiple conditions for validity:
        # 1. substring in joined: we included boundary tokens (joined is superset)
        # 2. joined in substring: we excluded boundary chars (joined is subset)
        # 3. stripped comparison: boundary whitespace tolerance
        content_match = (
            substring in joined
            or joined in substring
            or substring.strip() in joined
            or substring.strip() == joined.strip()
        )
        assert content_match, (
            f"Mapping verification failed!\n"
            f"  Expected substring: {repr(substring)}\n"
            f"  Joined tokens: {repr(joined)}\n"
            f"  Positions: {positions}"
        )

    return positions


def _find_time_value_positions(
    tokens: list[str], text: str, time_val
) -> list[int]:
    """Find token positions for a TimeValue's formatted string.

    Searches for the complete formatted time (e.g., "  1 month  ") as a unit,
    including any padding from format_time_value().
    """
    formatted = str(time_val)  # e.g., "  1 month  " with padding
    return _find_substring_token_range(tokens, text, formatted)


def _find_reward_value_positions(
    tokens: list[str], text: str, reward_val
) -> list[int]:
    """Find token positions for a RewardValue's formatted string.

    Searches for the complete formatted reward (e.g., "1,000 dollars") as a unit,
    not split into parts, to avoid matching partial strings elsewhere.
    """
    # Format as "value unit" (e.g., "1,000 dollars")
    formatted = str(reward_val)  # Uses RewardValue.__str__
    return _find_substring_token_range(tokens, text, formatted)


def _build_named_positions(
    sample: "PromptSample",
    pref: "PreferenceSample | None",
    decoded_tokens: list[str],
    prompt_len: int,
    full_len: int,
) -> dict[str, list[int]]:
    """Build named positions dictionary using BOTH PromptSample and PreferenceSample.

    Uses:
    - PromptSample.prompt for structured data (preference_pair with TimeValue/RewardValue)
    - PreferenceSample for scalar values and labels as they appear in text
    """
    prompt_tokens_decoded = decoded_tokens[:prompt_len]
    prompt_text = "".join(prompt_tokens_decoded)
    full_text = "".join(decoded_tokens)

    pair = sample.prompt.preference_pair
    fmt = DefaultPromptFormat()
    named_positions: dict[str, list[int]] = {}

    # === Prompt Markers ===
    markers = {
        "situation_marker": fmt.prompt_const_keywords.get("situation_marker", "SITUATION:"),
        "task_marker": fmt.prompt_const_keywords.get("task_marker", "TASK:"),
        "consider_marker": fmt.prompt_const_keywords.get("consider_marker", "CONSIDER:"),
        "constraint_marker": fmt.prompt_const_keywords.get("constraint_marker", "CONSTRAINT:"),
        "action_marker": fmt.prompt_const_keywords.get("action_marker", "ACTION:"),
        "format_marker": fmt.prompt_const_keywords.get("format_marker", "FORMAT:"),
        "format_choice_prefix": fmt.prompt_const_keywords.get("format_choice_prefix", "I choose:"),
        "format_reasoning_prefix": fmt.prompt_const_keywords.get("format_reasoning_prefix", "My reasoning:"),
    }

    marker_positions_list = []
    for name, marker_text in markers.items():
        char_pos = prompt_text.find(marker_text)
        if char_pos >= 0:
            positions = _find_substring_token_range(prompt_tokens_decoded, prompt_text, marker_text)
            if positions:
                named_positions[name] = positions
                marker_positions_list.append((char_pos, name, marker_text, positions))

    marker_positions_list.sort(key=lambda x: x[0])

    # === Context Keywords ===
    # Get context values - use defaults merged with BASE_CONTEXT
    # Build context with defaults + BASE_CONTEXT overrides
    ctx = ContextConfig.from_dict(BASE_CONTEXT)

    # Map context keywords to their values and find positions
    # NOTE: reward_units is NOT included because it overlaps with left_reward/right_reward
    # (e.g., "dollars" is part of "1,750 dollars")
    context_keywords = {
        "situation": ctx.situation,
        "role": ctx.role,
        "task_in_question": ctx.task_in_question,
        "reasoning_ask": ctx.reasoning_ask,
    }

    for keyword_name, keyword_value in context_keywords.items():
        if keyword_value:  # Skip empty values
            positions = _find_substring_token_range(
                prompt_tokens_decoded, prompt_text, keyword_value
            )
            if positions:
                named_positions[keyword_name] = positions

    # === Option Labels (a), b) etc.) ===
    # These appear in the task section
    left_label = pair.short_term.label if sample.short_term_first else pair.long_term.label
    right_label = pair.long_term.label if sample.short_term_first else pair.short_term.label

    left_label_pos = _find_substring_token_range(prompt_tokens_decoded, prompt_text, left_label)
    right_label_pos = _find_substring_token_range(prompt_tokens_decoded, prompt_text, right_label)

    if left_label_pos:
        named_positions["left_label"] = left_label_pos
    if right_label_pos:
        named_positions["right_label"] = right_label_pos

    # === Variable Positions (rewards, times) ===
    if sample.prompt.time_horizon is not None:
        time_horizon_positions = _find_time_value_positions(
            prompt_tokens_decoded, prompt_text, sample.prompt.time_horizon
        )
        if time_horizon_positions:
            named_positions["time_horizon"] = time_horizon_positions

            # Find post_time_horizon: positions between time_horizon and action_marker
            action_marker_text = markers.get("action_marker", "ACTION:")
            action_marker_char = prompt_text.find(action_marker_text)
            if action_marker_char >= 0:
                # Get char position after time_horizon ends
                time_horizon_end_pos = max(time_horizon_positions)
                char_count = 0
                for i, tok in enumerate(prompt_tokens_decoded):
                    if i == time_horizon_end_pos:
                        time_horizon_end_char = char_count + len(tok)
                        break
                    char_count += len(tok)
                else:
                    time_horizon_end_char = len(prompt_text)

                # Find tokens between time_horizon end and action_marker
                post_time_horizon_positions = []
                char_count = 0
                for i, tok in enumerate(prompt_tokens_decoded):
                    tok_start = char_count
                    tok_end = char_count + len(tok)
                    if tok_start >= time_horizon_end_char and tok_start < action_marker_char:
                        post_time_horizon_positions.append(i)
                    char_count = tok_end
                if post_time_horizon_positions:
                    named_positions["post_time_horizon"] = post_time_horizon_positions

    # Determine left/right based on presentation order
    if sample.short_term_first:
        left_option = pair.short_term
        right_option = pair.long_term
    else:
        left_option = pair.long_term
        right_option = pair.short_term

    # Left option (appears first in prompt)
    left_time_pos = _find_time_value_positions(
        prompt_tokens_decoded, prompt_text, left_option.time
    )
    left_reward_pos = _find_reward_value_positions(
        prompt_tokens_decoded, prompt_text, left_option.reward
    )
    if left_time_pos:
        named_positions["left_time"] = left_time_pos
    if left_reward_pos:
        named_positions["left_reward"] = left_reward_pos

    # Right option (appears second in prompt)
    right_time_pos = _find_time_value_positions(
        prompt_tokens_decoded, prompt_text, right_option.time
    )
    right_reward_pos = _find_reward_value_positions(
        prompt_tokens_decoded, prompt_text, right_option.reward
    )
    if right_time_pos:
        named_positions["right_time"] = right_time_pos
    if right_reward_pos:
        named_positions["right_reward"] = right_reward_pos

    # === Response Regions ===
    # Note: "response" is captured by traj_section, not format_pos
    response_text = full_text[len(prompt_text):]
    response_tokens = decoded_tokens[prompt_len:]

    # Choice prefix and choice label (e.g., "a)" or "b)")
    choice_prefix_text = fmt.response_const_keywords.get("response_choice_prefix", "I choose: ")
    choice_prefix_core = choice_prefix_text.rstrip()
    choice_prefix_pos = response_text.find(choice_prefix_core)
    if choice_prefix_pos >= 0:
        prefix_positions = _find_substring_token_range(
            response_tokens, response_text, choice_prefix_core
        )
        if prefix_positions:
            named_positions["response_choice_prefix"] = [p + prompt_len for p in prefix_positions]

        # Find the full choice label (e.g., "a)" or "b)") after the prefix
        choice_start_char = choice_prefix_pos + len(choice_prefix_core)
        # Try both labels from the preference pair
        short_label = pair.short_term.label  # e.g., "a)"
        long_label = pair.long_term.label    # e.g., "b)"

        # Find which label appears after the choice prefix
        for label in [short_label, long_label]:
            label_pos = response_text.find(label, choice_start_char)
            if label_pos >= 0 and label_pos < choice_start_char + 10:  # Must be near prefix
                choice_positions = _find_substring_token_range(
                    response_tokens, response_text, label
                )
                if choice_positions:
                    named_positions["response_choice"] = [p + prompt_len for p in choice_positions]
                    break

    # Reasoning prefix and content
    reasoning_prefix_text = fmt.response_const_keywords.get("response_reasoning_prefix", "My reasoning: ")
    reasoning_prefix_core = reasoning_prefix_text.rstrip()
    reasoning_prefix_pos = response_text.find(reasoning_prefix_core)
    if reasoning_prefix_pos >= 0:
        prefix_positions = _find_substring_token_range(
            response_tokens, response_text, reasoning_prefix_core
        )
        if prefix_positions:
            named_positions["response_reasoning_prefix"] = [p + prompt_len for p in prefix_positions]

        reasoning_content_start = reasoning_prefix_pos + len(reasoning_prefix_text)
        char_count = 0
        reasoning_positions = []
        for idx, tok in enumerate(response_tokens):
            tok_end = char_count + len(tok)
            if tok_end > reasoning_content_start:
                reasoning_positions.append(idx + prompt_len)
            char_count = tok_end
        if reasoning_positions:
            named_positions["response_reasoning"] = reasoning_positions

    # Clamp all positions
    for key in named_positions:
        named_positions[key] = [max(0, min(p, full_len - 1)) for p in named_positions[key]]

    # Remove empty entries
    named_positions = {k: v for k, v in named_positions.items() if v}

    # === Fill in content regions for remaining unassigned positions ===
    # Build set of all assigned positions
    assigned_positions = set()
    for positions in named_positions.values():
        assigned_positions.update(positions)

    # Section markers define content region boundaries
    # Sub-section prefixes (format_choice_prefix, format_reasoning_prefix) are
    # mapped but don't create boundaries - they're within the format section
    section_markers = {
        "situation_marker": "situation_content",
        "task_marker": "task_content",
        "consider_marker": "consider_content",
        "constraint_marker": "constraint_content",
        "action_marker": "action_content",
        "format_marker": "format_content",
    }

    # Build ordered list of SECTION marker boundaries only
    # Format: (char_pos, marker_name, marker_text)
    marker_boundaries = []
    for name in section_markers:
        if name in markers:
            marker_text = markers[name]
            char_pos = prompt_text.find(marker_text)
            if char_pos >= 0:
                marker_boundaries.append((char_pos, name, marker_text))
    marker_boundaries.sort(key=lambda x: x[0])

    # Assign chat_prefix: all unassigned prompt tokens before first marker
    if marker_boundaries:
        first_marker_char = marker_boundaries[0][0]
        chat_prefix_positions = []
        char_count = 0
        for i, tok in enumerate(prompt_tokens_decoded):
            if i >= prompt_len:
                break
            tok_end = char_count + len(tok)
            if tok_end <= first_marker_char and i not in assigned_positions:
                chat_prefix_positions.append(i)
            char_count = tok_end
        if chat_prefix_positions:
            named_positions["chat_prefix"] = chat_prefix_positions
            assigned_positions.update(chat_prefix_positions)

    # Detect chat template suffix at end of prompt
    # Look for common chat template patterns: <|im_end|>, <|im_start|>, assistant, etc.
    chat_template_tokens = {"<|im_end|>", "<|im_start|>", "assistant", "<|eot_id|>", "<|start_header_id|>"}
    chat_suffix_start_pos = None

    # Scan from end of prompt to find where chat suffix begins
    for i in range(prompt_len - 1, -1, -1):
        tok = prompt_tokens_decoded[i]
        tok_stripped = tok.strip()

        # Skip whitespace-only tokens (newlines, spaces) - they're part of chat suffix
        if not tok_stripped:
            if chat_suffix_start_pos is not None:
                # Include whitespace that follows chat template tokens
                chat_suffix_start_pos = i
            continue

        if tok_stripped in chat_template_tokens or tok_stripped.startswith("<|") or tok_stripped.endswith("|>"):
            chat_suffix_start_pos = i
        else:
            # Stop scanning once we hit non-chat-template token
            break

    # Calculate character position where chat suffix starts
    chat_suffix_start_char = len(prompt_text)  # default: end of prompt
    if chat_suffix_start_pos is not None:
        char_count = 0
        for i, tok in enumerate(prompt_tokens_decoded):
            if i == chat_suffix_start_pos:
                chat_suffix_start_char = char_count
                break
            char_count += len(tok)

        # Assign chat_suffix positions
        chat_suffix_positions = list(range(chat_suffix_start_pos, prompt_len))
        if chat_suffix_positions:
            named_positions["chat_suffix"] = chat_suffix_positions
            assigned_positions.update(chat_suffix_positions)

    # Assign content regions between section markers
    # Also track tail positions for each section
    tail_positions: dict[str, int] = {}

    # Find options region boundaries (for splitting task_content)
    options_start_char = None
    options_end_char = None
    if "left_label" in named_positions:
        # Find char position of left_label start
        left_label_pos = min(named_positions["left_label"])
        char_count = 0
        for i, tok in enumerate(prompt_tokens_decoded):
            if i == left_label_pos:
                options_start_char = char_count
                break
            char_count += len(tok)

    if "right_time" in named_positions:
        # Find char position after right_time ends
        right_time_pos = max(named_positions["right_time"])
        char_count = 0
        for i, tok in enumerate(prompt_tokens_decoded):
            if i == right_time_pos:
                options_end_char = char_count + len(tok)
                break
            char_count += len(tok)

    # Find consider_marker position for options region boundary
    consider_marker_text = markers.get("consider_marker", "CONSIDER:")
    consider_marker_char = prompt_text.find(consider_marker_text)

    for idx, (char_pos, marker_name, marker_text) in enumerate(marker_boundaries):
        if marker_name not in section_markers:
            continue

        content_name = section_markers[marker_name]
        marker_end_char = char_pos + len(marker_text)

        # Find where next marker starts (or chat suffix, or end of prompt)
        if idx + 1 < len(marker_boundaries):
            next_marker_char = marker_boundaries[idx + 1][0]
        else:
            # For last section, stop at chat suffix if present
            next_marker_char = chat_suffix_start_char

        # For task section, stop content before options begin
        content_end_char = next_marker_char
        if marker_name == "task_marker" and options_start_char is not None:
            content_end_char = options_start_char

        # Find unassigned tokens in this content region
        char_count = 0
        content_positions = []
        for i, tok in enumerate(prompt_tokens_decoded):
            if i >= prompt_len:
                break
            tok_start = char_count
            tok_end = char_count + len(tok)
            # Token is in content region if it starts after marker ends and before boundary
            if tok_start >= marker_end_char and tok_start < content_end_char:
                if i not in assigned_positions:
                    content_positions.append(i)
            char_count = tok_end
        if content_positions:
            named_positions[content_name] = content_positions
            assigned_positions.update(content_positions)
            # Derive tail name from section marker (situation_marker -> situation_tail)
            section_name = marker_name.replace("_marker", "")
            tail_positions[f"{section_name}_tail"] = max(content_positions)

    # Add option_content: unassigned positions in options region
    # options_tail is the last position before consider_marker
    if options_start_char is not None:
        # Options region extends from left_label to just before consider_marker
        options_region_end = consider_marker_char if consider_marker_char >= 0 else len(prompt_text)

        char_count = 0
        option_content_positions = []
        last_option_pos = None
        for i, tok in enumerate(prompt_tokens_decoded):
            if i >= prompt_len:
                break
            tok_start = char_count
            tok_end = char_count + len(tok)
            # Token is in options region if it starts after options_start and before consider_marker
            if tok_start >= options_start_char and tok_start < options_region_end:
                last_option_pos = i
                if i not in assigned_positions:
                    option_content_positions.append(i)
            char_count = tok_end

        if option_content_positions:
            named_positions["option_content"] = option_content_positions
            assigned_positions.update(option_content_positions)

        # options_tail is the last UNASSIGNED position in the options region
        # Don't use last_option_pos directly since it might be right_time
        if option_content_positions:
            options_tail_pos = max(option_content_positions)
            named_positions["options_tail"] = [options_tail_pos]

    # Add tail positions (last position of each content section)
    # Special case: consider_tail should be BEFORE time_horizon if it exists
    if "consider_tail" in tail_positions and "time_horizon" in named_positions:
        time_horizon_start = min(named_positions["time_horizon"])
        # Find the position just before time_horizon
        consider_positions = named_positions.get("consider_content", [])
        positions_before_horizon = [p for p in consider_positions if p < time_horizon_start]
        if positions_before_horizon:
            tail_positions["consider_tail"] = max(positions_before_horizon)

    # Special case: task_tail should be BEFORE options (left_label) if it exists
    if "task_tail" in tail_positions and "left_label" in named_positions:
        left_label_start = min(named_positions["left_label"])
        # Find the position just before left_label
        task_positions = named_positions.get("task_content", [])
        positions_before_options = [p for p in task_positions if p < left_label_start]
        if positions_before_options:
            tail_positions["task_tail"] = max(positions_before_options)

    for tail_name, tail_pos in tail_positions.items():
        named_positions[tail_name] = [tail_pos]
        # Remove tail position from corresponding content list to avoid overlap
        content_name = tail_name.replace("_tail", "_content")
        if content_name in named_positions and tail_pos in named_positions[content_name]:
            named_positions[content_name].remove(tail_pos)

    # Add chat_prefix_tail and chat_suffix_tail if those regions exist
    if "chat_prefix" in named_positions and named_positions["chat_prefix"]:
        tail_pos = max(named_positions["chat_prefix"])
        named_positions["chat_prefix_tail"] = [tail_pos]
        named_positions["chat_prefix"].remove(tail_pos)
    if "chat_suffix" in named_positions and named_positions["chat_suffix"]:
        tail_pos = max(named_positions["chat_suffix"])
        named_positions["chat_suffix_tail"] = [tail_pos]
        named_positions["chat_suffix"].remove(tail_pos)

    # Remove options_tail from option_content if present
    if "options_tail" in named_positions and "option_content" in named_positions:
        tail_pos = named_positions["options_tail"][0]
        if tail_pos in named_positions["option_content"]:
            named_positions["option_content"].remove(tail_pos)

    # Assign prompt_other: any remaining unassigned prompt tokens
    # (includes chat template suffix if present)
    prompt_other_positions = [
        i for i in range(prompt_len)
        if i not in assigned_positions
    ]
    if prompt_other_positions:
        named_positions["prompt_other"] = prompt_other_positions
        assigned_positions.update(prompt_other_positions)

    # Assign response_other: unassigned response tokens
    response_other_positions = [
        i for i in range(prompt_len, full_len)
        if i not in assigned_positions
    ]
    if response_other_positions:
        named_positions["response_other"] = response_other_positions

    return named_positions
