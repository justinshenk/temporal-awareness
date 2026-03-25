"""Sample position mapping for token-level analysis.

Maps each absolute token position to semantic position information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.common import BaseSchema

if TYPE_CHECKING:
    from .preference_types import PreferenceSample, PromptSample


@dataclass
class TokenPositionInfo(BaseSchema):
    """Info for a single token position.

    Attributes:
        abs_pos: Absolute position index in the sequence
        decoded_token: The decoded token string at this position
        traj_section: Either "prompt" or "response"
        format_pos: Semantic position name (e.g., "response_choice_prefix"), or None
        rel_pos: Relative position within format_pos (0-indexed), -1 if not in named position
    """

    abs_pos: int
    decoded_token: str
    traj_section: str  # "prompt" or "response"
    format_pos: str | None = None
    rel_pos: int = -1


@dataclass
class SamplePositionMapping(BaseSchema):
    """Position mapping for a single sample.

    Maps every absolute token position to its semantic meaning.

    Attributes:
        sample_idx: Index of the sample
        prompt_len: Number of tokens in the prompt
        full_len: Total number of tokens (prompt + response)
        positions: List of TokenPositionInfo, indexed by abs_pos
        named_positions: Dict mapping format_pos names to list of abs_pos indices
    """

    sample_idx: int
    prompt_len: int
    full_len: int
    positions: list[TokenPositionInfo] = field(default_factory=list)
    named_positions: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        sample: "PromptSample",
        pref: "PreferenceSample",
        tokenizer: Any,
    ) -> "SamplePositionMapping":
        """Build position mapping from sample data.

        Args:
            sample: The PromptSample with prompt structure info
            pref: The PreferenceSample with tokenized trajectory
            tokenizer: Tokenizer with decode method (or runner with _tokenizer)

        Returns:
            SamplePositionMapping with all positions mapped
        """
        from ..formatting.configs.default_prompt_format import DefaultPromptFormat

        # Get tokenizer if runner was passed
        if hasattr(tokenizer, "_tokenizer"):
            tokenizer = tokenizer._tokenizer

        # Get token IDs and decode each
        token_ids = pref.chosen_traj.token_ids
        full_len = len(token_ids)
        prompt_len = pref.prompt_token_count

        decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]

        # Build named positions using the same logic as resolve_positions
        named_positions = _build_named_positions(
            sample, pref, decoded_tokens, prompt_len, full_len
        )


        # Build reverse mapping: abs_pos -> (format_pos, rel_pos)
        pos_to_format: dict[int, tuple[str, int]] = {}
        for format_name, abs_positions in named_positions.items():
            for rel_idx, abs_pos in enumerate(abs_positions):
                # If position already mapped, prefer shorter position names
                # (more specific positions)
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

        return cls(
            sample_idx=sample.sample_idx,
            prompt_len=prompt_len,
            full_len=full_len,
            positions=positions,
            named_positions=named_positions,
        )

    def get_position(self, abs_pos: int) -> TokenPositionInfo | None:
        """Get position info by absolute position."""
        if 0 <= abs_pos < len(self.positions):
            return self.positions[abs_pos]
        return None

    def get_positions_by_name(self, format_pos: str) -> list[TokenPositionInfo]:
        """Get all positions for a named format position."""
        if format_pos not in self.named_positions:
            return []
        return [self.positions[i] for i in self.named_positions[format_pos]]

    def get_format_pos_names(self) -> list[str]:
        """Get list of all format position names in this sample."""
        return list(self.named_positions.keys())


@dataclass
class DatasetPositionMapping(BaseSchema):
    """Position mappings for all samples in a dataset."""

    mappings: list[SamplePositionMapping] = field(default_factory=list)

    def add(self, mapping: SamplePositionMapping):
        """Add a sample mapping."""
        self.mappings.append(mapping)

    def get(self, sample_idx: int) -> SamplePositionMapping | None:
        """Get mapping by sample index."""
        for m in self.mappings:
            if m.sample_idx == sample_idx:
                return m
        return None

    def __len__(self) -> int:
        return len(self.mappings)

    def __iter__(self):
        return iter(self.mappings)

    def save(self, path: Path):
        """Save to JSON file."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DatasetPositionMapping":
        """Load from JSON file."""
        import json

        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# Position Building (factored from geo_viz_data.py)
# =============================================================================


def _find_substring_token_range(
    tokens: list[str], text: str, substring: str
) -> list[int]:
    """Find all token positions spanning a substring in text."""
    char_idx = text.find(substring)
    if char_idx == -1:
        return []

    char_end = char_idx + len(substring)
    positions = []

    char_count = 0
    for i, tok in enumerate(tokens):
        tok_start = char_count
        tok_end = char_count + len(tok)
        char_count = tok_end

        if tok_end > char_idx and tok_start < char_end:
            positions.append(i)

        if tok_end >= char_end:
            break

    return positions


def _find_time_value_positions(
    tokens: list[str], text: str, time_val
) -> list[int]:
    """Find token positions for a TimeValue's formatted string.

    Searches for the complete formatted time (e.g., "1 month") as a unit,
    not split into parts, to avoid matching partial strings elsewhere.
    """
    formatted = str(time_val)  # e.g., "1 month", "2 years"
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
    pref: "PreferenceSample",
    decoded_tokens: list[str],
    prompt_len: int,
    full_len: int,
) -> dict[str, list[int]]:
    """Build named positions dictionary from sample structure.

    This mirrors the logic in geo_viz_data.resolve_positions but works
    with pre-decoded tokens.
    """
    from ..formatting.configs.default_prompt_format import DefaultPromptFormat

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
    # Get context values - use defaults merged with GEO_VIZ context
    from ..data.default_datasets import BASE_CONTEXT
    from ..prompt.prompt_dataset_config import ContextConfig

    # Build context with defaults + BASE_CONTEXT overrides
    ctx = ContextConfig.from_dict(BASE_CONTEXT)

    # Map context keywords to their values and find positions
    context_keywords = {
        "situation": ctx.situation,
        "role": ctx.role,
        "task_in_question": ctx.task_in_question,
        "reasoning_ask": ctx.reasoning_ask,
        "reward_units": ctx.reward_unit,
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

    # Assign content regions between section markers
    for idx, (char_pos, marker_name, marker_text) in enumerate(marker_boundaries):
        if marker_name not in section_markers:
            continue

        content_name = section_markers[marker_name]
        marker_end_char = char_pos + len(marker_text)

        # Find where next marker starts (or end of prompt)
        if idx + 1 < len(marker_boundaries):
            next_marker_char = marker_boundaries[idx + 1][0]
        else:
            next_marker_char = len(prompt_text)

        # Find unassigned tokens in this content region
        char_count = 0
        content_positions = []
        for i, tok in enumerate(prompt_tokens_decoded):
            if i >= prompt_len:
                break
            tok_start = char_count
            tok_end = char_count + len(tok)
            # Token is in content region if it starts after marker ends and before next marker
            if tok_start >= marker_end_char and tok_start < next_marker_char:
                if i not in assigned_positions:
                    content_positions.append(i)
            char_count = tok_end
        if content_positions:
            named_positions[content_name] = content_positions
            assigned_positions.update(content_positions)

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
