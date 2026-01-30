from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from ...datasets.schemas import PromptFormatConfig

if TYPE_CHECKING:
    from ...common.types import TimeValue


@dataclass
class DefaultPromptFormat(PromptFormatConfig):
    name: str = "default_prompt_format"

    situation_template: str = "[situation_marker] [situation] [extra_situation]"
    task_template: str = """[task_marker] You, [role], are tasked to [task_in_question]:
[left_term_label] [left_term_reward] [reward_units] in [left_term_time]
[right_term_label] [right_term_reward] [reward_units] in [right_term_time]"""
    consider_template: str = "[consider_marker] Think deeply about which option is preferable."
    time_horizon_spec_template: str = "You are primarily concerned about outcome in [time_horizon]."
    action_template: str = "[action_marker] Select one of the two options, and [reasoning_ask]"

    response_template: str = """[format_marker] Respond in this format:
[format_choice_prefix] <[left_term_label] or [right_term_label]>.
[format_reasoning_prefix] <reasoning in 1-3 sentences>"""

    def question_template(self, time_horizon: Optional[TimeValue] = None) -> str:
        """Assemble the question template, including time-horizon spec when present."""
        parts = [
            self.situation_template,
            self.task_template,
            self.consider_template,
        ]
        if time_horizon is not None:
            parts.append(self.time_horizon_spec_template)
        parts.append(self.action_template)
        return "\n".join(parts)

    const_keywords: dict = field(
        default_factory=lambda: {
            "situation_marker": "SITUATION:",
            "task_marker": "TASK:",
            "consider_marker": "CONSIDER:",
            "action_marker": "ACTION:",
            "format_marker": "FORMAT:",
            "format_choice_prefix": "I select:",
            "format_reasoning_prefix": "My reasoning:",
        }
    )

    response_const_keywords: dict = field(
        default_factory=lambda: {
            "response_choice_prefix": "I select:",
            "response_reasoning_prefix": "My reasoning:",
        }
    )

    keywords: list = field(
        default_factory=lambda: [
            "situation",
            "extra_situation",
            "role",
            "task_in_question",
            "reward_units",
            "reasoning_ask",
        ]
    )

    var_keywords: list = field(
        default_factory=lambda: [
            "time_horizon",
            "left_term_label",
            "left_term_reward",
            "left_term_time",
            "right_term_label",
            "right_term_reward",
            "right_term_time",
        ]
    )

    def get_interesting_positions(self) -> list[dict]:
        """Return token position specs for all prompt and response markers.

        Prompt const_keywords are searched as first-occurrence (they appear in
        the prompt/FORMAT section).  Response const_keywords are searched as
        last-occurrence (they appear in the model's actual response, after the
        FORMAT instructions that contain the same text).
        """
        positions: list[dict] = []
        # Prompt markers — first occurrence
        for _key, value in self.const_keywords.items():
            positions.append({"text": value})
        # Response markers — last occurrence (same text appears in FORMAT
        # section AND in the response; we want the response occurrence)
        for _key, value in self.response_const_keywords.items():
            positions.append({"text": value, "last": True})
        return positions
