from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Optional

from src.common import TimeValue

from .prompt_format_config import PromptFormatConfig


@dataclass
class DefaultPromptFormat(PromptFormatConfig):
    name: str = "default_prompt_format"

    situation_template: str = "[situation_marker] [situation] [extra_situation]"
    task_template: str = """[task_marker] You, [role], are tasked to [task_in_question]:
[left_term_label] [left_term_reward] [reward_units] in [left_term_time]
[right_term_label] [right_term_reward] [reward_units] in [right_term_time]"""
    objective_template: str = (
        "[objective_marker] Think deeply about which option is preferable."
    )
    constraint_template: str = "[constraint_marker] [constraint_prefix] [time_horizon]."
    action_template: str = (
        "[action_marker] Select one of the two options. [reasoning_ask]"
    )

    response_template: str = """\n[format_marker] Respond in this format:
[format_choice_prefix] <[left_term_label] or [right_term_label]>.
[format_reasoning_prefix] <reasoning in 1-3 sentences>\n"""

    def get_id(self):
        content = (
            self.question_template(None)
            + self.question_template(TimeValue(0))
            + self.response_template
        )
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def question_template(self, time_horizon: Optional[TimeValue] = None) -> str:
        """Assemble the question template, including time-horizon spec when present."""
        parts = [
            self.situation_template,
            self.task_template,
            self.objective_template,
        ]
        if time_horizon is not None:
            parts.append(self.constraint_template)
        parts.append(self.action_template)
        return "\n".join(parts)

    prompt_const_keywords: dict = field(
        default_factory=lambda: {
            "situation_marker": "SITUATION:",
            "task_marker": "TASK:",
            "objective_marker": "OBJECTIVE:",
            "constraint_marker": "CONSTRAINT:",
            "action_marker": "ACTION:",
            "format_marker": "FORMAT:",
            "constraint_prefix": "You must select the option that provides the greatest benefit for:",
            "format_choice_prefix": "I choose:",
            "format_reasoning_prefix": "My reasoning:",
        }
    )

    response_const_keywords: dict = field(
        default_factory=lambda: {
            "response_choice_prefix": "I choose: ",  # note space
            "response_reasoning_prefix": "My reasoning: ",  # note space
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

    def get_prompt_markers(self) -> dict[str, str]:
        """Return mapping of prompt section names to their marker text.

        Only includes prompt-structure markers (not response markers).
        Used for splitting prompt text into sections.
        """
        return {
            "situation": self.prompt_const_keywords["situation_marker"],
            "task": self.prompt_const_keywords["task_marker"],
            "consider": self.prompt_const_keywords["objective_marker"],
            "constraint": self.prompt_const_keywords["constraint_marker"],
            "action": self.prompt_const_keywords["action_marker"],
            "format": self.prompt_const_keywords["format_marker"],
        }

    def get_response_markers(self) -> dict[str, str]:
        """Return mapping of response section names to their marker text.

        Used for splitting response text into choice/reasoning sections.
        """
        return {
            "choice_prefix": self.response_const_keywords["response_choice_prefix"],
            "reasoning_prefix": self.response_const_keywords[
                "response_reasoning_prefix"
            ],
        }

    def get_response_prefix_before_choice(self) -> str:
        return self.response_const_keywords["response_choice_prefix"]
