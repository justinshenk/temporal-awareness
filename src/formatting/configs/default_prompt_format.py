from dataclasses import dataclass, field

from ...datasets.schemas import PromptFormatConfig


@dataclass
class DefaultPromptFormat(PromptFormatConfig):
    name: str = "default_prompt_format"

    question_template: str = """SITUATION: [situation] [extra_situation]
TASK: You, [role], are tasked to [task_in_question]:
[left_term_label] [left_term_reward] [reward_units] in [left_term_time]
[right_term_label] [right_term_reward] [reward_units] in [right_term_time]
CONSIDER: Think deeply about which option is preferable. [time_horizon_spec]
ACTION: Select one of the two options, and [reasoning_ask]"""

    response_template: str = """FORMAT: Respond in this format:
[choice_prefix] <[left_term_label] or [right_term_label]>.
[reasoning_prefix] <reasoning in [max_reasoning_length]>"""

    const_keywords: dict = field(
        default_factory=lambda: {
            "choice_prefix": "I select:",
            "reasoning_prefix": "My reasoning:",
            "time_horizon_spec": "You are primarily concerned about outcome in [time_horizon].",
            "max_reasoning_length": "1-3 sentences",
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
