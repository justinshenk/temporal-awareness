"""Canonical semantic position and layer constants for intertemporal experiments.

These constants define the semantic token positions and layer indices used
throughout the analysis pipeline.
"""

# Response positions (where model output is generated)
RESPONSE_POSITIONS = ["response_choice_prefix", "response_choice"]


DEFAULT_LAYERS = [8, 19, 21, 24, 31, 34]


# Prompt positions
PROMPT_CONSTRAINT_POSITIONS = [
    "time_horizon",
    "post_time_horizon",
]

PROMPT_INFO_POSITIONS = [
    "left_label",
    "right_label",
    "left_time",
    "right_time",
    "left_reward",
    "right_reward",
]


PROMPT_SRC_POSITIONS = PROMPT_CONSTRAINT_POSITIONS + PROMPT_INFO_POSITIONS


PROMPT_SECTION_TAILS = [
    "task_tail",
    "options_tail",
    "objective_tail",
    "action_tail",
    "format_tail",
    "chat_suffix",
    "chat_suffix_tail",
]


PROMPT_POSITIONS = PROMPT_SRC_POSITIONS + PROMPT_SECTION_TAILS
