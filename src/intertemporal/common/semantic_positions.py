"""Canonical semantic position and layer constants for intertemporal experiments.

These constants define the semantic token positions and layer indices used
throughout the analysis pipeline.
"""

# default layers
DEFAULT_LAYERS = [8, 19, 21, 24, 28, 31, 34, 35]


##################################
##################################
##################################

# Prompt positions
PROMPT_CONSTRAINT_POSITIONS = [
    "time_horizon",
    "post_time_horizon",
]

PROMPT_LABEL_POSITIONS = [
    "left_label",
    "right_label",
]

PROMPT_TIME_POSITIONS = [
    "left_time",
    "right_time",
]

PROMPT_REWARD_POSITIONS = [
    "left_reward",
    "right_reward",
]

PROMPT_INFO_POSITIONS = (
    PROMPT_LABEL_POSITIONS + PROMPT_TIME_POSITIONS + PROMPT_REWARD_POSITIONS
)


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

##################################
##################################
##################################

# Response positions
RESPONSE_POSITIONS = ["response_choice_prefix", "response_choice"]

# Prompt positions
PROMPT_POSITIONS = PROMPT_SRC_POSITIONS + PROMPT_SECTION_TAILS

# All positions
ALL_TRAJECTORY_POSITIONS = PROMPT_POSITIONS + RESPONSE_POSITIONS

##################################
##################################
##################################

# time positions
TIME_POSITIONS = (
    PROMPT_TIME_POSITIONS + PROMPT_CONSTRAINT_POSITIONS + RESPONSE_POSITIONS
)
