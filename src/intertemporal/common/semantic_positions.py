"""Canonical semantic position and layer constants for intertemporal experiments.

These constants define the semantic token positions and layer indices used
throughout the analysis pipeline.
"""

# Prompt positions (where time horizon info is encoded)
PROMPT_POSITIONS = [
    "task_tail",
    "consider_tail",
    "time_horizon",
    "post_time_horizon",
    "action_tail",
    "format_tail",
    "chat_suffix",
]

# Response positions (where model output is generated)
RESPONSE_POSITIONS = [
    "response_choice_prefix",
    "response_choice",
    "response_reasoning_prefix",
    "response_reasoning",
]

# Layers for circuit attention analysis
DEFAULT_LAYERS = [19, 21, 24, 31, 34]

# Early layers for intermediate attention analysis
EARLY_LAYERS = [10, 11, 12, 13, 14, 15, 16, 17]

# Intermediate positions (where info flows from source to dest)
INTERMEDIATE_POSITIONS = [
    "action_content",
    "format_content",
    "chat_suffix",
]
