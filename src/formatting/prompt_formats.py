"""
Shared formatting-dependent boundary markers for visualization and steering.

These markers map formatting_id to keywords used to identify boundaries
in the prompt/response for:
- Token position selection (which probes to use)
- Heatmap boundary lines
- Steering experiments

Update FORMATTING_BOUNDARY_MARKERS when formatting configs change.
"""

from __future__ import annotations


def find_prompt_format_config(prompt_format_config_name: str):
    from .configs import DefaultPromptFormat

    for cfg_ctor in [DefaultPromptFormat]:
        cfg = cfg_ctor()
        if cfg.name == prompt_format_config_name:
            return cfg
    raise Exception(f"{prompt_format_config_name} is not a valid prompt format config")
