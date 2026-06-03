"""Context variations for prompt dataset generation.

Defines a list of ContextConfig variations that can be used to generate
additional samples with different context_ids.
"""

from __future__ import annotations

from .prompt_dataset_config import ContextConfig


# Context variations for generating diverse samples
# Each variation changes semantic aspects of the prompt while keeping the same structure
CONTEXT_VARIATIONS: list[ContextConfig] = [
    # Variation 1: Different role framing - individual vs organization
    ContextConfig(
        role="an individual making personal decisions",
        task_in_question="choose the best option for yourself",
    ),
    # Variation 2: Different reasoning ask - analytical vs intuitive
    ContextConfig(
        reasoning_ask="Explain your reasoning process step by step.",
    ),
    # Variation 3: Different reasoning ask - brief
    ContextConfig(
        reasoning_ask="Briefly justify your choice.",
    ),
    # Variation 4: Extra situation context - emphasize tradeoffs
    ContextConfig(
        extra_situation="Consider the tradeoffs carefully before deciding.",
    ),
    # Variation 5: Extra situation context - emphasize long-term thinking
    ContextConfig(
        extra_situation="Think about the long-term consequences of your decision.",
    ),
    # Variation 6: Combined - individual + analytical reasoning
    ContextConfig(
        role="an individual making personal decisions",
        reasoning_ask="Explain your reasoning process step by step.",
    ),
    # Variation 7: Combined - organization + brief reasoning
    ContextConfig(
        role="a committee responsible for this decision",
        reasoning_ask="Briefly justify your choice.",
    ),
]


def get_context_variations() -> list[ContextConfig]:
    """Get the list of context variations."""
    return CONTEXT_VARIATIONS


def apply_context_variation(
    base_context: ContextConfig,
    variation: ContextConfig,
) -> ContextConfig:
    """Apply a variation to a base context config.

    Non-default fields from the variation override the base context.

    Args:
        base_context: The base context configuration
        variation: The variation to apply

    Returns:
        New ContextConfig with variation applied
    """
    # Get default values
    defaults = ContextConfig()

    # Start with base context as dict
    result_dict = base_context.to_dict()

    # Apply non-default values from variation
    variation_dict = variation.to_dict()
    for key, value in variation_dict.items():
        default_value = getattr(defaults, key)
        if value != default_value:
            result_dict[key] = value

    return ContextConfig.from_dict(result_dict)
