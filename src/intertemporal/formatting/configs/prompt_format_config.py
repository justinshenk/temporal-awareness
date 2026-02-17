"""Base prompt format configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.common import TimeValue


class PromptFormatConfig(ABC):
    """Abstract base class for prompt format configurations.

    Defines the interface that all prompt format implementations must follow.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this prompt format."""
        ...

    @abstractmethod
    def get_exact_prefix_before_choice(self) -> str:
        """Return the EXACT prefix (including) right before  choice"""
        ...

    @abstractmethod
    def question_template(self, time_horizon: Optional[TimeValue] = None) -> str:
        """Assemble the question template.

        Args:
            time_horizon: Optional time horizon constraint

        Returns:
            Complete question template string with placeholders
        """
        ...

    @abstractmethod
    def get_prompt_section_markers(self) -> dict[str, str]:
        """Return mapping of prompt section names to their marker text.

        Only includes prompt-structure markers (not response markers).
        Used for splitting prompt text into sections.
        """
        ...

    @abstractmethod
    def get_response_markers(self) -> dict[str, str]:
        """Return mapping of response section names to their marker text.

        Used for splitting response text into choice/reasoning sections.
        """
        ...

    @abstractmethod
    def get_interesting_positions(self) -> list[dict]:
        """Return token position specs for all prompt and response markers.

        Prompt const_keywords are searched as first-occurrence.
        Response const_keywords are searched as last-occurrence.
        """
        ...

    @abstractmethod
    def get_anchor_texts(self) -> list[str]:
        """Return text anchors for position alignment between sequences.

        Extracts text values suitable for aligning token positions
        across different prompts with the same structure.
        """
        ...
