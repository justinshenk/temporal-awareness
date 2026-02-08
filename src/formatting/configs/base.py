"""Base prompt format configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...common.types import TimeValue


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
    def question_template(self, time_horizon: Optional[TimeValue] = None) -> str:
        """Assemble the question template.

        Args:
            time_horizon: Optional time horizon constraint

        Returns:
            Complete question template string with placeholders
        """
        ...

    @abstractmethod
    def get_keyword_map(self) -> dict[str, str]:
        """Return mapping of short keyword names to their search text.

        Used by token_positions for resolving keyword-based position specs.
        Maps e.g. "situation" -> "SITUATION:", "choice_prefix" -> "I select:".
        """
        ...

    @abstractmethod
    def get_last_occurrence_keyword_names(self) -> set[str]:
        """Return keyword names that should match LAST occurrence.

        These keywords appear in both the FORMAT section (first) and the
        actual response (last). Position resolution should find the last
        occurrence to target the response instance.
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
