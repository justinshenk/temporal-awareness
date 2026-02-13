"""Abstract binary choice base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..base_schema import BaseSchema


@dataclass
class BinaryChoice(BaseSchema, ABC):
    """Abstract base class for binary choice results."""

    @property
    @abstractmethod
    def choice_idx(self) -> int:
        """Index of chosen option: 0 for A, 1 for B, -1 if tied/unknown."""
        ...

    @property
    @abstractmethod
    def alternative_idx(self) -> int:
        """Index of alternative option: 0 for A, 1 for B, -1 if tied/unknown."""
        ...

    @property
    @abstractmethod
    def choice_logprob(self) -> float | None:
        """Log probability of the chosen option."""
        ...

    @property
    @abstractmethod
    def alternative_logprob(self) -> float | None:
        """Log probability of the alternative (non-chosen) option."""
        ...


@dataclass
class LabeledBinaryChoice(BinaryChoice, ABC):
    """Abstract binary choice with semantic labels."""

    @property
    @abstractmethod
    def labels(self) -> tuple[str, str] | None:
        """Labels for options A and B."""
        ...

    @property
    def chosen_label(self) -> str | None:
        """Label of the chosen option."""
        if self.labels is None:
            return None
        idx = self.choice_idx
        if idx == -1:
            return None
        return self.labels[idx]

    @property
    def alternative_label(self) -> str | None:
        """Label of the alternative (non-chosen) option."""
        if self.labels is None:
            return None
        idx = self.choice_idx
        if idx == -1:
            return None
        return self.labels[1 - idx]
