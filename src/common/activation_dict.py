"""Arithmetic-capable dict for activation caches."""

from __future__ import annotations

from collections.abc import Mapping, Iterator
from typing import Callable

import torch


class ActivationDict(Mapping[str, torch.Tensor]):
    """Arithmetic-capable dict for activation caches.

    Implements the Mapping protocol and supports element-wise arithmetic
    operations with intersection semantics (operates on common keys).

    Keys use TransformerLens format: "blocks.{layer}.hook_{component}"
    """

    def __init__(
        self, data: dict[str, torch.Tensor] | None = None, *, frozen: bool = False
    ):
        """Initialize ActivationDict.

        Args:
            data: Initial data dict
            frozen: If True, prevent modifications
        """
        self._data: dict[str, torch.Tensor] = dict(data) if data else {}
        self._frozen = frozen

    # Mapping protocol
    def __getitem__(self, key: str) -> torch.Tensor:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    # Mutation
    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if self._frozen:
            raise TypeError("Cannot modify frozen ActivationDict")
        self._data[key] = value

    def freeze(self) -> ActivationDict:
        """Return a frozen copy that cannot be modified."""
        return ActivationDict(self._data, frozen=True)

    @property
    def frozen(self) -> bool:
        """Return whether this ActivationDict is frozen."""
        return self._frozen

    # Arithmetic (intersection semantics)
    def __sub__(self, other: ActivationDict) -> ActivationDict:
        """Element-wise subtraction on common keys."""
        common = self.keys() & other.keys()
        return ActivationDict({k: self._data[k] - other._data[k] for k in common})

    def __add__(self, other: ActivationDict) -> ActivationDict:
        """Element-wise addition on common keys."""
        common = self.keys() & other.keys()
        return ActivationDict({k: self._data[k] + other._data[k] for k in common})

    def __mul__(self, other: ActivationDict | float | int) -> ActivationDict:
        """Element-wise multiplication."""
        if isinstance(other, (int, float)):
            return ActivationDict({k: v * other for k, v in self._data.items()})
        common = self.keys() & other.keys()
        return ActivationDict({k: self._data[k] * other._data[k] for k in common})

    def __rmul__(self, other: float | int) -> ActivationDict:
        """Right multiplication by scalar."""
        return self.__mul__(other)

    def __truediv__(self, scalar: float | int) -> ActivationDict:
        """Division by scalar."""
        return ActivationDict({k: v / scalar for k, v in self._data.items()})

    # Gradient control
    def detach(self) -> ActivationDict:
        """Return a copy with all tensors detached from the computation graph."""
        return ActivationDict({k: v.detach() for k, v in self._data.items()})

    def clone(self) -> ActivationDict:
        """Return a copy with cloned tensors."""
        return ActivationDict({k: v.clone() for k, v in self._data.items()})

    # Utilities
    def get(self, key: str, default: torch.Tensor | None = None) -> torch.Tensor | None:
        """Get a value by key, returning default if not found."""
        return self._data.get(key, default)

    def filter(self, predicate: Callable[[str], bool]) -> ActivationDict:
        """Return a new ActivationDict with only keys matching the predicate."""
        return ActivationDict({k: v for k, v in self._data.items() if predicate(k)})

    def to_device(self, device: str | torch.device) -> ActivationDict:
        """Return a copy with all tensors moved to the specified device."""
        return ActivationDict({k: v.to(device) for k, v in self._data.items()})

    def sum_hidden(self) -> ActivationDict:
        """Sum over hidden dimension -> [batch, seq] or [seq]."""
        return ActivationDict({k: v.sum(dim=-1) for k, v in self._data.items()})

    def __repr__(self) -> str:
        frozen_str = ", frozen=True" if self._frozen else ""
        return f"ActivationDict({list(self._data.keys())}{frozen_str})"
