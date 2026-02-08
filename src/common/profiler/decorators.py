"""Profiling decorators for experiment steps."""

from __future__ import annotations

import functools
from typing import Callable, TypeVar

from .timer import P
from ..device import log_memory

F = TypeVar("F", bound=Callable)


def profile_fn(
    identifier: str,
) -> Callable[[F], F]:
    """Decorator to profile functions."""

    profile_name = identifier.lower().replace(" ", "_")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Print step header
            print(f"\n{'=' * 60}")
            print(f"{identifier}")
            print("=" * 60)

            # Run with profiling
            with P(profile_name):
                result = func(*args, **kwargs)

            # Log memory
            log_memory(f"after_{profile_name}")

            return result

        return wrapper  # type: ignore

    return decorator
