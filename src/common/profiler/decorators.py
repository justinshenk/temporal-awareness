"""Profiling decorators for experiment steps."""

from __future__ import annotations

import functools
from typing import Callable, TypeVar, Union, overload

from .timer import P
from ..device_utils import log_memory

F = TypeVar("F", bound=Callable)


@overload
def profile(func: F) -> F: ...


@overload
def profile(identifier: str) -> Callable[[F], F]: ...


def profile(
    func_or_identifier: Union[F, str, None] = None, verbose: bool = False
) -> Union[F, Callable[[F], F]]:
    """Decorator to profile functions.

    Can be used with or without arguments:
        @profile
        def my_func(): ...

        @profile("custom_name")
        def my_func(): ...

    If no identifier is provided, uses the function's name.
    """

    def make_wrapper(func: F, name: str, use_verbose: bool) -> F:
        profile_name = name.lower().replace(" ", "_")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Print step header
            if use_verbose:
                print(f"\n{'=' * 60}")
                print(f"PROFILE: {name}")
                print("=" * 60)

            # Run with profiling
            with P(profile_name):
                result = func(*args, **kwargs)

            # Log memory
            log_memory(f"after_{profile_name}", use_verbose)

            return result

        return wrapper  # type: ignore

    # Called as @profile (no parens) - func_or_identifier is the function
    if callable(func_or_identifier):
        return make_wrapper(func_or_identifier, func_or_identifier.__name__, verbose)

    # Called as @profile() or @profile("name") - func_or_identifier is str or None
    identifier = func_or_identifier or ""

    def decorator(func: F) -> F:
        name = identifier if identifier else func.__name__
        return make_wrapper(func, name, verbose)

    return decorator
