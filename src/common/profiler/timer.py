"""Simple profiling timer with singleton API.

Usage:
    from src.common.profiler import P

    # Context manager (recommended)
    with P("load_data"):
        data = load()

    # Manual start/stop
    P.start("train")
    # ... training ...
    P.stop("train")

    # Nested timing
    with P("outer"):
        with P("inner"):
            work()

    # Report
    P.report()      # Print summary
    P.reset()       # Clear all timings
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TimingEntry:
    """Single timing entry."""

    name: str
    total: float = 0.0
    count: int = 0
    children: list[str] = field(default_factory=list)
    parent: Optional[str] = None

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


class Profiler:
    """Simple profiler with hierarchical timing."""

    def __init__(self):
        self._entries: dict[str, TimingEntry] = {}
        self._starts: dict[str, float] = {}
        self._stack: list[str] = []
        self._enabled: bool = True

    def enable(self) -> None:
        """Enable profiling."""
        self._enabled = True

    def disable(self) -> None:
        """Disable profiling (no-op mode)."""
        self._enabled = False

    def reset(self) -> None:
        """Clear all timing data."""
        self._entries.clear()
        self._starts.clear()
        self._stack.clear()

    def start(self, name: str) -> None:
        """Start timing a section."""
        if not self._enabled:
            return
        self._starts[name] = time.perf_counter()

        # Track parent-child relationship
        if name not in self._entries:
            parent = self._stack[-1] if self._stack else None
            self._entries[name] = TimingEntry(name=name, parent=parent)
            if parent and name not in self._entries[parent].children:
                self._entries[parent].children.append(name)

        self._stack.append(name)

    def stop(self, name: str) -> float:
        """Stop timing and return elapsed time."""
        if not self._enabled:
            return 0.0

        elapsed = time.perf_counter() - self._starts.get(name, time.perf_counter())

        if name in self._entries:
            self._entries[name].total += elapsed
            self._entries[name].count += 1

        if self._stack and self._stack[-1] == name:
            self._stack.pop()

        return elapsed

    @contextmanager
    def __call__(self, name: str):
        """Context manager for timing."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def time(self, name: str):
        """Alias for __call__ context manager."""
        return self(name)

    def report(self, min_ms: float = 0.1) -> None:
        """Print timing report."""
        if not self._entries:
            print("No timing data.")
            return

        # Find root entries (no parent)
        roots = [e for e in self._entries.values() if e.parent is None]

        print("\n" + "=" * 50)
        print("PROFILER REPORT")
        print("=" * 50)

        def print_entry(entry: TimingEntry, indent: int = 0):
            ms = entry.total * 1000
            if ms < min_ms:
                return
            prefix = "  " * indent
            avg_ms = entry.avg * 1000
            if entry.count > 1:
                print(
                    f"{prefix}{entry.name}: {ms:.1f}ms ({entry.count}x, avg {avg_ms:.1f}ms)"
                )
            else:
                print(f"{prefix}{entry.name}: {ms:.1f}ms")

            for child_name in entry.children:
                if child_name in self._entries:
                    print_entry(self._entries[child_name], indent + 1)

        for root in sorted(roots, key=lambda e: e.total, reverse=True):
            print_entry(root)

        total = sum(e.total for e in roots)
        print("-" * 50)
        print(f"Total: {total * 1000:.1f}ms")
        print("=" * 50 + "\n")

    def summary(self) -> dict[str, float]:
        """Return timing summary as dict (name -> total_ms)."""
        return {name: e.total * 1000 for name, e in self._entries.items()}

    def get(self, name: str) -> float:
        """Get total time for entry in ms."""
        if name in self._entries:
            return self._entries[name].total * 1000
        return 0.0


# Singleton instance
P = Profiler()
