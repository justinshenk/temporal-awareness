"""Core logging primitives for console output."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO


class TeeStream:
    """Stream that writes to both original stream and a file."""

    def __init__(self, original: TextIO, log_file: TextIO):
        self.original = original
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.original.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.original.flush()
        self.log_file.flush()

    def fileno(self) -> int:
        return self.original.fileno()

    def isatty(self) -> bool:
        return self.original.isatty()


# Global state
_log_file: TextIO | None = None
_original_stdout: TextIO | None = None
_original_stderr: TextIO | None = None


def set_log_file(path: Path | str | None) -> None:
    """Set log file path. Redirects stdout/stderr to also write to file."""
    global _log_file, _original_stdout, _original_stderr

    # Restore original streams first
    if _original_stdout is not None:
        sys.stdout = _original_stdout
        _original_stdout = None
    if _original_stderr is not None:
        sys.stderr = _original_stderr
        _original_stderr = None
    if _log_file is not None:
        _log_file.close()
        _log_file = None

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _log_file = open(path, "w", encoding="utf-8")

        # Tee stdout and stderr to the log file
        _original_stdout = sys.stdout
        _original_stderr = sys.stderr
        sys.stdout = TeeStream(_original_stdout, _log_file)
        sys.stderr = TeeStream(_original_stderr, _log_file)


def close_log_file() -> None:
    """Close the log file and restore original streams."""
    global _log_file, _original_stdout, _original_stderr

    if _original_stdout is not None:
        sys.stdout = _original_stdout
        _original_stdout = None
    if _original_stderr is not None:
        sys.stderr = _original_stderr
        _original_stderr = None
    if _log_file is not None:
        _log_file.close()
        _log_file = None


def log(msg: str = "", end: str = "\n", gap: int = 0) -> None:
    """Print with immediate flush."""
    for _ in range(gap):
        print(flush=True)
    print(msg, end=end, flush=True)


def log_flush() -> None:
    """Flush stdout."""
    sys.stdout.flush()


def log_progress(current: int, total: int, prefix: str = "") -> None:
    """Print progress indicator (overwrites line)."""
    log(f"{prefix}{current}/{total}", end="\r")


def log_done(msg: str = "") -> None:
    """Print completion message (clears progress line)."""
    log(msg)
