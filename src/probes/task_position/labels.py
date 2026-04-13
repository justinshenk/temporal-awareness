"""Pure-function labelers for task-position targets.

Produces per-token labels for:
  - task_index: 1..N ordinal position of the current task in the trace
  - within_task_fraction: [0, 1] fractional progress through the current task
  - tokens_until_boundary: positive int, distance to the next task boundary

Labelers are pure functions on (trace_length, case_boundaries). No model required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.common.base_schema import BaseSchema


@dataclass
class TaskPositionLabels(BaseSchema):
    """Per-token task-position labels for one trace.

    All arrays have length equal to the trace's token count.
    """

    task_index: list[int] = field(default_factory=list)
    within_task_fraction: list[float] = field(default_factory=list)
    tokens_until_boundary: list[int] = field(default_factory=list)
    raw_token_position: list[int] = field(default_factory=list)
    total_context_length: int = 0


def label_trace(
    trace_length: int,
    case_boundaries: Sequence[int],
) -> TaskPositionLabels:
    """Produce per-token task-position labels.

    Args:
        trace_length: total number of tokens in the trace
        case_boundaries: sorted list of token indices where each case begins.
            Must start at 0 (first case begins at token 0). The sentinel
            boundary at `trace_length` is appended internally — callers must
            NOT include it.

    Returns:
        TaskPositionLabels with arrays of length `trace_length`.

    Raises:
        ValueError: if boundaries are not sorted, not starting at 0, or empty.
    """
    if not case_boundaries:
        raise ValueError("case_boundaries must be non-empty")
    if case_boundaries[0] != 0:
        raise ValueError(f"case_boundaries must start at 0, got {case_boundaries[0]}")
    if list(case_boundaries) != sorted(case_boundaries):
        raise ValueError("case_boundaries must be sorted ascending")
    if case_boundaries[-1] >= trace_length:
        raise ValueError(
            f"last boundary {case_boundaries[-1]} must be < trace_length {trace_length}"
        )

    boundaries = list(case_boundaries) + [trace_length]
    n_cases = len(case_boundaries)

    task_index = np.zeros(trace_length, dtype=np.int64)
    within_task_fraction = np.zeros(trace_length, dtype=np.float64)
    tokens_until_boundary = np.zeros(trace_length, dtype=np.int64)

    for i in range(n_cases):
        start = boundaries[i]
        end = boundaries[i + 1]
        case_length = end - start
        for t in range(start, end):
            task_index[t] = i + 1  # 1-indexed
            within_task_fraction[t] = (t - start) / case_length
            tokens_until_boundary[t] = end - t

    return TaskPositionLabels(
        task_index=task_index.tolist(),
        within_task_fraction=within_task_fraction.tolist(),
        tokens_until_boundary=tokens_until_boundary.tolist(),
        raw_token_position=list(range(trace_length)),
        total_context_length=trace_length,
    )
