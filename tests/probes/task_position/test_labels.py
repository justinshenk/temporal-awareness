"""Tests for task-position labelers."""

import pytest

from src.probes.task_position.labels import TaskPositionLabels, label_trace


def test_single_task_full_trace():
    labels = label_trace(trace_length=10, case_boundaries=[0])
    assert labels.task_index == [1] * 10
    assert labels.within_task_fraction == pytest.approx(
        [i / 10 for i in range(10)]
    )
    assert labels.tokens_until_boundary == list(range(10, 0, -1))
    assert labels.raw_token_position == list(range(10))
    assert labels.total_context_length == 10


def test_two_equal_tasks():
    labels = label_trace(trace_length=10, case_boundaries=[0, 5])
    assert labels.task_index == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    assert labels.within_task_fraction == pytest.approx(
        [0.0, 0.2, 0.4, 0.6, 0.8, 0.0, 0.2, 0.4, 0.6, 0.8]
    )
    assert labels.tokens_until_boundary == [5, 4, 3, 2, 1, 5, 4, 3, 2, 1]


def test_three_unequal_tasks():
    labels = label_trace(trace_length=12, case_boundaries=[0, 4, 10])
    assert labels.task_index == [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3]
    assert labels.tokens_until_boundary == [4, 3, 2, 1, 6, 5, 4, 3, 2, 1, 2, 1]


def test_final_token_has_tokens_until_boundary_of_one():
    """Documents the contract: the last token of the trace gets value 1, not 0."""
    labels = label_trace(trace_length=5, case_boundaries=[0])
    assert labels.tokens_until_boundary[-1] == 1


def test_boundaries_must_start_at_zero():
    with pytest.raises(ValueError, match="must start at 0"):
        label_trace(trace_length=10, case_boundaries=[2, 5])


def test_boundaries_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        label_trace(trace_length=10, case_boundaries=[0, 5, 3])


def test_duplicate_boundaries_rejected():
    """Zero-length cases would cause division-by-zero — must be rejected."""
    with pytest.raises(ValueError, match="strictly increasing"):
        label_trace(trace_length=10, case_boundaries=[0, 5, 5])


def test_last_boundary_must_be_less_than_trace_length():
    with pytest.raises(ValueError, match="must be <"):
        label_trace(trace_length=10, case_boundaries=[0, 10])


def test_empty_boundaries_raises():
    with pytest.raises(ValueError, match="non-empty"):
        label_trace(trace_length=10, case_boundaries=[])


def test_serialization_roundtrip():
    labels = label_trace(trace_length=6, case_boundaries=[0, 3])
    as_dict = labels.to_dict()
    assert as_dict["task_index"] == [1, 1, 1, 2, 2, 2]
    assert as_dict["total_context_length"] == 6
