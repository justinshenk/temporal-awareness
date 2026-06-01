"""Tests for softmax entropy."""

import numpy as np
import pytest

from src.probes.safety.distribution_metrics import softmax_entropy


def test_uniform_is_log_n():
    assert softmax_entropy(np.zeros(4)) == pytest.approx(np.log(4))


def test_one_hot_is_zero():
    logits = np.array([100.0, 0.0, 0.0])
    assert softmax_entropy(logits) == pytest.approx(0.0, abs=1e-9)


def test_entropy_decreases_with_peakedness():
    flat = softmax_entropy(np.array([1.0, 1.0, 1.0]))
    peaked = softmax_entropy(np.array([5.0, 1.0, 1.0]))
    assert peaked < flat
