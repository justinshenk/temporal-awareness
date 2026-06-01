"""Tests for refusal-direction computation and signed projection."""

import numpy as np
import pytest

from src.probes.safety.refusal_direction import project_onto, refusal_direction


def test_direction_points_harmful_minus_harmless():
    harmful = np.array([[2.0, 0.0], [4.0, 0.0]])   # mean (3, 0)
    harmless = np.array([[0.0, 0.0], [0.0, 0.0]])  # mean (0, 0)
    r = refusal_direction(harmful, harmless)
    assert np.allclose(r, np.array([1.0, 0.0]))  # unit, +x


def test_direction_is_unit():
    rng = np.random.default_rng(0)
    r = refusal_direction(rng.normal(size=(20, 16)) + 3, rng.normal(size=(20, 16)))
    assert np.linalg.norm(r) == pytest.approx(1.0)


def test_direction_zero_raises():
    x = np.ones((4, 3))
    with pytest.raises(ValueError, match="zero"):
        refusal_direction(x, x.copy())


def test_projection_sign_and_magnitude():
    r = np.array([1.0, 0.0])
    shifts = np.array([[2.0, 9.0], [-3.0, 1.0], [0.0, 5.0]])
    proj = project_onto(shifts, r)
    # projection onto unit x-axis = the x-component
    assert np.allclose(proj, np.array([2.0, -3.0, 0.0]))


def test_projection_toward_compliance_is_negative():
    # refusal direction +x; a shift pointing -x means "less refusal"
    r = np.array([1.0, 0.0, 0.0])
    shift = np.array([[-4.0, 1.0, 0.0]])
    assert project_onto(shift, r)[0] < 0


def test_projection_zero_direction_raises():
    with pytest.raises(ValueError, match="zero direction"):
        project_onto(np.ones((2, 3)), np.zeros(3))
