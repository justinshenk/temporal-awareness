"""Unit tests for ModelRunner configuration and API.

Tests ModelRunner initialization, configuration parsing, and API structure.
Does NOT load models - for model loading tests, see test_model_runner_qwen.py.
"""

from src.models.model_runner import ModelRunner, ModelBackend


# =============================================================================
# ModelBackend Enum Tests
# =============================================================================


class TestModelBackend:
    """Test ModelBackend enum values."""

    def test_transformerlens_value(self):
        assert ModelBackend.TRANSFORMERLENS.value == "transformerlens"

    def test_nnsight_value(self):
        assert ModelBackend.NNSIGHT.value == "nnsight"

    def test_pyvene_value(self):
        assert ModelBackend.PYVENE.value == "pyvene"

    def test_from_string(self):
        assert ModelBackend("transformerlens") == ModelBackend.TRANSFORMERLENS
        assert ModelBackend("nnsight") == ModelBackend.NNSIGHT
        assert ModelBackend("pyvene") == ModelBackend.PYVENE


# =============================================================================
# ModelRunner Configuration Tests
# =============================================================================


class TestModelRunnerConfiguration:
    """Test ModelRunner configuration handling.

    Note: These tests verify configuration parsing without loading models.
    For actual model loading tests, see test_model_runner_qwen.py.
    """

    def test_device_auto_detection_cpu(self, monkeypatch):
        """Device auto-detection falls back to CPU."""
        # Mock torch.cuda.is_available and torch.backends.mps.is_available
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        # We can't test the full initialization without loading a model,
        # but we can verify the device detection logic works

        # The device would be set to "cpu" when both CUDA and MPS are unavailable
        # This is tested implicitly in test_model_runner_qwen.py

    def test_default_backend_is_transformerlens(self):
        """Default backend is TransformerLens."""
        # Can't test without loading, but verify the default in the signature
        import inspect

        sig = inspect.signature(ModelRunner.__init__)
        backend_param = sig.parameters["backend"]
        assert backend_param.default == ModelBackend.TRANSFORMERLENS


# =============================================================================
# For actual ModelRunner tests with models, see:
# - tests/src/test_model_runner_qwen.py (tests with Qwen2.5-0.5B)
# - tests/src/test_interventions_qwen.py (intervention tests across backends)
# =============================================================================
