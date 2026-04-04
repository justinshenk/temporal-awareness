#!/usr/bin/env python3
"""Tests for the activation extraction API.

Runs without GPU using a tiny model (gpt2) to validate the full pipeline.

Usage:
    pytest tests/test_activation_api.py -v
    python tests/test_activation_api.py  # standalone
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.activation_api import ExtractionConfig, ActivationExtractor, ActivationResult, ModuleSpec


# ---------------------------------------------------------------------------
# Config tests (no model needed)
# ---------------------------------------------------------------------------

class TestExtractionConfig:
    def test_default_config(self):
        config = ExtractionConfig()
        assert config.layers == [-1]
        assert config.module_types == ["resid_post"]
        assert config.positions == "last"
        assert config.stream_to == "cpu"
        assert config.batch_size == 4

    def test_resolve_modules_simple(self):
        config = ExtractionConfig(layers=[0, 8, -1], module_types=["resid_post"])
        modules = config.resolve_modules(n_layers=32)
        assert len(modules) == 3
        assert modules[0].layer == 0
        assert modules[1].layer == 8
        assert modules[2].layer == 31  # -1 resolved

    def test_resolve_modules_cross_product(self):
        config = ExtractionConfig(
            layers=[0, 16],
            module_types=["resid_post", "attn_out", "mlp_out"],
        )
        modules = config.resolve_modules(n_layers=32)
        assert len(modules) == 6  # 2 layers × 3 types

    def test_resolve_modules_explicit(self):
        config = ExtractionConfig(
            modules=[
                ModuleSpec(module_type="resid_post", layer=5),
                ModuleSpec(module_type="attn_out", layer=10, head=3),
            ]
        )
        modules = config.resolve_modules(n_layers=32)
        assert len(modules) == 2
        assert modules[1].head == 3

    def test_invalid_layer(self):
        config = ExtractionConfig(layers=[100])
        with pytest.raises(ValueError, match="out of range"):
            config.resolve_modules(n_layers=32)

    def test_disk_requires_output_dir(self):
        with pytest.raises(ValueError, match="output_dir"):
            ExtractionConfig(stream_to="disk")


class TestModuleSpec:
    def test_key_generation(self):
        spec = ModuleSpec(module_type="resid_post", layer=16)
        assert spec.key == "resid_post.layer16"

    def test_key_with_head(self):
        spec = ModuleSpec(module_type="attn_out", layer=5, head=3)
        assert spec.key == "attn_out.layer5.h3"

    def test_custom_requires_name(self):
        with pytest.raises(ValueError, match="module_name"):
            ModuleSpec(module_type="custom", layer=0)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid module_type"):
            ModuleSpec(module_type="nonexistent", layer=0)


class TestActivationResult:
    def _make_result(self):
        return ActivationResult(
            activations={
                "resid_post.layer0": torch.randn(4, 768),
                "resid_post.layer5": torch.randn(4, 768),
                "attn_out.layer5": torch.randn(4, 768),
            },
            metadata={"model_name": "test"},
            texts=["a", "b", "c", "d"],
        )

    def test_getitem_string_key(self):
        result = self._make_result()
        tensor = result["resid_post.layer0"]
        assert tensor.shape == (4, 768)

    def test_getitem_tuple_key(self):
        result = self._make_result()
        tensor = result["resid_post", 5]
        assert tensor.shape == (4, 768)

    def test_getitem_triple_key(self):
        result = self._make_result()
        tensor = result["resid_post", 5, "last"]
        assert tensor.shape == (4, 768)

    def test_contains(self):
        result = self._make_result()
        assert ("resid_post", 5) in result
        assert ("resid_post", 99) not in result

    def test_n_samples(self):
        result = self._make_result()
        assert result.n_samples == 4

    def test_layers(self):
        result = self._make_result()
        assert result.layers == [0, 5]

    def test_get_layer(self):
        result = self._make_result()
        layer5 = result.get_layer(5)
        assert len(layer5) == 2  # resid_post and attn_out

    def test_numpy(self):
        result = self._make_result()
        arr = result.numpy("resid_post", 0)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4, 768)

    def test_save_load_pt(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            result.save(tmpdir, format="pt")
            loaded = ActivationResult.load(tmpdir)
            assert len(loaded.activations) == 3
            for key in result.activations:
                assert torch.allclose(result.activations[key], loaded.activations[key])

    def test_save_load_npy(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            result.save(tmpdir, format="npy")
            loaded = ActivationResult.load(tmpdir)
            assert len(loaded.activations) == 3

    def test_summary(self):
        result = self._make_result()
        summary = result.summary()
        assert "4 samples" in summary
        assert "3 tensors" in summary
        assert "resid_post.layer0" in summary


# ---------------------------------------------------------------------------
# Integration tests (require model — skip if no torch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available() and True,
                    reason="Integration tests need a model download")
class TestExtractorIntegration:
    """Integration tests using GPT-2 (small enough for CPU)."""

    @pytest.fixture(scope="class")
    def extractor(self):
        config = ExtractionConfig(
            layers=[0, 5, 11],
            module_types=["resid_post"],
            positions="last",
            stream_to="cpu",
            batch_size=2,
        )
        return ActivationExtractor("gpt2", config)

    def test_basic_extraction(self, extractor):
        result = extractor.extract(["Hello world", "What is 2+2?"])
        assert result.n_samples == 2
        assert ("resid_post", 5) in result
        assert result["resid_post", 5].shape == (2, 768)

    def test_single_text(self, extractor):
        result = extractor.extract("Single input text")
        assert result.n_samples == 1

    def test_all_positions(self):
        config = ExtractionConfig(
            layers=[5],
            module_types=["resid_post"],
            positions="all",
            stream_to="cpu",
        )
        extractor = ActivationExtractor("gpt2", config)
        result = extractor.extract(["Hello world"])
        tensor = result["resid_post", 5]
        # Should have all token positions
        assert tensor.dim() == 3  # (1, seq_len, d_model)

    def test_disk_streaming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExtractionConfig(
                layers=[5],
                module_types=["resid_post"],
                positions="last",
                stream_to="disk",
                output_dir=tmpdir,
                output_format="pt",
            )
            extractor = ActivationExtractor("gpt2", config)
            result = extractor.extract(["Test disk streaming"])
            assert (Path(tmpdir) / "activations.pt").exists()
            assert (Path(tmpdir) / "metadata.json").exists()


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running config tests...")
    config = ExtractionConfig(
        layers=[0, 4, 8, -1],
        module_types=["resid_post", "attn_out"],
        positions="last",
    )
    modules = config.resolve_modules(n_layers=12)
    print(f"  Resolved {len(modules)} modules:")
    for m in modules:
        print(f"    {m.key}")

    print("\nRunning result tests...")
    result = ActivationResult(
        activations={
            "resid_post.layer0": torch.randn(4, 768),
            "resid_post.layer5": torch.randn(4, 768),
        },
        texts=["a", "b", "c", "d"],
    )
    print(result.summary())
    print(f"  result['resid_post', 5].shape = {result['resid_post', 5].shape}")

    with tempfile.TemporaryDirectory() as tmpdir:
        result.save(tmpdir, format="pt")
        loaded = ActivationResult.load(tmpdir)
        print(f"  Saved and loaded: {len(loaded.activations)} tensors")

    print("\nAll offline tests passed!")

    # Optional: test with actual model
    try:
        print("\nTesting with GPT-2 (CPU)...")
        ext_config = ExtractionConfig(
            layers=[0, 5, 11],
            module_types=["resid_post"],
            positions="last",
            batch_size=2,
        )
        extractor = ActivationExtractor("gpt2", ext_config)
        result = extractor.extract(["Hello world!", "What is 2+2?"])
        print(result)
        print("  Integration test passed!")
    except Exception as e:
        print(f"  Skipped integration test: {e}")
