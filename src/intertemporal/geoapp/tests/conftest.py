"""Pytest fixtures for geoapp tests."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ..data_loader import GeometryDataLoader
from ..server import create_app


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create a minimal sample data directory for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    samples_dir = data_dir / "samples"
    samples_dir.mkdir()

    # Create sample data
    n_samples = 10
    d_model = 64
    n_layers = 3

    # Create metadata.json
    metadata = {"n_samples": n_samples, "compressed": False, "model_name": "test-model"}
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Create per-sample files
    for sample_idx in range(n_samples):
        sample_dir = samples_dir / f"sample_{sample_idx}"
        sample_dir.mkdir()

        has_horizon = sample_idx < 7  # 7 with horizon, 3 without

        # Create prompt_sample.json
        prompt_sample = {
            "text": f"Sample {sample_idx} prompt text",
            "formatting_id": "default",
            "prompt": {
                "preference_pair": {
                    "short_term": {"reward": {"value": 10 + sample_idx}, "time": {"value": 1, "unit": "days"}},
                    "long_term": {"reward": {"value": 20 + sample_idx}, "time": {"value": 1, "unit": "years"}},
                },
            },
        }
        if has_horizon:
            prompt_sample["prompt"]["time_horizon"] = {"value": 1 + sample_idx, "unit": "months"}
        with open(sample_dir / "prompt_sample.json", "w") as f:
            json.dump(prompt_sample, f)

        # Create choice.json
        choice = {
            "time_horizon_months": (1 + sample_idx) if has_horizon else None,
            "chose_long_term": sample_idx % 2 == 0,
            "chosen_time_months": 1 + sample_idx,
            "chosen_reward": 15 + sample_idx,
            "choice_prob": 0.8,
        }
        with open(sample_dir / "choice.json", "w") as f:
            json.dump(choice, f)

        # Create position_mapping.json
        positions = [
            {"abs_pos": 0, "format_pos": "time_horizon", "rel_pos": 0},
            {"abs_pos": 1, "format_pos": "time_horizon", "rel_pos": 1},
            {"abs_pos": 2, "format_pos": "response_choice", "rel_pos": 0},
        ]
        # Only include time_horizon positions for samples with horizon
        if not has_horizon:
            positions = [{"abs_pos": 2, "format_pos": "response_choice", "rel_pos": 0}]

        position_mapping = {
            "sample_idx": sample_idx,
            "positions": positions,
        }
        with open(sample_dir / "position_mapping.json", "w") as f:
            json.dump(position_mapping, f)

        # Create activation files in layer subdirectories
        # Format: sample_dir/L{layer}/{component}_{abs_pos}.npy
        for layer in range(n_layers):
            layer_dir = sample_dir / f"L{layer}"
            layer_dir.mkdir()
            for component in ["resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"]:
                # response_choice at position 2
                act = np.random.randn(d_model).astype(np.float32)
                np.save(layer_dir / f"{component}_2.npy", act)

                # time_horizon at positions 0, 1 (only for samples with horizon)
                if has_horizon:
                    for pos in [0, 1]:
                        act = np.random.randn(d_model).astype(np.float32)
                        np.save(layer_dir / f"{component}_{pos}.npy", act)

    return tmp_path


@pytest.fixture
def loader(sample_data_dir: Path) -> GeometryDataLoader:
    """Create a GeometryDataLoader with sample data."""
    return GeometryDataLoader(sample_data_dir)


@pytest.fixture
def test_client(sample_data_dir: Path) -> TestClient:
    """Create a FastAPI test client."""
    app = create_app(data_dir=sample_data_dir)
    return TestClient(app)
