"""Pytest fixtures for geoapp tests."""

import json
import tempfile
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
    targets_dir = data_dir / "targets"
    targets_dir.mkdir()

    # Create sample data
    n_samples = 10
    d_model = 64
    n_layers = 3

    # Create samples.json
    samples = []
    for i in range(n_samples):
        has_horizon = i < 7  # 7 with horizon, 3 without
        sample = {
            "text": f"Sample {i} prompt text",
            "prompt": {
                "preference_pair": {
                    "short_term": {"reward": {"value": 10 + i}, "time": {"value": 1, "unit": "days"}},
                    "long_term": {"reward": {"value": 20 + i}, "time": {"value": 1, "unit": "years"}},
                },
            },
        }
        if has_horizon:
            sample["prompt"]["time_horizon"] = {"value": 1 + i, "unit": "months"}
        samples.append(sample)

    with open(data_dir / "samples.json", "w") as f:
        json.dump(samples, f)

    # Create choices.json
    choices = []
    for i in range(n_samples):
        choices.append({
            "chose_long_term": i % 2 == 0,
            "chosen_time_months": 1 + i,
            "chosen_reward": 15 + i,
        })
    with open(data_dir / "choices.json", "w") as f:
        json.dump(choices, f)

    # Create metadata.json
    metadata = {"model_name": "test-model"}
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Create position mapping
    position_mapping = {"mappings": []}
    for i in range(n_samples):
        positions = [
            {"abs_pos": 0, "format_pos": "time_horizon", "rel_pos": 0},
            {"abs_pos": 1, "format_pos": "time_horizon", "rel_pos": 1},
            {"abs_pos": 2, "format_pos": "response_choice", "rel_pos": 0},
        ]
        # Only include time_horizon positions for samples with horizon
        if i >= 7:
            positions = [{"abs_pos": 2, "format_pos": "response_choice", "rel_pos": 0}]

        position_mapping["mappings"].append({
            "sample_idx": i,
            "positions": positions,
        })
    with open(data_dir / "sample_position_mapping.json", "w") as f:
        json.dump(position_mapping, f)

    # Create activation files
    for sample_idx in range(n_samples):
        sample_dir = targets_dir / f"sample_{sample_idx}"
        sample_dir.mkdir()

        for layer in range(n_layers):
            for component in ["resid_pre", "resid_post", "attn_out", "mlp_out"]:
                # response_choice at position 2
                act = np.random.randn(d_model).astype(np.float32)
                np.save(sample_dir / f"L{layer}_{component}_2.npy", act)

                # time_horizon at positions 0, 1 (only for samples with horizon)
                if sample_idx < 7:
                    for pos in [0, 1]:
                        act = np.random.randn(d_model).astype(np.float32)
                        np.save(sample_dir / f"L{layer}_{component}_{pos}.npy", act)

    return tmp_path


@pytest.fixture
def loader(sample_data_dir: Path) -> GeometryDataLoader:
    """Create a GeometryDataLoader with sample data."""
    return GeometryDataLoader(sample_data_dir)


@pytest.fixture
def test_client(sample_data_dir: Path) -> TestClient:
    """Create a FastAPI test client."""
    app = create_app(data_dir=sample_data_dir, warmup=False)
    return TestClient(app)
