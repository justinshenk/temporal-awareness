"""Tests for geoapp API routes."""

import pytest
from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Tests for API endpoints."""

    def test_get_config(self, test_client: TestClient):
        """Test GET /api/config."""
        response = test_client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert "layers" in data
        assert "components" in data
        assert "positions" in data
        assert "color_options" in data
        assert "position_labels" in data
        assert "rel_pos_counts" in data
        assert len(data["layers"]) == 3
        assert "resid_pre" in data["components"]

    def test_get_embedding_pca(self, test_client: TestClient):
        """Test GET /api/embedding/{layer}/{component}/{position} with PCA."""
        response = test_client.get("/api/embedding/0/resid_pre/response_choice?method=pca")
        assert response.status_code == 200
        data = response.json()
        assert "coordinates" in data
        assert "sample_indices" in data
        assert len(data["coordinates"]) == 10  # 10 samples
        assert "x" in data["coordinates"][0]
        assert "y" in data["coordinates"][0]
        assert "z" in data["coordinates"][0]

    def test_get_embedding_umap(self, test_client: TestClient):
        """Test GET /api/embedding with UMAP."""
        response = test_client.get("/api/embedding/0/resid_pre/response_choice?method=umap")
        assert response.status_code == 200
        data = response.json()
        assert "coordinates" in data

    def test_get_embedding_tsne(self, test_client: TestClient):
        """Test GET /api/embedding with t-SNE."""
        response = test_client.get("/api/embedding/0/resid_pre/response_choice?method=tsne")
        assert response.status_code == 200
        data = response.json()
        assert "coordinates" in data

    def test_get_embedding_time_horizon(self, test_client: TestClient):
        """Test GET /api/embedding for time_horizon position."""
        response = test_client.get("/api/embedding/0/resid_pre/time_horizon?method=pca")
        assert response.status_code == 200
        data = response.json()
        assert "coordinates" in data
        assert "sample_indices" in data
        # Only 7 samples have time_horizon
        assert len(data["coordinates"]) == 7
        assert len(data["sample_indices"]) == 7

    def test_get_embedding_invalid_layer(self, test_client: TestClient):
        """Test GET /api/embedding with invalid layer."""
        response = test_client.get("/api/embedding/999/resid_pre/response_choice")
        assert response.status_code == 404

    def test_get_embedding_invalid_position(self, test_client: TestClient):
        """Test GET /api/embedding with invalid position."""
        response = test_client.get("/api/embedding/0/resid_pre/invalid_position")
        assert response.status_code == 404

    def test_get_metadata(self, test_client: TestClient):
        """Test GET /api/metadata."""
        response = test_client.get("/api/metadata?color_by=time_horizon")
        assert response.status_code == 200
        data = response.json()
        assert "values" in data
        assert "dtype" in data
        assert data["dtype"] == "numeric"
        assert len(data["values"]) == 10

    def test_get_metadata_has_horizon(self, test_client: TestClient):
        """Test GET /api/metadata for has_horizon."""
        response = test_client.get("/api/metadata?color_by=has_horizon")
        assert response.status_code == 200
        data = response.json()
        assert "values" in data
        assert data["dtype"] == "boolean"
        # Boolean values
        assert all(isinstance(v, bool) for v in data["values"])

    def test_get_metadata_invalid_color_by(self, test_client: TestClient):
        """Test GET /api/metadata with invalid color_by."""
        response = test_client.get("/api/metadata?color_by=invalid_field")
        assert response.status_code == 400

    def test_get_sample(self, test_client: TestClient):
        """Test GET /api/sample/{idx}."""
        response = test_client.get("/api/sample/0")
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["text"] == "Sample 0 prompt text"
        assert "idx" in data
        assert data["idx"] == 0

    def test_get_sample_not_found(self, test_client: TestClient):
        """Test GET /api/sample/{idx} with invalid index."""
        response = test_client.get("/api/sample/9999")
        assert response.status_code == 404

    def test_get_metrics(self, test_client: TestClient):
        """Test GET /api/metrics/{layer}/{component}/{position}."""
        response = test_client.get("/api/metrics/0/resid_pre/response_choice")
        assert response.status_code == 200
        data = response.json()
        assert "layer" in data
        assert "component" in data
        assert "position" in data

    def test_get_heatmap(self, test_client: TestClient):
        """Test GET /api/heatmap/{component}."""
        response = test_client.get("/api/heatmap/resid_pre?metric=r2")
        assert response.status_code == 200
        data = response.json()
        assert "cells" in data
        assert "layers" in data
        assert "positions" in data

    def test_get_warmup_status(self, test_client: TestClient):
        """Test GET /api/warmup/status."""
        response = test_client.get("/api/warmup/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "is_running" in data["status"]

    def test_layer_trajectory(self, test_client: TestClient):
        """Test GET /api/trajectory/layers/{component}/{position}."""
        response = test_client.get("/api/trajectory/layers/resid_pre/response_choice")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "x_values" in data
        assert data["x_axis"] == "layer"

    def test_position_trajectory(self, test_client: TestClient):
        """Test GET /api/trajectory/positions/{layer}/{component}."""
        response = test_client.get("/api/trajectory/positions/0/resid_pre")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "x_values" in data
        assert data["x_axis"] == "position"


class TestAPIPerformance:
    """Performance tests for API."""

    def test_embedding_response_time(self, test_client: TestClient):
        """Test that embedding requests return reasonably fast after first load."""
        import time

        # First request (may compute)
        test_client.get("/api/embedding/0/resid_pre/response_choice?method=pca")

        # Second request (should be cached)
        start = time.time()
        response = test_client.get("/api/embedding/0/resid_pre/response_choice?method=pca")
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 0.5  # Should be fast when cached

    def test_multiple_layers_performance(self, test_client: TestClient):
        """Test loading embeddings for multiple layers."""
        import time

        # First pass to warm cache
        for layer in range(3):
            test_client.get(f"/api/embedding/{layer}/resid_pre/response_choice?method=pca")

        # Second pass should be fast
        start = time.time()
        for layer in range(3):
            response = test_client.get(f"/api/embedding/{layer}/resid_pre/response_choice?method=pca")
            assert response.status_code == 200
        elapsed = time.time() - start

        assert elapsed < 1.0  # All 3 layers should be fast when cached

    def test_config_response_time(self, test_client: TestClient):
        """Test that config endpoint responds quickly."""
        import time

        start = time.time()
        response = test_client.get("/api/config")
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 0.5  # Config should be fast
