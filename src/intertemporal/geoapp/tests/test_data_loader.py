"""Tests for GeometryDataLoader."""

import numpy as np
from pathlib import Path

from ..data_loader import GeometryDataLoader


class TestGeometryDataLoader:
    """Tests for GeometryDataLoader."""

    def test_init(self, loader: GeometryDataLoader):
        """Test loader initializes correctly."""
        assert loader.n_samples == 10
        assert len(loader.get_layers()) == 3
        assert "response_choice" in loader.get_positions()
        assert "time_horizon" in loader.get_positions()

    def test_get_sample_info(self, loader: GeometryDataLoader):
        """Test getting sample info."""
        info = loader.get_sample_info(0)
        assert "prompt" in info
        assert "text" in info
        assert info["text"] == "Sample 0 prompt text"

    def test_get_sample_text(self, loader: GeometryDataLoader):
        """Test getting sample text."""
        text = loader.get_sample_text(0)
        assert text == "Sample 0 prompt text"

    def test_load_activations(self, loader: GeometryDataLoader):
        """Test loading activations."""
        activations = loader.load_activations(0, "resid_pre", "response_choice")
        assert activations is not None
        assert activations.shape == (10, 64)  # 10 samples, 64 d_model

    def test_load_activations_time_horizon(self, loader: GeometryDataLoader):
        """Test loading time_horizon activations (only samples with horizon)."""
        activations = loader.load_activations(0, "resid_pre", "time_horizon")
        assert activations is not None
        assert activations.shape == (7, 64)  # Only 7 samples have time_horizon

    def test_get_valid_sample_indices(self, loader: GeometryDataLoader):
        """Test getting valid sample indices."""
        indices = loader.get_valid_sample_indices(0, "resid_pre", "response_choice")
        assert len(indices) == 10

        indices_th = loader.get_valid_sample_indices(0, "resid_pre", "time_horizon")
        assert len(indices_th) == 7

    def test_load_pca(self, loader: GeometryDataLoader):
        """Test PCA embedding."""
        embedding = loader.load_pca(0, "resid_pre", "response_choice", n_components=3)
        assert embedding is not None
        assert embedding.shape[1] == 3

    def test_load_pca_cached(self, loader: GeometryDataLoader):
        """Test PCA caching."""
        # First call computes
        emb1 = loader.load_pca(0, "resid_pre", "response_choice", n_components=3)
        # Second call should return cached
        emb2 = loader.load_pca(0, "resid_pre", "response_choice", n_components=3)
        assert emb1 is emb2  # Same object (cached)

    def test_load_umap(self, loader: GeometryDataLoader):
        """Test UMAP embedding."""
        embedding = loader.load_umap(0, "resid_pre", "response_choice", n_components=2)
        assert embedding is not None
        assert embedding.shape[1] == 2

    def test_load_tsne(self, loader: GeometryDataLoader):
        """Test t-SNE embedding."""
        embedding = loader.load_tsne(0, "resid_pre", "response_choice", n_components=2)
        assert embedding is not None
        assert embedding.shape[1] == 2

    def test_get_sample_metadata_time_horizon(self, loader: GeometryDataLoader):
        """Test getting time horizon metadata."""
        metadata = loader.get_sample_metadata("time_horizon")
        assert len(metadata) == 10
        # First 7 have horizons (converted to months)
        assert metadata[0] > 0
        # Last 3 don't have horizons
        assert metadata[7] == -1

    def test_get_sample_metadata_has_horizon(self, loader: GeometryDataLoader):
        """Test getting has_horizon metadata."""
        metadata = loader.get_sample_metadata("has_horizon")
        assert len(metadata) == 10
        assert metadata[:7].all()  # First 7 have horizons
        assert not metadata[7:].any()  # Last 3 don't

    def test_get_sample_metadata_chosen_time(self, loader: GeometryDataLoader):
        """Test getting chosen_time metadata."""
        metadata = loader.get_sample_metadata("chosen_time")
        assert len(metadata) == 10

    def test_get_no_horizon_mask(self, loader: GeometryDataLoader):
        """Test getting no-horizon mask."""
        mask = loader.get_no_horizon_mask()
        assert len(mask) == 10
        assert not mask[:7].any()  # First 7 have horizons
        assert mask[7:].all()  # Last 3 don't

    def test_get_rel_pos_counts(self, loader: GeometryDataLoader):
        """Test getting relative position counts."""
        counts = loader.get_rel_pos_counts()
        assert "time_horizon" in counts
        assert counts["time_horizon"] == 2  # rel_pos 0 and 1
        assert "response_choice" in counts
        assert counts["response_choice"] == 1

    def test_parse_position_combined(self, loader: GeometryDataLoader):
        """Test parsing combined position."""
        format_pos, rel_pos = loader._parse_position("time_horizon")
        assert format_pos == "time_horizon"
        assert rel_pos is None

    def test_parse_position_specific(self, loader: GeometryDataLoader):
        """Test parsing specific rel_pos position."""
        format_pos, rel_pos = loader._parse_position("time_horizon:0")
        assert format_pos == "time_horizon"
        assert rel_pos == 0

    def test_load_activations_specific_rel_pos(self, loader: GeometryDataLoader):
        """Test loading activations for specific rel_pos."""
        # Combined (averages rel_pos 0 and 1)
        act_combined = loader.load_activations(0, "resid_pre", "time_horizon")
        # Specific rel_pos 0
        act_specific = loader.load_activations(0, "resid_pre", "time_horizon:0")

        assert act_combined is not None
        assert act_specific is not None
        # Same number of samples
        assert act_combined.shape[0] == act_specific.shape[0]

    def test_get_model_name(self, loader: GeometryDataLoader):
        """Test getting model name."""
        assert loader.get_model_name() == "test-model"

    def test_get_color_options(self, loader: GeometryDataLoader):
        """Test getting color options."""
        options = loader.get_color_options()
        assert "time_horizon" in options
        assert "chosen_time" in options
        assert "has_horizon" in options

    def test_get_position_labels(self, loader: GeometryDataLoader):
        """Test getting position labels."""
        labels = loader.get_position_labels()
        assert "time_horizon" in labels
        assert labels["time_horizon"] == "Time Horizon"

    def test_disk_cache_pca(self, loader: GeometryDataLoader, sample_data_dir: Path):
        """Test disk caching for PCA."""
        # Load once to create cache
        emb1 = loader.load_pca(0, "resid_pre", "response_choice", n_components=3)
        cache_path = loader._get_cache_path("pca", 0, "resid_pre", "response_choice")
        assert cache_path.exists()

        # Create new loader and load from disk cache
        loader2 = GeometryDataLoader(sample_data_dir)
        emb2 = loader2.load_pca(0, "resid_pre", "response_choice", n_components=3)
        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_load_all_layer_embeddings(self, loader: GeometryDataLoader):
        """Test loading embeddings for all layers."""
        embeddings = loader.load_all_layer_embeddings(
            "resid_pre", "response_choice", method="pca", n_components=3
        )
        assert len(embeddings) == 3  # 3 layers
        for layer, emb in embeddings.items():
            assert emb.shape[1] == 3


class TestDerivedFields:
    """Tests for derived color fields."""

    def test_matches_largest_reward(self, loader: GeometryDataLoader):
        """Test matches_largest_reward computation."""
        # All samples have larger long-term reward (20+i > 10+i)
        # Even samples chose long-term, odd samples chose short-term
        info_even = loader.get_sample_info(0)
        assert info_even.get("matches_largest_reward") is True

        info_odd = loader.get_sample_info(1)
        assert info_odd.get("matches_largest_reward") is False

    def test_matches_rational(self, loader: GeometryDataLoader):
        """Test matches_rational computation."""
        # Same as matches_largest_reward when rewards differ
        metadata = loader.get_sample_metadata("matches_rational")
        assert len(metadata) == 10
