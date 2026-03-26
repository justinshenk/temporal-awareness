"""Performance tests for geoapp server.

These tests verify that after warmup, API responses are near-instantaneous.
"""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ..data_loader import GeometryDataLoader
from ..server import create_app


@pytest.fixture
def real_data_loader():
    """Load actual data if available for realistic performance testing."""
    data_dir = Path("out/geometry")
    if not data_dir.exists():
        pytest.skip("Real data not available at out/geometry")
    return GeometryDataLoader(data_dir)


@pytest.fixture
def real_test_client(real_data_loader):
    """Create test client with real data, warmup enabled."""
    data_dir = Path("out/geometry")
    app = create_app(data_dir=data_dir, warmup=False)  # We'll manually warmup
    return TestClient(app)


class TestServerPerformance:
    """Performance tests for server API after warmup."""

    def test_embedding_response_after_warmup(self, real_data_loader, real_test_client):
        """Test that embedding requests are fast after warmup (cached)."""
        layer = real_data_loader.get_layers()[0]
        position = real_data_loader.get_positions()[0]

        # Warmup HTTP path by making initial requests (first request loads cache)
        real_test_client.get(f"/api/embedding/{layer}/resid_pre/{position}?method=pca")
        real_test_client.get(f"/api/embedding/{layer}/resid_pre/{position}?method=pca")

        # Now measure response time (cache is warm)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            response = real_test_client.get(
                f"/api/embedding/{layer}/resid_pre/{position}?method=pca"
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
            assert response.status_code == 200

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\nEmbedding response times (warmed): avg={avg_time:.1f}ms, min={min_time:.1f}ms, max={max_time:.1f}ms")

        # After warmup, cached requests should be very fast
        assert avg_time < 20, f"Average response time {avg_time:.1f}ms exceeds 20ms threshold"
        assert min_time < 5, f"Min response time {min_time:.1f}ms exceeds 5ms - cache not working"

    def test_metadata_response_time(self, real_test_client):
        """Test that metadata requests are fast."""
        # Warmup
        real_test_client.get("/api/metadata?color_by=time_horizon")

        times = []
        for _ in range(10):
            start = time.perf_counter()
            response = real_test_client.get("/api/metadata?color_by=time_horizon")
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
            assert response.status_code == 200

        avg_time = sum(times) / len(times)
        print(f"\nMetadata response times: avg={avg_time:.1f}ms")

        # Metadata should be very fast
        assert avg_time < 50, f"Average metadata time {avg_time:.1f}ms exceeds 50ms threshold"

    def test_config_response_time(self, real_test_client):
        """Test that config requests are instant."""
        # Warmup
        real_test_client.get("/api/config")

        times = []
        for _ in range(10):
            start = time.perf_counter()
            response = real_test_client.get("/api/config")
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
            assert response.status_code == 200

        avg_time = sum(times) / len(times)
        print(f"\nConfig response times: avg={avg_time:.1f}ms")

        # Config should be fast (has significant JSON to serialize)
        assert avg_time < 100, f"Average config time {avg_time:.1f}ms exceeds 100ms threshold"

    def test_all_layers_cached_performance(self, real_data_loader, real_test_client):
        """Test that all layers respond quickly when HTTP-warmed."""
        layers = real_data_loader.get_layers()
        position = real_data_loader.get_positions()[0]

        # Warmup ALL layers via HTTP (first pass loads cache)
        for layer in layers:
            real_test_client.get(f"/api/embedding/{layer}/resid_pre/{position}?method=pca")

        # Second pass measures cached performance
        layer_times = {}
        for layer in layers:
            start = time.perf_counter()
            response = real_test_client.get(
                f"/api/embedding/{layer}/resid_pre/{position}?method=pca"
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            layer_times[layer] = elapsed_ms
            assert response.status_code == 200

        avg_time = sum(layer_times.values()) / len(layer_times)
        max_time = max(layer_times.values())

        print(f"\nLayer response times (HTTP-warmed): avg={avg_time:.1f}ms, max={max_time:.1f}ms")
        print(f"Per-layer times: {layer_times}")

        # All layers should be very fast when cached via HTTP
        assert avg_time < 10, f"Average layer time {avg_time:.1f}ms exceeds 10ms threshold"
        assert max_time < 30, f"Slowest layer took {max_time:.1f}ms, exceeds 30ms threshold"


class TestDiskCachePerformance:
    """Test disk cache loading performance."""

    def test_disk_cache_load_time(self, real_data_loader):
        """Test that loading from disk cache is fast."""
        layers = real_data_loader.get_layers()
        positions = real_data_loader.get_positions()

        if not layers or not positions:
            pytest.skip("No data available")

        # Clear memory cache
        real_data_loader._pca_cache.clear()

        # Measure disk cache load time
        layer = layers[0]
        position = positions[0]

        start = time.perf_counter()
        result = real_data_loader.load_pca(layer, "resid_pre", position)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if result is not None:
            print(f"\nDisk cache load time: {elapsed_ms:.1f}ms for {result.shape} array")
            # Disk cache should load quickly
            assert elapsed_ms < 500, f"Disk cache load took {elapsed_ms:.1f}ms, exceeds 500ms"
        else:
            print(f"\nNo cached data found, computation took {elapsed_ms:.1f}ms")


def run_performance_benchmark():
    """Run a comprehensive performance benchmark and print results."""
    import gc

    data_dir = Path("out/geometry")
    if not data_dir.exists():
        print("ERROR: Data not found at out/geometry")
        return

    print("=" * 60)
    print("  PERFORMANCE BENCHMARK")
    print("=" * 60)

    loader = GeometryDataLoader(data_dir)
    app = create_app(data_dir=data_dir, warmup=False)
    client = TestClient(app)

    layers = loader.get_layers()
    positions = loader.get_positions()

    print(f"\nData: {len(layers)} layers, {len(positions)} positions")

    # 1. Warmup phase - load all PCA into memory
    print("\n--- WARMUP PHASE ---")
    warmup_start = time.perf_counter()
    for layer in layers:
        for pos in positions[:5]:  # First 5 positions for quick test
            loader.load_pca(layer, "resid_pre", pos)
    warmup_time = time.perf_counter() - warmup_start
    print(f"Warmup completed in {warmup_time:.1f}s")

    # Disable GC for accurate measurement
    gc.collect()
    gc.disable()

    # 2. API Response times (after warmup)
    print("\n--- API RESPONSE TIMES (10 iterations each) ---")

    # Config
    client.get("/api/config")  # Warmup
    times = []
    for _ in range(10):
        start = time.perf_counter()
        client.get("/api/config")
        times.append((time.perf_counter() - start) * 1000)
    print(f"Config:    avg={sum(times)/10:.1f}ms, min={min(times):.1f}ms, max={max(times):.1f}ms")

    # Metadata
    client.get("/api/metadata?color_by=time_horizon")  # Warmup
    times = []
    for _ in range(10):
        start = time.perf_counter()
        client.get("/api/metadata?color_by=time_horizon")
        times.append((time.perf_counter() - start) * 1000)
    print(f"Metadata:  avg={sum(times)/10:.1f}ms, min={min(times):.1f}ms, max={max(times):.1f}ms")

    # Embedding (cached) - test multiple layers
    layer = layers[0]
    position = positions[0]
    client.get(f"/api/embedding/{layer}/resid_pre/{position}?method=pca")  # Warmup
    times = []
    for _ in range(10):
        start = time.perf_counter()
        client.get(f"/api/embedding/{layer}/resid_pre/{position}?method=pca")
        times.append((time.perf_counter() - start) * 1000)
    print(f"Embedding: avg={sum(times)/10:.1f}ms, min={min(times):.1f}ms, max={max(times):.1f}ms")

    # Layer switching (cached)
    print("\n--- LAYER SWITCHING (all layers, cached) ---")
    layer_times = []
    for layer in layers:
        loader.load_pca(layer, "resid_pre", position)  # Ensure cached
        start = time.perf_counter()
        client.get(f"/api/embedding/{layer}/resid_pre/{position}?method=pca")
        layer_times.append((time.perf_counter() - start) * 1000)
    print(f"All {len(layers)} layers: avg={sum(layer_times)/len(layer_times):.1f}ms, max={max(layer_times):.1f}ms")

    gc.enable()

    # 3. Performance assessment
    print("\n" + "=" * 60)
    embedding_avg = sum(times) / len(times)
    if embedding_avg < 10:
        print("  RESULT: EXCELLENT - Embedding responses under 10ms")
    elif embedding_avg < 50:
        print("  RESULT: GOOD - Embedding responses under 50ms")
    elif embedding_avg < 100:
        print("  RESULT: OK - Embedding responses under 100ms")
    else:
        print("  RESULT: SLOW - Embedding responses over 100ms, needs optimization")
    print("=" * 60)


if __name__ == "__main__":
    run_performance_benchmark()
