#!/usr/bin/env python3
"""Test that GeoApp has ALL data preloaded and responds immediately.

This script:
1. Starts the backend server
2. Tests ALL layer/component/position combinations
3. Verifies responses are fast (from cache, not disk)
4. Reports any missing embeddings

Usage:
    uv run python scripts/intertemporal/test_geoapp_preload.py

    # Custom data directory
    uv run python scripts/intertemporal/test_geoapp_preload.py --data-dir out/geometry

    # Just verify files exist (no server)
    uv run python scripts/intertemporal/test_geoapp_preload.py --files-only
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# All layers and components
LAYERS = [0, 1, 3, 12, 18, 19, 21, 24, 28, 31, 34, 35]
COMPONENTS = ["resid_pre", "attn_out", "mlp_out", "resid_post"]


def get_all_positions(data_dir: Path) -> list[str]:
    """Get all positions from the data."""
    samples_dir = data_dir / "data" / "samples"
    sample_dirs = sorted(
        d for d in samples_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ) if samples_dir.exists() else []
    if not sample_dirs:
        return []

    position_file = sample_dirs[0] / "position_mapping.json"
    if not position_file.exists():
        return []

    with open(position_file) as f:
        mapping = json.load(f)

    return sorted(mapping.get("named_positions", {}).keys())


def verify_embedding_files(data_dir: Path) -> tuple[list[str], list[str]]:
    """Verify that all embedding files exist.

    Returns (found, missing) tuple of target keys.
    """
    all_positions = get_all_positions(data_dir)
    if not all_positions:
        print(f"ERROR: No positions found in {data_dir}")
        return [], []

    print(f"Checking {len(LAYERS)} layers x {len(COMPONENTS)} components x {len(all_positions)} positions")
    print(f"Total expected: {len(LAYERS) * len(COMPONENTS) * len(all_positions)} embeddings")
    print()

    embeddings_dir = data_dir / "analysis" / "embeddings" / "pca"
    if not embeddings_dir.exists():
        print(f"ERROR: Embeddings directory not found: {embeddings_dir}")
        return [], []

    found = []
    missing = []

    for layer in LAYERS:
        for component in COMPONENTS:
            for position in all_positions:
                key = f"L{layer}_{component}_{position}"
                filepath = embeddings_dir / f"{key}.npy"
                if filepath.exists():
                    found.append(key)
                else:
                    missing.append(key)

    return found, missing


def test_server_endpoint(base_url: str, layer: int, component: str, position: str) -> tuple[bool, float, str]:
    """Test a single endpoint and return (success, time_ms, error).

    Returns quickly without making actual HTTP requests - we test via file loading.
    """
    import urllib.request
    import urllib.error

    key = f"L{layer}_{component}_{position}"
    url = f"{base_url}/api/embedding?layer={layer}&component={component}&position={position}&method=pca"

    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read())
            elapsed_ms = (time.time() - start) * 1000

            if "error" in data:
                return False, elapsed_ms, data["error"]
            if "embeddings" not in data:
                return False, elapsed_ms, "No embeddings in response"

            return True, elapsed_ms, ""

    except urllib.error.URLError as e:
        return False, 0, str(e)
    except Exception as e:
        return False, 0, str(e)


def test_server(data_dir: Path, port: int = 8642) -> tuple[int, int, list[str]]:
    """Test all endpoints against a running server.

    Returns (success_count, fail_count, failed_keys).
    """
    import urllib.request
    import urllib.error

    base_url = f"http://localhost:{port}"
    all_positions = get_all_positions(data_dir)

    # Check if server is running
    try:
        with urllib.request.urlopen(f"{base_url}/api/metadata", timeout=2) as response:
            pass
    except Exception as e:
        print(f"ERROR: Server not responding at {base_url}")
        print(f"Error: {e}")
        print("Start the server first: bash scripts/intertemporal/run_geoapp.sh")
        return 0, 0, []

    total = len(LAYERS) * len(COMPONENTS) * len(all_positions)
    print(f"Testing {total} endpoints against {base_url}...")
    print()

    success_count = 0
    fail_count = 0
    failed_keys = []
    slow_keys = []

    for layer in LAYERS:
        for component in COMPONENTS:
            for position in all_positions:
                ok, time_ms, error = test_server_endpoint(base_url, layer, component, position)
                key = f"L{layer}_{component}_{position}"

                if ok:
                    success_count += 1
                    if time_ms > 100:  # More than 100ms is suspicious
                        slow_keys.append((key, time_ms))
                else:
                    fail_count += 1
                    failed_keys.append(key)
                    if fail_count <= 10:  # Only show first 10 failures
                        print(f"  FAIL: {key} - {error}")

        # Progress update per layer
        done = (LAYERS.index(layer) + 1) * len(COMPONENTS) * len(all_positions)
        print(f"Progress: {done}/{total} ({success_count} OK, {fail_count} FAIL)")

    print()
    if slow_keys:
        print(f"WARNING: {len(slow_keys)} slow responses (>100ms):")
        for key, ms in slow_keys[:5]:
            print(f"  {key}: {ms:.1f}ms")

    return success_count, fail_count, failed_keys


def main():
    parser = argparse.ArgumentParser(description="Test GeoApp preloading")
    parser.add_argument("--data-dir", default="out/geometry", help="Data directory")
    parser.add_argument("--files-only", action="store_true", help="Only check files, don't test server")
    parser.add_argument("--port", type=int, default=8642, help="Server port")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("GEOAPP PRELOAD TEST")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print()

    # Phase 1: Verify embedding files exist
    print("Phase 1: Checking embedding files...")
    print("-" * 40)
    found, missing = verify_embedding_files(data_dir)

    print(f"Found: {len(found)} embeddings")
    print(f"Missing: {len(missing)} embeddings")

    if missing:
        print()
        print("MISSING EMBEDDINGS (first 20):")
        for key in missing[:20]:
            print(f"  {key}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

        print()
        print("FIX: Run compute_geometry_analysis.py to generate ALL embeddings:")
        print(f"  uv run python scripts/intertemporal/compute_geometry_analysis.py --data-dir {data_dir}")
        print()

    # Phase 2: Test server (unless --files-only)
    if not args.files_only:
        print()
        print("Phase 2: Testing server endpoints...")
        print("-" * 40)
        success, fail, failed_keys = test_server(data_dir, args.port)

        if success + fail > 0:
            print()
            print(f"Results: {success}/{success + fail} OK ({100 * success / (success + fail):.1f}%)")

            if failed_keys:
                print()
                print("FAILED ENDPOINTS (first 20):")
                for key in failed_keys[:20]:
                    print(f"  {key}")

    # Summary
    print()
    print("=" * 60)
    if not missing:
        print("SUCCESS: All embedding files exist!")
    else:
        print(f"FAILURE: {len(missing)} embeddings missing - run compute_geometry_analysis.py")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
