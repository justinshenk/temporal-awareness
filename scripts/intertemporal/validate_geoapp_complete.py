#!/usr/bin/env python3
"""Complete validation of GeoApp precomputed data.

Creates /tmp/validation.md with detailed tables showing:
- 3D embeddings: layer × component × position × method with status/delay
- 1D x Layer trajectories: component × position with status
- 1D x Position trajectories: layer × component with status
- Server response times for all endpoints

Usage:
    uv run python scripts/intertemporal/validate_geoapp_complete.py

    # Test server too (requires running server)
    uv run python scripts/intertemporal/validate_geoapp_complete.py --test-server

    # Custom data directory
    uv run python scripts/intertemporal/validate_geoapp_complete.py --data-dir out/geometry
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Methods to check
METHODS = ["pca"]


def load_summary(data_dir: Path) -> dict | None:
    """Load summary.json if it exists."""
    summary_path = data_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def get_config_from_summary(data_dir: Path) -> tuple[list[int], list[str], list[str]]:
    """Get layers, components, positions from summary.json.

    Falls back to hardcoded defaults if summary.json doesn't exist.
    """
    summary = load_summary(data_dir)

    if summary:
        layers = summary.get("layers", [0, 1, 3, 12, 18, 19, 21, 24, 28, 31, 34, 35])
        components = summary.get("components", ["resid_pre", "attn_out", "mlp_out", "resid_post"])
        positions = summary.get("positions", [])
        return layers, components, positions

    # Fallback to defaults
    from src.intertemporal.common.semantic_positions import (
        PROMPT_POSITIONS,
        RESPONSE_POSITIONS,
    )
    return (
        [0, 1, 3, 12, 18, 19, 21, 24, 28, 31, 34, 35],
        ["resid_pre", "attn_out", "mlp_out", "resid_post"],
        PROMPT_POSITIONS + RESPONSE_POSITIONS,
    )


def get_all_positions(data_dir: Path) -> list[str]:
    """Get all positions that should have activation data.

    Reads from summary.json if available, otherwise falls back to semantic_positions.
    """
    _, _, positions = get_config_from_summary(data_dir)
    return positions


def validate_3d_embeddings(
    data_dir: Path,
    layers: list[int],
    components: list[str],
    positions: list[str],
) -> dict:
    """Validate all 3D embedding files exist.

    Checks multiple path structures for backwards compatibility:
    1. analysis/embeddings/{method}/L{layer}_{component}_{position}.npy (new flat)
    2. analysis/embeddings/L{layer}_{component}_{position}/{method}_embedding.npy (streaming analysis)
    3. cache/{method}/L{layer}_{component}_{position}.npy (legacy)

    Returns dict with results for each method.
    """
    results = {}

    for method in METHODS:
        method_results = {}

        for layer in layers:
            for component in components:
                for position in positions:
                    key = f"L{layer}_{component}_{position}"

                    # Check multiple paths
                    paths_to_check = [
                        data_dir / "analysis" / "embeddings" / method / f"{key}.npy",
                        data_dir / "analysis" / "embeddings" / key / f"{method}_embedding.npy",
                        data_dir / "cache" / method / f"{key}.npy",
                    ]

                    found_path = None
                    for filepath in paths_to_check:
                        if filepath.exists():
                            found_path = filepath
                            break

                    if found_path is not None:
                        try:
                            start = time.time()
                            arr = np.load(found_path)
                            load_ms = (time.time() - start) * 1000
                            method_results[key] = {
                                "status": "OK",
                                "shape": arr.shape,
                                "size_kb": found_path.stat().st_size / 1024,
                                "load_ms": load_ms,
                            }
                        except Exception as e:
                            method_results[key] = {"status": f"ERROR: {e}", "shape": None, "load_ms": None}
                    else:
                        method_results[key] = {"status": "MISSING", "shape": None, "load_ms": None}

        results[method] = method_results

    return results


def validate_layer_trajectories(
    data_dir: Path,
    components: list[str],
    positions: list[str],
) -> dict:
    """Validate 1D x Layer trajectory files (component × position)."""
    trajectories_dir = data_dir / "analysis" / "trajectories"
    results = {}

    for component in components:
        for position in positions:
            key = f"layers_{component}_{position}"
            filepath = trajectories_dir / f"{key}.npz"

            if filepath.exists():
                try:
                    start = time.time()
                    data = np.load(filepath, allow_pickle=True)
                    load_ms = (time.time() - start) * 1000
                    results[key] = {
                        "status": "OK",
                        "keys": list(data.keys()),
                        "size_kb": filepath.stat().st_size / 1024,
                        "load_ms": load_ms,
                    }
                except Exception as e:
                    results[key] = {"status": f"ERROR: {e}", "load_ms": None}
            else:
                results[key] = {"status": "MISSING", "load_ms": None}

    return results


def validate_position_trajectories(
    data_dir: Path,
    layers: list[int],
    components: list[str],
) -> dict:
    """Validate 1D x Position trajectory files (layer × component)."""
    trajectories_dir = data_dir / "analysis" / "trajectories"
    results = {}

    for layer in layers:
        for component in components:
            key = f"positions_L{layer}_{component}"
            filepath = trajectories_dir / f"{key}.npz"

            if filepath.exists():
                try:
                    start = time.time()
                    data = np.load(filepath, allow_pickle=True)
                    load_ms = (time.time() - start) * 1000
                    results[key] = {
                        "status": "OK",
                        "keys": list(data.keys()),
                        "size_kb": filepath.stat().st_size / 1024,
                        "load_ms": load_ms,
                    }
                except Exception as e:
                    results[key] = {"status": f"ERROR: {e}", "load_ms": None}
            else:
                results[key] = {"status": "MISSING", "load_ms": None}

    return results


def test_server_endpoint(base_url: str, layer: int, component: str, position: str, method: str) -> dict:
    """Test a server endpoint and measure response time."""
    url = f"{base_url}/api/embedding?layer={layer}&component={component}&position={position}&method={method}"

    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read())
            elapsed_ms = (time.time() - start) * 1000

            if "error" in data:
                return {"status": "ERROR", "error": data["error"], "time_ms": elapsed_ms}
            if "embeddings" not in data:
                return {"status": "ERROR", "error": "No embeddings", "time_ms": elapsed_ms}

            return {"status": "OK", "time_ms": elapsed_ms, "n_points": len(data["embeddings"])}

    except urllib.error.URLError as e:
        return {"status": "ERROR", "error": str(e)}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


def test_server(
    data_dir: Path,
    port: int,
    layers: list[int],
    components: list[str],
    positions: list[str],
) -> dict:
    """Test all server endpoints."""
    base_url = f"http://localhost:{port}"

    # Check if server is running
    try:
        with urllib.request.urlopen(f"{base_url}/api/metadata", timeout=2):
            pass
    except Exception:
        return None  # Server not running

    results = {}

    for layer in layers:
        for component in components:
            for position in positions:
                for method in METHODS:
                    key = f"L{layer}_{component}_{position}_{method}"
                    results[key] = test_server_endpoint(base_url, layer, component, position, method)

    return results


def generate_markdown_report(
    data_dir: Path,
    layers: list[int],
    components: list[str],
    positions: list[str],
    embeddings_results: dict,
    layer_traj_results: dict,
    pos_traj_results: dict,
    server_results: dict | None,
) -> str:
    """Generate comprehensive Markdown validation report."""
    lines = []
    lines.append("# GeoApp Complete Validation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Data Directory:** `{data_dir}`")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")

    total_3d = len(layers) * len(components) * len(positions) * len(METHODS)
    ok_3d = sum(1 for m in embeddings_results.values() for r in m.values() if r["status"] == "OK")
    missing_3d = total_3d - ok_3d

    total_layer_traj = len(components) * len(positions)
    ok_layer_traj = sum(1 for r in layer_traj_results.values() if r["status"] == "OK")

    total_pos_traj = len(layers) * len(components)
    ok_pos_traj = sum(1 for r in pos_traj_results.values() if r["status"] == "OK")

    lines.append(f"| Data Type | Total | OK | Missing/Error |")
    lines.append(f"|-----------|-------|----|--------------:|")
    lines.append(f"| 3D Embeddings | {total_3d} | {ok_3d} | {missing_3d} |")
    lines.append(f"| 1D × Layer Trajectories | {total_layer_traj} | {ok_layer_traj} | {total_layer_traj - ok_layer_traj} |")
    lines.append(f"| 1D × Position Trajectories | {total_pos_traj} | {ok_pos_traj} | {total_pos_traj - ok_pos_traj} |")
    lines.append("")

    if missing_3d == 0 and ok_layer_traj == total_layer_traj and ok_pos_traj == total_pos_traj:
        lines.append("**STATUS: ALL DATA VALIDATED SUCCESSFULLY**")
    else:
        lines.append("**STATUS: VALIDATION FAILED - SEE DETAILS BELOW**")
    lines.append("")

    # 3D Embeddings detail table
    lines.append("## 3D Embeddings (Layer × Component × Position × Method)")
    lines.append("")

    for method in METHODS:
        lines.append(f"### Method: {method}")
        lines.append("")

        # Create table with positions as columns, layers × components as rows
        header = "| Layer | Component |" + "|".join(f" {p[:8]} " for p in positions) + "|"
        separator = "|-------|-----------|" + "|".join("---" for _ in positions) + "|"
        lines.append(header)
        lines.append(separator)

        method_data = embeddings_results.get(method, {})
        for layer in layers:
            for component in components:
                cells = []
                for position in positions:
                    key = f"L{layer}_{component}_{position}"
                    result = method_data.get(key, {"status": "?"})
                    status = result["status"]
                    load_ms = result.get("load_ms")
                    if status == "OK":
                        if load_ms is not None:
                            cells.append(f"{load_ms:.1f}")
                        else:
                            cells.append("OK")
                    elif status == "MISSING":
                        cells.append("**MISS**")
                    else:
                        cells.append("ERR")
                row = f"| L{layer} | {component} |" + "|".join(cells) + "|"
                lines.append(row)

        lines.append("")

        # Timing stats for this method
        all_times = [r.get("load_ms") for r in method_data.values() if r.get("load_ms") is not None]
        if all_times:
            lines.append(f"**Load time stats (ms):** min={min(all_times):.1f}, max={max(all_times):.1f}, avg={sum(all_times)/len(all_times):.1f}")
            lines.append("")

    # Missing 3D embeddings list
    missing_embeds = []
    for method, method_data in embeddings_results.items():
        for key, result in method_data.items():
            if result["status"] != "OK":
                missing_embeds.append(f"{key} ({method}): {result['status']}")

    if missing_embeds:
        lines.append("### Missing/Error 3D Embeddings")
        lines.append("")
        for item in missing_embeds[:50]:
            lines.append(f"- `{item}`")
        if len(missing_embeds) > 50:
            lines.append(f"- ... and {len(missing_embeds) - 50} more")
        lines.append("")

    # 1D × Layer Trajectories
    lines.append("## 1D × Layer Trajectories (Component × Position)")
    lines.append("")
    lines.append("| Component |" + "|".join(f" {p[:8]} " for p in positions) + "|")
    lines.append("|-----------|" + "|".join("---" for _ in positions) + "|")

    for component in components:
        cells = []
        for position in positions:
            key = f"layers_{component}_{position}"
            result = layer_traj_results.get(key, {"status": "?"})
            if result["status"] == "OK":
                load_ms = result.get("load_ms")
                if load_ms is not None:
                    cells.append(f"{load_ms:.1f}")
                else:
                    cells.append("OK")
            else:
                cells.append("**MISS**")
        lines.append(f"| {component} |" + "|".join(cells) + "|")

    lines.append("")

    # Timing stats for layer trajectories
    layer_traj_times = [r.get("load_ms") for r in layer_traj_results.values() if r.get("load_ms") is not None]
    if layer_traj_times:
        lines.append(f"**Load time stats (ms):** min={min(layer_traj_times):.1f}, max={max(layer_traj_times):.1f}, avg={sum(layer_traj_times)/len(layer_traj_times):.1f}")
        lines.append("")

    # 1D × Position Trajectories
    lines.append("## 1D × Position Trajectories (Layer × Component)")
    lines.append("")
    lines.append("| Layer |" + "|".join(f" {c} " for c in components) + "|")
    lines.append("|-------|" + "|".join("---" for _ in components) + "|")

    for layer in layers:
        cells = []
        for component in components:
            key = f"positions_L{layer}_{component}"
            result = pos_traj_results.get(key, {"status": "?"})
            if result["status"] == "OK":
                load_ms = result.get("load_ms")
                if load_ms is not None:
                    cells.append(f"{load_ms:.1f}")
                else:
                    cells.append("OK")
            else:
                cells.append("**MISS**")
        lines.append(f"| L{layer} |" + "|".join(cells) + "|")

    lines.append("")

    # Timing stats for position trajectories
    pos_traj_times = [r.get("load_ms") for r in pos_traj_results.values() if r.get("load_ms") is not None]
    if pos_traj_times:
        lines.append(f"**Load time stats (ms):** min={min(pos_traj_times):.1f}, max={max(pos_traj_times):.1f}, avg={sum(pos_traj_times)/len(pos_traj_times):.1f}")
        lines.append("")

    # Server results (if available)
    if server_results:
        lines.append("## Server Response Times")
        lines.append("")

        # Calculate stats
        times = [r["time_ms"] for r in server_results.values() if "time_ms" in r]
        errors = [k for k, r in server_results.items() if r["status"] != "OK"]

        if times:
            lines.append(f"- **Min:** {min(times):.1f} ms")
            lines.append(f"- **Max:** {max(times):.1f} ms")
            lines.append(f"- **Avg:** {sum(times) / len(times):.1f} ms")
            lines.append(f"- **Errors:** {len(errors)}")
            lines.append("")

        if errors:
            lines.append("### Server Errors")
            lines.append("")
            for key in errors[:20]:
                result = server_results[key]
                lines.append(f"- `{key}`: {result.get('error', 'Unknown')}")
            if len(errors) > 20:
                lines.append(f"- ... and {len(errors) - 20} more")
            lines.append("")
    else:
        lines.append("## Server Response Times")
        lines.append("")
        lines.append("*Server not running - skipped*")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Complete GeoApp validation")
    parser.add_argument("--data-dir", default="out/geometry", help="Data directory")
    parser.add_argument("--test-server", action="store_true", help="Test server endpoints")
    parser.add_argument("--port", type=int, default=8642, help="Server port")
    parser.add_argument("--output", default="/tmp/validation.md", help="Output file")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_file = Path(args.output)

    print("=" * 60)
    print("COMPLETE GEOAPP VALIDATION")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output: {output_file}")
    print()

    # Get config from summary.json
    layers, components, positions = get_config_from_summary(data_dir)
    if not positions:
        print("ERROR: No positions found")
        return 1

    print(f"Found {len(positions)} positions")
    print(f"Validating {len(layers)} layers × {len(components)} components × {len(positions)} positions")
    print()

    # Validate 3D embeddings
    print("Validating 3D embeddings...")
    embeddings_results = validate_3d_embeddings(data_dir, layers, components, positions)

    total = len(layers) * len(components) * len(positions)
    ok = sum(1 for m in embeddings_results.values() for r in m.values() if r["status"] == "OK")
    print(f"  3D embeddings: {ok}/{total}")

    # Validate layer trajectories
    print("Validating 1D × Layer trajectories...")
    layer_traj_results = validate_layer_trajectories(data_dir, components, positions)

    total = len(components) * len(positions)
    ok = sum(1 for r in layer_traj_results.values() if r["status"] == "OK")
    print(f"  Layer trajectories: {ok}/{total}")

    # Validate position trajectories
    print("Validating 1D × Position trajectories...")
    pos_traj_results = validate_position_trajectories(data_dir, layers, components)

    total = len(layers) * len(components)
    ok = sum(1 for r in pos_traj_results.values() if r["status"] == "OK")
    print(f"  Position trajectories: {ok}/{total}")

    # Test server (optional)
    server_results = None
    if args.test_server:
        print("Testing server endpoints...")
        server_results = test_server(data_dir, args.port, layers, components, positions)
        if server_results:
            ok = sum(1 for r in server_results.values() if r["status"] == "OK")
            print(f"  Server endpoints: {ok}/{len(server_results)}")
        else:
            print("  Server not running")

    # Generate report
    print()
    print(f"Generating report: {output_file}")
    report = generate_markdown_report(
        data_dir, layers, components, positions,
        embeddings_results, layer_traj_results, pos_traj_results, server_results
    )

    with open(output_file, "w") as f:
        f.write(report)

    print()
    print("=" * 60)
    print(f"VALIDATION COMPLETE - See {output_file}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
