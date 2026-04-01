#!/usr/bin/env python3
"""Run the Geometry Explorer interactive app.

Usage:
    # Load all datasets from out/geo/:
    uv run python scripts/intertemporal/run_geoapp.py

    # Load only specific dataset(s):
    uv run python scripts/intertemporal/run_geoapp.py geometry
    uv run python scripts/intertemporal/run_geoapp.py geometry another_exp

    # Development mode (backend only, run frontend separately):
    uv run python scripts/intertemporal/run_geoapp.py --dev

    # Custom port:
    uv run python scripts/intertemporal/run_geoapp.py --port 8080

    # Or use the shell script to run everything:
    ./scripts/intertemporal/run_geoapp.sh
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.intertemporal.geoapp import run_app

# Frontend directory location
FRONTEND_DIR = PROJECT_ROOT / "src" / "intertemporal" / "geoapp" / "frontend"

# Default base directory for datasets
DEFAULT_BASE_DIR = Path("out/geo")


def discover_datasets(base_dir: Path) -> list[tuple[str, Path]]:
    """Discover all valid dataset directories under base_dir.

    A valid dataset directory has:
    - data/samples/ subdirectory
    - analysis/embeddings/ subdirectory

    Returns list of (name, path) tuples.
    """
    datasets = []
    if not base_dir.exists():
        return datasets

    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Check for required structure
        if (subdir / "data" / "samples").exists() and (subdir / "analysis" / "embeddings").exists():
            datasets.append((subdir.name, subdir))

    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Run Geometry Explorer - Interactive 3D visualization with React frontend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset names to load from out/geo/ (default: all datasets)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="out/geo",
        help="Base directory containing dataset folders (default: out/geo)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for backend server (default: 8000)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development mode: run backend only, frontend runs separately",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    # Determine which datasets to load
    if args.datasets:
        # Load specific datasets
        data_dirs = []
        for name in args.datasets:
            path = base_dir / name
            if not path.exists():
                print(f"Error: Dataset not found: {path}")
                sys.exit(1)
            data_dirs.append((name, path))
    else:
        # Discover all datasets
        data_dirs = discover_datasets(base_dir)
        if not data_dirs:
            print(f"Error: No valid datasets found in {base_dir}")
            print("Run the geometry pipeline first to generate data.")
            sys.exit(1)

    print("=" * 60)
    print("Geometry Explorer")
    print("=" * 60)
    print()
    print(f"Loading {len(data_dirs)} dataset(s):")
    for name, path in data_dirs:
        print(f"  - {name}: {path}")
    print()

    if args.dev:
        # Development mode
        print("MODE: Development (backend only)")
        print()
        print(f"Backend API: http://{args.host}:{args.port}")
        print(f"API Docs:    http://{args.host}:{args.port}/docs")
        print()
        print("-" * 60)
        print("To run the frontend development server:")
        print()
        print(f"  cd {FRONTEND_DIR}")
        print("  npm install  # (first time only)")
        print("  npm run dev")
        print()
        print("The frontend will be available at http://localhost:3000")
        print("-" * 60)
        print()

        # Run backend without serving static files
        run_app(
            data_dirs=data_dirs,
            frontend_dir=None,  # Don't serve frontend in dev mode
            host=args.host,
            port=args.port,
            reload=True,  # Enable auto-reload in dev mode
        )
    else:
        # Production mode
        frontend_dist = FRONTEND_DIR / "dist"

        if not frontend_dist.exists():
            print("Error: Frontend build not found!")
            print()
            print("To build the frontend:")
            print(f"  cd {FRONTEND_DIR}")
            print("  npm install")
            print("  npm run build")
            print()
            print("Or use --dev mode to run frontend separately.")
            sys.exit(1)

        print("MODE: Production")
        print()
        print(f"App URL:  http://{args.host}:{args.port}")
        print(f"API Docs: http://{args.host}:{args.port}/docs")
        print()

        run_app(
            data_dirs=data_dirs,
            frontend_dir=frontend_dist,
            host=args.host,
            port=args.port,
            reload=False,
        )


if __name__ == "__main__":
    main()
