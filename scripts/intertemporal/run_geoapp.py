#!/usr/bin/env python3
"""Run the Geometry Explorer interactive app.

Usage:
    # Development mode (backend only, run frontend separately):
    uv run python scripts/intertemporal/run_geoapp.py --dev

    # Production mode (serves built frontend from FastAPI):
    uv run python scripts/intertemporal/run_geoapp.py

    # Custom data directory:
    uv run python scripts/intertemporal/run_geoapp.py --data-dir out/geo_test

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


def main():
    parser = argparse.ArgumentParser(
        description="Run Geometry Explorer - Interactive 3D visualization with React frontend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="out/geometry",
        help="Path to geometry output directory (default: out/geometry)",
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

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Run the geometry pipeline first to generate data.")
        sys.exit(1)

    if args.dev:
        # Development mode
        print("=" * 60)
        print("Geometry Explorer - DEVELOPMENT MODE")
        print("=" * 60)
        print()
        print(f"Backend API: http://{args.host}:{args.port}")
        print(f"API Docs:    http://{args.host}:{args.port}/docs")
        print(f"Data:        {data_dir}")
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
            data_dir=data_dir,
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

        print("=" * 60)
        print("Geometry Explorer - PRODUCTION MODE")
        print("=" * 60)
        print()
        print(f"App URL:  http://{args.host}:{args.port}")
        print(f"API Docs: http://{args.host}:{args.port}/docs")
        print(f"Data:     {data_dir}")
        print()

        run_app(
            data_dir=data_dir,
            frontend_dir=frontend_dist,
            host=args.host,
            port=args.port,
            reload=False,
        )


if __name__ == "__main__":
    main()
