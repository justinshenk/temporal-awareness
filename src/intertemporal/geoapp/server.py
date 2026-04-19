"""FastAPI server for geometry visualization backend.

Architecture: Load-Only Pattern with Multi-Dataset Support
- All embeddings must be pre-computed using compute_geometry_analysis.py
- Server loads pre-computed data from analysis/embeddings/
- No runtime computation - instant loading from disk
- Memory caching for repeated access
- Supports multiple datasets under /api/{dataset}/...
"""

import json
import os
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..geoapp.data_loader import GeometryDataLoader

from .routes import create_router

# Background executor for prefetching - separate from request handling
# Use 4 workers to parallelize prefetch of multiple layers/positions
_prefetch_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="prefetch")

# Default paths
DEFAULT_DATA_DIR = Path("out/geo/geometry")
DEFAULT_FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"


def _validate_dataset(data_dir: Path, dataset_name: str) -> tuple[int, list[str]]:
    """Validate a single dataset directory.

    Returns (pca_count, missing_methods) tuple.
    Raises RuntimeError if data is missing.
    """
    if not data_dir.exists():
        raise RuntimeError(f"Data directory does not exist: {data_dir}")

    data_subdir = data_dir / "data"
    if not data_subdir.exists():
        raise RuntimeError(f"Data subdirectory not found: {data_subdir}")

    samples_dir = data_subdir / "samples"
    if not samples_dir.exists():
        raise RuntimeError(f"Samples directory not found: {samples_dir}")

    metadata_file = data_subdir / "metadata.json"
    if not metadata_file.exists():
        raise RuntimeError(f"Metadata file not found: {metadata_file}")

    # Create temporary data loader to check embeddings
    data_loader = GeometryDataLoader(data_dir)

    embeddings_dir = data_dir / "analysis" / "embeddings"

    # Get required targets from summary.json
    summary_path = data_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        layers = summary.get("layers", data_loader.get_layers())
        components = summary.get("components", data_loader.get_components())
        positions = summary.get("positions", data_loader.get_positions())
    else:
        layers = data_loader.get_layers()
        components = data_loader.get_components()
        positions = data_loader.get_positions()

    # Check for PCA embeddings
    pca_count = 0
    for layer in layers:
        for component in components:
            for position in positions:
                safe_pos = position.replace(":", "_")
                filename = f"L{layer}_{component}_{safe_pos}.npy"
                path = embeddings_dir / "pca" / filename
                if path.exists():
                    pca_count += 1

    if pca_count == 0:
        raise RuntimeError(
            f"No pre-computed embeddings found for {dataset_name}!\n"
            f"Expected path: {embeddings_dir / 'pca'}\n"
            "Run: uv run python scripts/intertemporal/compute_geometry_analysis.py"
        )

    # Check available methods - only PCA is required, UMAP/t-SNE are optional
    available_methods = data_loader.get_available_methods()
    if "pca" not in available_methods:
        raise RuntimeError(
            f"PCA embeddings required but not found for {dataset_name}!\n"
            f"Run: uv run python scripts/intertemporal/compute_geometry_analysis.py"
        )

    return pca_count, available_methods


def create_app(
    data_dirs: list[tuple[str, Path]] | None = None,
    frontend_dir: Path | str | None = None,
    enable_cors: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application with multi-dataset support.

    Uses load-only architecture:
    - All embeddings must be pre-computed using compute_geometry_analysis.py
    - Server loads from analysis/embeddings/
    - No runtime computation - instant loading
    - Memory caching for repeated access

    Args:
        data_dirs: List of (dataset_name, path) tuples. Defaults to single "geometry" dataset.
        frontend_dir: Directory containing built React frontend. Defaults to frontend/dist.
        enable_cors: Enable CORS for local development. Defaults to True.

    Returns:
        Configured FastAPI application instance.
    """
    # Resolve data directories
    if data_dirs is None:
        data_dirs = [("geometry", DEFAULT_DATA_DIR)]

    # Ensure paths are Path objects
    data_dirs = [(name, Path(path)) for name, path in data_dirs]

    # Resolve frontend directory
    if frontend_dir is None:
        frontend_dir = DEFAULT_FRONTEND_DIR
    frontend_dir = Path(frontend_dir)

    # Create FastAPI app
    app = FastAPI(
        title="Geometry API",
        description="Streaming API for embedding visualizations - lazy loading with smart prefetch",
        version="3.0.0",
    )

    # Store prefetch executor reference for routes to use
    app.state.prefetch_executor = _prefetch_executor

    # Store dataset info for listing endpoint
    app.state.datasets = {name: str(path) for name, path in data_dirs}

    # Add CORS middleware for local development
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Add request timing middleware for performance logging
    @app.middleware("http")
    async def log_request_time(request, call_next):
        import time as time_mod
        start = time_mod.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time_mod.perf_counter() - start) * 1000
        # Only log API requests, skip static files
        path = request.url.path
        if path.startswith("/api/"):
            status = "FAST" if elapsed_ms < 100 else "SLOW" if elapsed_ms > 500 else "OK"
            print(f"[{status}] {path} - {elapsed_ms:.0f}ms")
        return response

    # Create data loaders and routers for each dataset
    data_loaders: dict[str, GeometryDataLoader] = {}
    for dataset_name, data_dir in data_dirs:
        data_loader = GeometryDataLoader(data_dir)
        data_loaders[dataset_name] = data_loader

        # Create router with dataset prefix
        api_router = create_router(data_loader, dataset_name=dataset_name)
        app.include_router(api_router)

    # Root API endpoint to list datasets
    from fastapi import APIRouter
    root_router = APIRouter(prefix="/api", tags=["root"])

    @root_router.get("/datasets")
    async def list_datasets():
        """List all available datasets."""
        datasets = []
        for name, loader in data_loaders.items():
            datasets.append({
                "name": name,
                "n_samples": loader.n_samples,
                "n_layers": len(loader.get_layers()),
                "n_positions": len(loader.get_positions()),
                "methods": loader.get_available_methods(),
            })
        return {"datasets": datasets}

    app.include_router(root_router)

    # Startup event to validate and preload ALL data into memory
    @app.on_event("startup")
    async def preload_all_data():
        """Validate and preload ALL data into memory at startup.

        STRICT MODE: Server will CRASH if any required data is missing.
        No fallbacks, no graceful degradation. Data MUST be precomputed.
        """
        start_time = time.time()

        print()
        print("=" * 60)
        print("  STRICT VALIDATION: ALL DATA MUST BE PRE-COMPUTED")
        print("=" * 60)
        print()
        print(f"  Loading {len(data_dirs)} dataset(s):")
        for name, path in data_dirs:
            print(f"    - {name}: {path}")
        print()

        total_pca_count = 0
        total_metadata_count = 0
        total_token_count = 0

        for dataset_name, data_dir in data_dirs:
            print(f"  [{dataset_name}] Validating...")

            # Phase 0: Validate data directory structure
            pca_count, available_methods = _validate_dataset(data_dir, dataset_name)

            print("    Data structure: OK")
            print(f"    PCA embeddings: {pca_count} files")
            print(f"    Available methods: {available_methods}")

            data_loader = data_loaders[dataset_name]

            # Phase 1: Skip heavy warmup - load on demand
            print(f"    Embeddings: {pca_count} available for lazy loading")
            total_pca_count += pca_count

            # Phase 2: Preload metadata
            phase2_start = time.time()
            metadata_count = data_loader.preload_all_metadata()
            phase2_elapsed = time.time() - phase2_start
            print(f"    Metadata: {metadata_count} color options in {phase2_elapsed:.1f}s")
            total_metadata_count += metadata_count

            # Phase 3: Validate token mappings
            phase3_start = time.time()
            token_count = data_loader.preload_all_tokens()
            phase3_elapsed = time.time() - phase3_start
            print(f"    Tokens: {token_count} mappings in {phase3_elapsed:.1f}s")
            total_token_count += token_count
            print()

        total_elapsed = time.time() - start_time

        # Get memory usage
        mem_usage = resource.getrusage(resource.RUSAGE_SELF)
        mem_mb = mem_usage.ru_maxrss / (1024 * 1024)
        if sys.platform == "linux":
            mem_mb = mem_usage.ru_maxrss / 1024

        print("  PRELOAD COMPLETE")
        print(f"    Datasets: {len(data_dirs)}")
        print(f"    Total time: {total_elapsed:.1f}s")
        print(f"    Total embeddings: {total_pca_count}")
        print(f"    Total metadata: {total_metadata_count}")
        print(f"    Total tokens: {total_token_count}")
        print()
        print("  MEMORY USAGE:")
        print(f"    Peak RSS: {mem_mb:.1f} MB")
        print()
        print("  All visualizations will be served INSTANTLY from memory.")
        print("=" * 60)
        print()

    # Mount static files for frontend if directory exists
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app


def app_factory() -> FastAPI:
    """Factory function for uvicorn reload mode.

    Reads configuration from environment variables set by run_app.
    """
    data_dirs_str = os.environ.get("GEOMETRY_DATA_DIRS", "")
    frontend_dir = os.environ.get("GEOMETRY_FRONTEND_DIR")

    # Parse data dirs from "name1:path1,name2:path2" format
    data_dirs = []
    if data_dirs_str:
        for item in data_dirs_str.split(","):
            if ":" in item:
                name, path = item.split(":", 1)
                data_dirs.append((name, Path(path)))

    if not data_dirs:
        data_dirs = None  # Use default

    return create_app(
        data_dirs=data_dirs,
        frontend_dir=frontend_dir,
        enable_cors=True,
    )


def run_app(
    data_dirs: list[tuple[str, Path]] | None = None,
    frontend_dir: Path | str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Run the FastAPI application with uvicorn.

    Uses load-only architecture - all embeddings must be pre-computed
    using compute_geometry_analysis.py. Server loads from disk cache.

    Args:
        data_dirs: List of (dataset_name, path) tuples.
        frontend_dir: Directory containing built React frontend.
        host: Host to bind to. Defaults to 127.0.0.1.
        port: Port to bind to. Defaults to 8000.
        reload: Enable auto-reload for development. Defaults to False.
    """
    # Store config in environment for factory function
    if data_dirs:
        data_dirs_str = ",".join(f"{name}:{path}" for name, path in data_dirs)
        os.environ["GEOMETRY_DATA_DIRS"] = data_dirs_str
    if frontend_dir:
        os.environ["GEOMETRY_FRONTEND_DIR"] = str(frontend_dir)

    print(f"Starting Geometry API server at http://{host}:{port}")
    if data_dirs:
        print(f"Datasets: {[name for name, _ in data_dirs]}")
    print(f"API docs: http://{host}:{port}/docs")
    print("Mode: Load-only (pre-computed embeddings)")

    # Use import string for reload mode, direct app otherwise
    if reload:
        uvicorn.run(
            "src.intertemporal.geoapp.server:app_factory",
            host=host,
            port=port,
            reload=reload,
            factory=True,
        )
    else:
        app = create_app(
            data_dirs=data_dirs,
            frontend_dir=frontend_dir,
            enable_cors=True,
        )
        uvicorn.run(
            app,
            host=host,
            port=port,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Geometry API server")
    parser.add_argument(
        "--data-dir",
        type=str,
        action="append",
        dest="data_dirs",
        help="Dataset directory (can specify multiple: --data-dir name:path)",
    )
    parser.add_argument(
        "--frontend-dir",
        type=str,
        default=None,
        help="Directory containing built React frontend",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Parse data dirs
    data_dirs = None
    if args.data_dirs:
        data_dirs = []
        for item in args.data_dirs:
            if ":" in item:
                name, path = item.split(":", 1)
                data_dirs.append((name, Path(path)))
            else:
                # Single path without name - use directory name
                path = Path(item)
                data_dirs.append((path.name, path))

    run_app(
        data_dirs=data_dirs,
        frontend_dir=args.frontend_dir,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
