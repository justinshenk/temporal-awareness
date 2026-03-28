"""FastAPI server for geometry visualization backend.

Architecture: Load-Only Pattern
- All embeddings must be pre-computed using compute_geometry_analysis.py
- Server loads pre-computed data from analysis/embeddings/
- No runtime computation - instant loading from disk
- Memory caching for repeated access
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
DEFAULT_DATA_DIR = Path("out/geometry")
DEFAULT_FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"


def create_app(
    data_dir: Path | str | None = None,
    frontend_dir: Path | str | None = None,
    enable_cors: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Uses load-only architecture:
    - All embeddings must be pre-computed using compute_geometry_analysis.py
    - Server loads from analysis/embeddings/ (or legacy cache/)
    - No runtime computation - instant loading
    - Memory caching for repeated access

    Args:
        data_dir: Directory containing geometry output data. Defaults to out/geometry.
        frontend_dir: Directory containing built React frontend. Defaults to frontend/dist.
        enable_cors: Enable CORS for local development. Defaults to True.

    Returns:
        Configured FastAPI application instance.
    """
    # Resolve data directory
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    # Resolve frontend directory
    if frontend_dir is None:
        frontend_dir = DEFAULT_FRONTEND_DIR
    frontend_dir = Path(frontend_dir)

    # Create data loader with lazy loading - no upfront computation
    data_loader = GeometryDataLoader(data_dir)

    # Create FastAPI app
    app = FastAPI(
        title="Geometry API",
        description="Streaming API for embedding visualizations - lazy loading with smart prefetch",
        version="2.0.0",
    )

    # Store prefetch executor reference for routes to use
    app.state.prefetch_executor = _prefetch_executor

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

    # Create and include API router
    api_router = create_router(data_loader)
    app.include_router(api_router)

    # Startup event to validate and preload ALL data into memory
    @app.on_event("startup")
    async def preload_all_data():
        """Validate and preload ALL data into memory at startup.

        STRICT MODE: Server will CRASH if any required data is missing.
        No fallbacks, no graceful degradation. Data MUST be precomputed.

        Validates:
        - Data directory exists with required structure
        - Samples exist and have required files
        - Pre-computed embeddings exist for all targets

        Preloads:
        - All embeddings (PCA, UMAP, t-SNE) in parallel
        - All metadata/color values
        - All token mappings
        """
        start_time = time.time()

        print()
        print("=" * 60)
        print("  STRICT VALIDATION: ALL DATA MUST BE PRE-COMPUTED")
        print("=" * 60)
        print()

        # PHASE 0: Validate data directory structure
        print("  Phase 0: Validating data directory structure...")
        if not data_dir.exists():
            raise RuntimeError(
                f"Data directory does not exist: {data_dir}\n"
                "Run compute_geometry_analysis.py to generate data first."
            )

        data_subdir = data_dir / "data"
        if not data_subdir.exists():
            raise RuntimeError(
                f"Data subdirectory not found: {data_subdir}\n"
                "Run compute_geometry_analysis.py to generate data first."
            )

        samples_dir = data_subdir / "samples"
        if not samples_dir.exists():
            raise RuntimeError(
                f"Samples directory not found: {samples_dir}\n"
                "Run compute_geometry_analysis.py to generate data first."
            )

        metadata_file = data_subdir / "metadata.json"
        if not metadata_file.exists():
            raise RuntimeError(
                f"Metadata file not found: {metadata_file}\n"
                "Run compute_geometry_analysis.py to generate data first."
            )

        print("    Data directory structure: OK")

        # STRICT VALIDATION: Verify ALL required embedding files exist
        embeddings_dir = data_dir / "analysis" / "embeddings"
        legacy_cache_dir = data_dir / "cache"

        # Get required targets from summary.json (positions that have precomputed data)
        # NOT from data_loader.get_positions() which returns ALL semantic positions
        summary_path = data_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            layers = summary.get("layers", data_loader.get_layers())
            components = summary.get("components", data_loader.get_components())
            positions = summary.get("positions", data_loader.get_positions())
        else:
            # Fallback if no summary.json (shouldn't happen with proper setup)
            layers = data_loader.get_layers()
            components = data_loader.get_components()
            positions = data_loader.get_positions()

        # Check for PCA embeddings - these are REQUIRED
        missing_embeddings = []
        pca_count = 0
        for layer in layers:
            for component in components:
                for position in positions:
                    # Sanitize position for filename
                    safe_pos = position.replace(":", "_")
                    filename = f"L{layer}_{component}_{safe_pos}.npy"

                    # Check new path first, then legacy
                    new_path = embeddings_dir / "pca" / filename
                    legacy_path = legacy_cache_dir / "pca" / filename

                    if new_path.exists() or legacy_path.exists():
                        pca_count += 1
                    else:
                        missing_embeddings.append(f"PCA L{layer}_{component}_{position}")

        if pca_count == 0:
            raise RuntimeError(
                "No pre-computed embeddings found!\n"
                f"Checked: {embeddings_dir / 'pca'}\n"
                f"Checked: {legacy_cache_dir / 'pca'}\n"
                "Run: uv run python scripts/intertemporal/compute_geometry_analysis.py"
            )

        if missing_embeddings:
            # Show first 10 missing as examples
            examples = missing_embeddings[:10]
            more_count = len(missing_embeddings) - 10 if len(missing_embeddings) > 10 else 0
            raise RuntimeError(
                f"Missing {len(missing_embeddings)} required embedding files!\n"
                f"Examples: {', '.join(examples)}"
                + (f" (+{more_count} more)" if more_count > 0 else "") +
                "\nRun: uv run python scripts/intertemporal/compute_geometry_analysis.py"
            )

        # Use warmup_all for comprehensive parallel preloading
        print(f"  Found {pca_count} PCA embedding files (all required files present)")
        print(f"  Samples: {data_loader.n_samples}")
        print(f"  Layers: {len(data_loader.get_layers())}")
        print(f"  Positions: {len(data_loader.get_positions())}")
        print(f"  Components: {len(data_loader.get_components())}")
        print()

        # Phase 1: Load all embeddings in parallel (PCA only - UMAP/t-SNE may not exist)
        print("  Phase 1: Loading all embeddings in parallel...")
        phase1_start = time.time()
        results = data_loader.warmup_all(
            methods=["pca"],  # Only PCA is required
            layers=layers,
            components=components,
            positions=positions,  # Use positions from summary.json
            include_metadata=False,  # We'll do this separately for logging
            include_tokens=False,    # We'll do this separately for logging
        )
        phase1_elapsed = time.time() - phase1_start
        print(f"    Loaded {results['embeddings']} embeddings in {phase1_elapsed:.1f}s")

        # Phase 2: Preload all metadata/color values
        print("  Phase 2: Loading all metadata/color values...")
        phase2_start = time.time()
        metadata_count = data_loader.preload_all_metadata()
        phase2_elapsed = time.time() - phase2_start
        print(f"    Loaded {metadata_count} color options in {phase2_elapsed:.1f}s")

        # Phase 3: Validate token mappings
        print("  Phase 3: Validating token mappings...")
        phase3_start = time.time()
        token_count = data_loader.preload_all_tokens()
        phase3_elapsed = time.time() - phase3_start
        print(f"    Validated {token_count} sample token mappings in {phase3_elapsed:.1f}s")

        total_elapsed = time.time() - start_time

        # Get memory usage after preloading
        mem_usage = resource.getrusage(resource.RUSAGE_SELF)
        mem_mb = mem_usage.ru_maxrss / (1024 * 1024)  # Convert to MB (macOS returns bytes)
        # On Linux, ru_maxrss is in KB, so we'd divide by 1024 only
        if sys.platform == "linux":
            mem_mb = mem_usage.ru_maxrss / 1024

        print()
        print("  PRELOAD COMPLETE")
        print(f"    Total time: {total_elapsed:.1f}s")
        print(f"    Embeddings: {results['embeddings']}")
        print(f"    Metadata options: {metadata_count}")
        print(f"    Token mappings: {token_count}")
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
    data_dir = os.environ.get("GEOMETRY_DATA_DIR")
    frontend_dir = os.environ.get("GEOMETRY_FRONTEND_DIR")

    return create_app(
        data_dir=data_dir,
        frontend_dir=frontend_dir,
        enable_cors=True,
    )


def run_app(
    data_dir: Path | str | None = None,
    frontend_dir: Path | str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Run the FastAPI application with uvicorn.

    Uses load-only architecture - all embeddings must be pre-computed
    using compute_geometry_analysis.py. Server loads from disk cache.

    Args:
        data_dir: Directory containing geometry output data.
        frontend_dir: Directory containing built React frontend.
        host: Host to bind to. Defaults to 127.0.0.1.
        port: Port to bind to. Defaults to 8000.
        reload: Enable auto-reload for development. Defaults to False.
    """
    # Store config in environment for factory function
    if data_dir:
        os.environ["GEOMETRY_DATA_DIR"] = str(data_dir)
    if frontend_dir:
        os.environ["GEOMETRY_FRONTEND_DIR"] = str(frontend_dir)

    print(f"Starting Geometry API server at http://{host}:{port}")
    print(f"Data directory: {data_dir or DEFAULT_DATA_DIR}")
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
            data_dir=data_dir,
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
        default=None,
        help="Directory containing geometry output data",
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

    run_app(
        data_dir=args.data_dir,
        frontend_dir=args.frontend_dir,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
