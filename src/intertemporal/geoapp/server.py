"""FastAPI server for geometry visualization backend.

Architecture: Streaming/Reactive Pattern
- No upfront warmup - server starts instantly
- Lazy loading - embeddings computed on first request
- Smart prefetching - adjacent data loaded in background after each request
- Everything cached once loaded
"""

import asyncio
import os
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

    Uses streaming/reactive architecture:
    - No upfront warmup - server starts instantly
    - Lazy loading - embeddings computed on first request
    - Smart prefetching - adjacent data loaded in background
    - Everything cached once loaded

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

    # Startup event to precompute priority embeddings
    @app.on_event("startup")
    async def precompute_common():
        """Precompute embeddings for first, middle, and last layers on startup."""
        def _precompute():
            layers = data_loader.get_layers()
            positions = data_loader.get_positions()
            if not layers or not positions:
                return

            # Priority layers: first, middle, last
            priority_layers = [
                layers[0],
                layers[len(layers) // 2],
                layers[-1],
            ]
            # Deduplicate while preserving order
            priority_layers = list(dict.fromkeys(priority_layers))

            # Top 5 positions
            priority_positions = positions[:5]

            print(f"[STARTUP] Precomputing PCA for {len(priority_layers)} layers x {len(priority_positions)} positions")
            for layer in priority_layers:
                for pos in priority_positions:
                    data_loader.load_pca(layer, "resid_post", pos)
            print("[STARTUP] Precomputation complete")

        loop = asyncio.get_event_loop()
        loop.run_in_executor(_prefetch_executor, _precompute)

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

    Uses streaming/reactive architecture - no upfront warmup needed.
    Embeddings are loaded on-demand and cached. Smart prefetching
    loads adjacent data in background.

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
    print("Mode: Streaming (lazy load + smart prefetch)")

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
        "--warmup",
        action="store_true",
        help="Pre-compute embeddings on startup",
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
        warmup=args.warmup,
        reload=args.reload,
    )
