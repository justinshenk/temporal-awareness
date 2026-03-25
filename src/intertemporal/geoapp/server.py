"""FastAPI server for geometry visualization backend."""

import os
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..geoapp.data_loader import GeometryDataLoader

from .routes import create_router

# Default paths
DEFAULT_DATA_DIR = Path("out/geometry")
DEFAULT_FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"


def create_app(
    data_dir: Path | str | None = None,
    frontend_dir: Path | str | None = None,
    enable_cors: bool = True,
    warmup: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        data_dir: Directory containing geometry output data. Defaults to out/geometry.
        frontend_dir: Directory containing built React frontend. Defaults to frontend/dist.
        enable_cors: Enable CORS for local development. Defaults to True.
        warmup: Pre-compute embeddings on startup. Defaults to False.

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

    # Create data loader
    data_loader = GeometryDataLoader(data_dir)

    # NON-BLOCKING WARMUP: Compute embeddings in background after server starts
    def background_warmup():
        import time
        # Small delay to let server start first
        time.sleep(0.5)
        print("\n[Background] Starting embedding warmup...")

        pca_count = data_loader.warmup(
            methods=["pca"],
            components=["resid_pre", "resid_post", "attn_out", "mlp_out"],
            positions=None,
        )
        print(f"[Background] PCA: {pca_count} cached")

        umap_count = data_loader.warmup(
            methods=["umap"],
            components=["resid_pre", "resid_post", "attn_out", "mlp_out"],
            positions=None,
        )
        print(f"[Background] UMAP: {umap_count} cached")

        tsne_count = data_loader.warmup(
            methods=["tsne"],
            components=["resid_pre", "resid_post"],
            positions=None,
        )
        print(f"[Background] t-SNE: {tsne_count} cached")
        print(f"[Background] Warmup complete: {pca_count + umap_count + tsne_count} total")

    threading.Thread(target=background_warmup, daemon=True).start()

    # Create FastAPI app
    app = FastAPI(
        title="Geometry API",
        description="API for exploring embedding visualizations of transformer activations",
        version="2.0.0",
    )

    # Add CORS middleware for local development
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Create and include API router
    api_router = create_router(data_loader)
    app.include_router(api_router)

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
    warmup = os.environ.get("GEOMETRY_WARMUP") == "1"

    return create_app(
        data_dir=data_dir,
        frontend_dir=frontend_dir,
        enable_cors=True,
        warmup=warmup,
    )


def run_app(
    data_dir: Path | str | None = None,
    frontend_dir: Path | str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    warmup: bool = False,
    reload: bool = False,
) -> None:
    """Run the FastAPI application with uvicorn.

    Args:
        data_dir: Directory containing geometry output data.
        frontend_dir: Directory containing built React frontend.
        host: Host to bind to. Defaults to 127.0.0.1.
        port: Port to bind to. Defaults to 8000.
        warmup: Pre-compute embeddings on startup. Defaults to False.
        reload: Enable auto-reload for development. Defaults to False.
    """
    # Store config in environment for factory function
    if data_dir:
        os.environ["GEOMETRY_DATA_DIR"] = str(data_dir)
    if frontend_dir:
        os.environ["GEOMETRY_FRONTEND_DIR"] = str(frontend_dir)
    if warmup:
        os.environ["GEOMETRY_WARMUP"] = "1"

    print(f"Starting Geometry API server at http://{host}:{port}")
    print(f"Data directory: {data_dir or DEFAULT_DATA_DIR}")
    print(f"API docs: http://{host}:{port}/docs")

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
            warmup=warmup,
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
