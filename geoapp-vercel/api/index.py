"""Vercel Python (Fluid Compute) ASGI entrypoint for the geometry explorer.

All requests are routed here by vercel.json. The FastAPI app serves both the
built React SPA (from webdist/) at `/` and the load-only geometry API under
`/api/{dataset}/...`. Data is bundled under data/geometry and validated on startup.
"""

import sys
import traceback
from pathlib import Path

from fastapi import FastAPI

# Make the self-contained flat `geoapp` package importable (sibling of api/).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _build() -> FastAPI:
    try:
        from geoapp.server import create_app

        return create_app(
            data_dirs=[("geometry", ROOT / "data" / "geometry")],
            frontend_dir=ROOT / "webdist",
            enable_cors=True,
        )
    except Exception:
        tb = traceback.format_exc()
        info = {
            "root": str(ROOT),
            "root_children": sorted(p.name for p in ROOT.iterdir()) if ROOT.exists() else "MISSING",
            "data_geometry_exists": (ROOT / "data" / "geometry").exists(),
            "metadata_exists": (ROOT / "data" / "geometry" / "data" / "metadata.json").exists(),
        }
        debug = FastAPI()

        @debug.get("/{path:path}")
        async def _debug(path: str):
            return {"init_error": True, "info": info, "traceback": tb}

        return debug


app = _build()
