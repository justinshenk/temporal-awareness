# geoapp-vercel — public deploy of the geometry explorer

Self-contained Vercel (Python / Fluid Compute) deployment of the load-only geoapp
FastAPI backend + built React SPA. Deploying this and setting `PUBLIC_GEOAPP_URL` on the
public site makes the `/explore/` page embed the live explorer instead of the
"run it locally" fallback.

## Layout

```
api/index.py     ASGI entrypoint — create_app() serving SPA at / and API at /api/{dataset}
geoapp/          Flat copy of the 4 serve-time modules (server, routes, models, data_loader)
                 — copied so imports don't pull in the heavy src/ research tree (auto_export).
webdist/         Built React SPA (from src/intertemporal/geoapp/frontend/dist)
data/geometry/   Precomputed dataset (STAGE THIS — see below). Bundled via vercel.json includeFiles.
requirements.txt fastapi, numpy, pydantic, uvicorn (no torch/sklearn — serve is load-only)
vercel.json      Routes all requests to the function; bundles geoapp/, webdist/, data/
```

If the geoapp serve-time source (`src/intertemporal/geoapp/{server,routes,models,data_loader}.py`)
changes, re-copy it into `geoapp/` and re-apply the one edit: in `geoapp/server.py`,
`from ..geoapp.data_loader import GeometryDataLoader` → `from .data_loader import GeometryDataLoader`.
Re-build the SPA (`cd src/intertemporal/geoapp/frontend && npm ci && npm run build`) into `webdist/`.

## Stage the data

Copy the precomputed dataset (the `out/geo/geometry` tree produced by
`scripts/intertemporal/compute_geometry_analysis.py`) to `data/geometry/`. It must contain:

```
data/geometry/data/samples/…
data/geometry/data/metadata.json
data/geometry/summary.json
data/geometry/analysis/embeddings/pca/L{layer}_{component}_{position}.npy
```

The server strictly validates this on startup and the loader reads it eagerly on boot, so
the function will not start without it. If the tree is large, check Vercel function bundle
limits — if it exceeds them, move the `.npy` data to Vercel Blob and adapt
`geoapp/data_loader.py` to fetch (cache to `/tmp`) instead of reading local paths.

## Deploy

```
cd geoapp-vercel
vercel            # first run: create a new project (e.g. temporal-geoapp)
vercel --prod     # promote to production
```

Then verify:
- `curl https://<project>.vercel.app/api/datasets` → lists the `geometry` dataset
- open `https://<project>.vercel.app/` → the explorer SPA loads and its `/api/geometry/…`
  calls return 200.

## Wire the site

Set `PUBLIC_GEOAPP_URL=https://<project>.vercel.app` in both build paths of the public site:
- GitHub Pages: repo variable `PUBLIC_GEOAPP_URL` (already referenced in
  `.github/workflows/deploy-site.yml`).
- The `temporal-awareness` Vercel site project: add the env var (`vercel env add`).

Re-run the site build; the `/explore/` page will render the live iframe.
