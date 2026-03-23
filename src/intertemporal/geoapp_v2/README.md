# GeoViz Explorer v2

Interactive 3D visualization app for exploring transformer activation embeddings.

## Architecture

- **Backend**: FastAPI server serving a JSON API
- **Frontend**: React app with Three.js for 3D visualization

## Running the App

### Development Mode

In development mode, the backend and frontend run as separate processes for hot reloading.

1. Start the backend:
   ```bash
   uv run python scripts/intertemporal/run_geoapp_v2.py --dev
   ```

2. In a separate terminal, start the frontend:
   ```bash
   cd src/intertemporal/geoapp_v2/frontend
   npm install  # first time only
   npm run dev
   ```

3. Open http://localhost:5173 in your browser

The frontend dev server proxies API requests to the backend at http://localhost:8000.

### Production Mode

In production mode, the FastAPI server serves the built React frontend.

1. Build the frontend:
   ```bash
   cd src/intertemporal/geoapp_v2/frontend
   npm install
   npm run build
   ```

2. Start the server:
   ```bash
   uv run python scripts/intertemporal/run_geoapp_v2.py
   ```

3. Open http://localhost:8000 in your browser

## Command Line Options

```
--data-dir PATH   Path to geo_viz output directory (default: out/geo_viz)
--host HOST       Host to bind to (default: 127.0.0.1)
--port PORT       Port for backend server (default: 8000)
--dev             Development mode: run backend only
--warmup          Pre-compute embeddings on startup
```

## API Endpoints

- `GET /api/config` - Get available layers, components, positions, and color options
- `GET /api/embedding/{layer}/{component}/{position}` - Get 3D embedding coordinates
- `GET /api/metadata?color_by={field}` - Get color values for samples
- `GET /api/sample/{idx}` - Get detailed sample information
- `GET /api/metrics/{layer}/{component}/{position}` - Get probe metrics

API documentation is available at http://localhost:8000/docs when the server is running.
