#!/usr/bin/env bash
#
# bundle_geo.sh — Package the Geometry Explorer UI + datasets into out/geo.zip
#
# Produces a self-contained, double-clickable Mac bundle that anyone can run
# without cloning the repo or knowing Python. The recipient unzips, opens
# start.command (or runs ./start.sh), and the UI launches in their browser.
#
# What goes in:
#   - geoapp/ Python backend (no src.common dependency)
#   - frontend/dist (pre-built React UI)
#   - data/geo/<dataset>/ (JSON metadata + analysis/ outputs only — no raw activations)
#   - run_geoapp.py entry point
#   - start.sh / start.command launchers (auto-install uv if missing)
#   - pyproject.toml (fastapi, uvicorn, numpy)
#   - README.md
#
# What's stripped to keep size sane:
#   - sample_*/L*/  (raw per-layer activations, ~110GB; UI never reads these)
#   - analysis/embeddings/{umap,tsne}/ (only PCA needed for default view)
#
# Usage:
#   bash scripts/intertemporal/bundle_geo.sh [dataset...]
#
# With no args, bundles every dataset under out/geo/.
# With args, bundles only the listed dataset names.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GEO_DIR="$REPO_ROOT/out/geo"
SRC_GEOAPP="$REPO_ROOT/src/intertemporal/geoapp"
FRONTEND_DIST="$SRC_GEOAPP/frontend/dist"

STAGING="$REPO_ROOT/out/geo-bundle"
ZIP_OUT="$REPO_ROOT/out/geo.zip"

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

if [[ ! -d "$GEO_DIR" ]]; then
  echo "ERROR: $GEO_DIR not found." >&2
  exit 1
fi

if [[ ! -d "$FRONTEND_DIST" ]]; then
  echo "Frontend dist/ not found at $FRONTEND_DIST."
  echo "Building frontend now (this takes ~30s)..."
  ( cd "$SRC_GEOAPP/frontend" && npm install --silent && npm run build )
fi

# Pick datasets
if [[ $# -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=()
  while IFS= read -r d; do DATASETS+=("$(basename "$d")"); done < <(find "$GEO_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "ERROR: no datasets found in $GEO_DIR" >&2
  exit 1
fi

echo "==> Bundling Geometry Explorer"
echo "    Datasets: ${DATASETS[*]}"
echo "    Staging:  $STAGING"
echo "    Output:   $ZIP_OUT"
echo

# ---------------------------------------------------------------------------
# Clean staging
# ---------------------------------------------------------------------------

rm -rf "$STAGING" "$ZIP_OUT"
mkdir -p "$STAGING/geoapp" "$STAGING/data/geo"

# ---------------------------------------------------------------------------
# Copy backend code (flat geoapp/ — no src.common dep)
# ---------------------------------------------------------------------------

echo "==> Copying backend code..."
cp "$SRC_GEOAPP/data_loader.py" "$STAGING/geoapp/"
cp "$SRC_GEOAPP/models.py"      "$STAGING/geoapp/"
cp "$SRC_GEOAPP/routes.py"      "$STAGING/geoapp/"
cp "$SRC_GEOAPP/server.py"      "$STAGING/geoapp/"

# Patch server.py: replace the cross-package import with a sibling import.
# Original line: `from ..geoapp.data_loader import GeometryDataLoader`
sed -i.bak \
  's|from \.\.geoapp\.data_loader import GeometryDataLoader|from .data_loader import GeometryDataLoader|' \
  "$STAGING/geoapp/server.py"
rm -f "$STAGING/geoapp/server.py.bak"

# Patch data_loader.py: get_markers() pulls in src.intertemporal.formatting,
# which transitively requires src.common. Inline the marker constants instead.
python3 - "$STAGING/geoapp/data_loader.py" <<'PYEOF'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
s = p.read_text()
new_func = '''    def get_markers(self) -> dict[str, str]:
        """Section markers used to highlight prompt regions in the UI."""
        return {
            "situation_marker": "SITUATION:",
            "task_marker": "TASK:",
            "objective_marker": "OBJECTIVE:",
            "action_marker": "ACTION:",
            "format_marker": "FORMAT:",
        }
'''
s, n = re.subn(
    r"    def get_markers\(self\) -> dict\[str, str\]:.*?return markers\n",
    new_func,
    s,
    count=1,
    flags=re.DOTALL,
)
if n != 1:
    raise SystemExit("ERROR: failed to patch get_markers in data_loader.py")
p.write_text(s)
PYEOF

# Replace __init__.py: drop the auto_export machinery, just expose run_app.
cat > "$STAGING/geoapp/__init__.py" <<'PYEOF'
"""Geometry Explorer — interactive PCA visualization of model activations.

Self-contained bundle. The original project uses src.common.auto_export, but
this bundle exposes only what run_geoapp.py needs.
"""

from .server import run_app, create_app

__all__ = ["run_app", "create_app"]
PYEOF

# ---------------------------------------------------------------------------
# Copy pre-built frontend
# ---------------------------------------------------------------------------

echo "==> Copying frontend (dist)..."
mkdir -p "$STAGING/geoapp/frontend"
cp -R "$FRONTEND_DIST" "$STAGING/geoapp/frontend/dist"

# ---------------------------------------------------------------------------
# Copy data — JSON manifests + analysis outputs only.
# Excludes:
#   - sample_*/L*/  (raw activations, never read by the UI)
#   - analysis/embeddings/{umap,tsne}/ (default view uses PCA)
# ---------------------------------------------------------------------------

for ds in "${DATASETS[@]}"; do
  src="$GEO_DIR/$ds"
  dst="$STAGING/data/geo/$ds"
  if [[ ! -d "$src" ]]; then
    echo "WARNING: dataset '$ds' not found at $src — skipping" >&2
    continue
  fi

  echo "==> Copying dataset: $ds"
  mkdir -p "$dst"

  # Top-level manifests
  for f in summary.json config.json; do
    [[ -f "$src/$f" ]] && cp "$src/$f" "$dst/$f"
  done

  # data/ — metadata + sample JSONs (no L*/ dirs)
  mkdir -p "$dst/data"
  [[ -f "$src/data/metadata.json" ]] && cp "$src/data/metadata.json" "$dst/data/metadata.json"
  [[ -f "$src/data/prompt_dataset.json" ]] && cp "$src/data/prompt_dataset.json" "$dst/data/prompt_dataset.json"

  if [[ -d "$src/data/samples" ]]; then
    rsync -a \
      --include='sample_*/' \
      --include='sample_*/*.json' \
      --exclude='sample_*/L*' \
      --exclude='*' \
      "$src/data/samples/" "$dst/data/samples/"

    # Loader probes sample_dirs[0]/L*/ to discover layers — re-create them as
    # empty dirs in sample_0 so that probe still works.
    if [[ -f "$dst/summary.json" ]] && command -v python3 &>/dev/null; then
      # `head -n1` on a pipe trips SIGPIPE under `set -o pipefail`. Use python
      # for the listing too so the whole step is one atomic process.
      python3 - "$dst" <<'PYEOF'
import json, pathlib, sys
dst = pathlib.Path(sys.argv[1])
samples = sorted(
    (p for p in (dst / "data" / "samples").iterdir() if p.is_dir()),
    key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0,
)
if not samples:
    sys.exit(0)
summary = json.loads((dst / "summary.json").read_text())
for layer in summary.get("layers", []):
    (samples[0] / f"L{layer}").mkdir(exist_ok=True)
PYEOF
    fi
  fi

  # analysis/ — PCA + trajectories + linear_probe + relpos counts
  mkdir -p "$dst/analysis"
  [[ -f "$src/analysis/relpos_counts.json" ]] && cp "$src/analysis/relpos_counts.json" "$dst/analysis/relpos_counts.json"
  for sub in pca trajectories linear_probe; do
    if [[ -d "$src/analysis/$sub" ]]; then
      cp -R "$src/analysis/$sub" "$dst/analysis/$sub"
    fi
  done
  if [[ -d "$src/analysis/embeddings/pca" ]]; then
    mkdir -p "$dst/analysis/embeddings"
    cp -R "$src/analysis/embeddings/pca" "$dst/analysis/embeddings/pca"
  fi
done

# ---------------------------------------------------------------------------
# Project files — pyproject, run script, launchers, README
# ---------------------------------------------------------------------------

echo "==> Writing project files..."

cat > "$STAGING/pyproject.toml" <<'TOMLEOF'
[project]
name = "geo-explorer"
version = "1.0.0"
description = "Geometry Explorer — interactive PCA visualization of model activations"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "numpy>=1.24",
]

[tool.uv]
package = false
TOMLEOF

cat > "$STAGING/run_geoapp.py" <<'PYEOF'
#!/usr/bin/env python3
"""Launch the Geometry Explorer UI.

Auto-discovers every dataset under data/geo/ and serves them on a single port.
Open http://127.0.0.1:8000 once the server prints "PRELOAD COMPLETE".
"""

import argparse
import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE_ROOT))

from geoapp import run_app

DATA_ROOT = BUNDLE_ROOT / "data" / "geo"
FRONTEND_DIR = BUNDLE_ROOT / "geoapp" / "frontend" / "dist"


def discover_datasets():
    if not DATA_ROOT.exists():
        return []
    found = []
    for d in sorted(DATA_ROOT.iterdir()):
        if d.is_dir() and (d / "data" / "samples").exists() and (d / "analysis" / "embeddings").exists():
            found.append((d.name, d))
    return found


def main():
    parser = argparse.ArgumentParser(description="Run the Geometry Explorer UI.")
    parser.add_argument("datasets", nargs="*", help="Dataset names to load (default: all under data/geo/)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.datasets:
        data_dirs = []
        for name in args.datasets:
            path = DATA_ROOT / name
            if not path.exists():
                print(f"ERROR: dataset not found: {path}")
                sys.exit(1)
            data_dirs.append((name, path))
    else:
        data_dirs = discover_datasets()

    if not data_dirs:
        print(f"ERROR: no datasets in {DATA_ROOT}")
        sys.exit(1)

    print("=" * 60)
    print("Geometry Explorer")
    print("=" * 60)
    for name, path in data_dirs:
        print(f"  - {name}: {path.relative_to(BUNDLE_ROOT)}")
    print()
    print(f"Server starting at http://{args.host}:{args.port}")
    print(f"Open in browser:  http://{args.host}:{args.port}/{data_dirs[0][0]}")
    print()

    run_app(
        data_dirs=data_dirs,
        frontend_dir=FRONTEND_DIR,
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
PYEOF

cat > "$STAGING/start.sh" <<'SHEOF'
#!/usr/bin/env bash
# Bootstrap launcher. Installs uv if needed, syncs deps, runs the server,
# and opens the UI in your default browser once it's ready.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
HOST="127.0.0.1"

# 1. Ensure uv is on PATH (single-binary Python package manager from Astral).
if ! command -v uv &>/dev/null; then
  echo "==> Installing uv (one-time)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# 2. Install Python deps into a local .venv.
echo "==> Installing dependencies (first run takes ~30s)..."
uv sync --quiet

# 3. Pick the first dataset for the auto-open URL.
FIRST_DATASET="$(ls -1 data/geo 2>/dev/null | head -n1)"
URL="http://${HOST}:${PORT}/${FIRST_DATASET}"

# 4. Open browser shortly after launch (server prints PRELOAD COMPLETE first).
( sleep 6 && open "$URL" 2>/dev/null || true ) &

echo
echo "==> Starting server. Browser will open at: $URL"
echo "    Press Ctrl-C to stop."
echo
exec uv run python run_geoapp.py --host "$HOST" --port "$PORT"
SHEOF
chmod +x "$STAGING/start.sh"

# Mac Finder-friendly: .command files are double-clickable.
cat > "$STAGING/start.command" <<'CMDEOF'
#!/usr/bin/env bash
cd "$(dirname "$0")"
./start.sh
CMDEOF
chmod +x "$STAGING/start.command"

cat > "$STAGING/README.md" <<'MDEOF'
# Geometry Explorer

Interactive visualization of language-model activation geometry across layers,
positions, and time scales. PCA projections, trajectories, and per-sample
inspection — all rendered in your browser, served from a tiny local FastAPI
backend.

## Quick start (macOS)

1. Unzip `geo.zip` (you've already done this if you're reading the file).
2. Double-click **`start.command`** in Finder.
   - The first launch installs `uv` (a fast Python package manager from Astral)
     and syncs dependencies (~30s, ~50MB download).
   - On subsequent launches it boots in a few seconds.
3. Your browser opens to the first dataset automatically.
4. To stop the server, return to the Terminal window and press **Ctrl-C**.

If macOS blocks `start.command` ("can't be opened because Apple cannot check
it"), right-click → **Open** → **Open** in the dialog. Only required once.

## Quick start (terminal, any platform)

```bash
cd geo
./start.sh
```

## What's in the box

```
geo/
├── start.command        # double-click this on Mac
├── start.sh             # or run this in a terminal
├── run_geoapp.py        # entry point (called by the launchers)
├── pyproject.toml       # Python deps (fastapi, uvicorn, numpy)
├── geoapp/              # backend — load-only, no model inference
│   ├── data_loader.py   # reads precomputed PCA from data/geo/<ds>/analysis/
│   ├── routes.py        # /api/<dataset>/... endpoints
│   ├── server.py        # FastAPI + static frontend mount
│   └── frontend/dist/   # prebuilt React UI (Vite)
└── data/geo/
    ├── investment_geometry/      # ~600MB precomputed analysis
    └── investment_horizon_sweep/ # ~250MB precomputed analysis
```

The bundle ships **only the data the UI reads** — JSON manifests, PCA
embeddings, and trajectory tensors. Raw per-layer activations (which the UI
never touches and which weighed in at 110GB+) are stripped.

## Using the UI

- **View modes** (top-left): 2D / 3D scatter, 1D & 2D trajectories along layers
  or positions, scree plots, alignment heatmaps.
- **Color by**: time horizon, time scale, choice type, etc. Time-scale colors
  are a temporal gradient (Seconds = blue → Centuries = red, No Horizon = grey).
- **Filters**: with-horizon / no-horizon toggles, reward & time ranges.
- **Click a point/trajectory** to pin it and see the underlying prompt and
  model response in the right-hand panel.
- **Export**: the camera-icon button downloads a 12× scale PNG. Filenames
  encode the view mode and only the parameters that are fixed for that view
  (e.g. trajectory-across-layers exports omit the layer).

## Datasets

| Name | Samples | Layers | Notes |
|---|---|---|---|
| investment_geometry | 4,588 | 12 | Investment-horizon prompts spanning Seconds → Centuries |
| investment_horizon_sweep | varies | 12 | Sweep over horizon values for the same prompt skeleton |

Each dataset's `analysis/` folder holds the PCA outputs the UI consumes; the
`data/samples/sample_*/` folders hold the prompt + choice metadata for the
sample-detail panel. Empty `L*/` directories under `sample_0/` exist so the
loader can probe layer numbers — they're meant to be empty in this bundle.

## Troubleshooting

- **Port already in use** — set `PORT=9000 ./start.sh` (or any free port).
- **Browser didn't open** — visit `http://127.0.0.1:8000/<dataset-name>` once
  the terminal prints `PRELOAD COMPLETE`.
- **`uv` install fails** — install Homebrew first, then `brew install uv`,
  then re-run `./start.sh`.

## Requirements

- macOS 12+ or any Linux with `bash`, `curl`, and Python 3.10+.
- ~1GB free disk for `.venv` + dataset, ~2GB RAM while running.
- No GPU, no model weights, no internet (after the first dependency install).
MDEOF

# ---------------------------------------------------------------------------
# Zip
# ---------------------------------------------------------------------------

echo
echo "==> Compressing to $ZIP_OUT..."
( cd "$REPO_ROOT/out" && zip -qr geo.zip "$(basename "$STAGING")" )

# Rename top-level dir inside the zip from geo-bundle/ to geo/ so users
# extract a `geo/` folder. Easiest way: re-stage with the renamed top dir.
rm -f "$ZIP_OUT"
TMP_TOP="$REPO_ROOT/out/.geo-bundle-top"
rm -rf "$TMP_TOP"
mkdir -p "$TMP_TOP"
mv "$STAGING" "$TMP_TOP/geo"
( cd "$TMP_TOP" && zip -qr "$ZIP_OUT" "geo" )
mv "$TMP_TOP/geo" "$STAGING"
rmdir "$TMP_TOP"

SIZE=$(du -sh "$ZIP_OUT" | awk '{print $1}')
echo
echo "==> Bundle complete."
echo "    File: $ZIP_OUT  ($SIZE)"
echo "    Staging kept at: $STAGING (delete to reclaim space)"
echo
echo "    To test locally:"
echo "      cd $(dirname "$ZIP_OUT") && unzip -q geo.zip -d /tmp/geo-test && bash /tmp/geo-test/geo/start.sh"
