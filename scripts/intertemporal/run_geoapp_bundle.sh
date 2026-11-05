#!/bin/bash
# Run the geoapp on an analysis-only geo bundle (zip with analysis outputs +
# per-sample JSONs, no raw activations) — e.g. a bundle attached to a GitHub
# release. Downloads (if a URL), extracts into out/geo/, and launches the app.
#
# Accepts either a plain zip, or a .manifest for bundles that are published as
# split parts (GitHub release assets reassembled and checksum-verified here).
#
# Usage:
#   ./scripts/intertemporal/run_geoapp_bundle.sh <zip-or-manifest, path-or-url>
#
# Example:
#   ./scripts/intertemporal/run_geoapp_bundle.sh \
#     https://github.com/justinshenk/temporal-awareness/releases/download/geo-bundles/investment_geometry_analysis.zip.manifest

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_DIR="${GEO_BASE_DIR:-$PROJECT_ROOT/out/geo}"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <zip-or-manifest, path-or-url>" >&2
    exit 1
fi

SOURCE="$1"
mkdir -p "$BASE_DIR"

is_url() {
    [[ "$1" == http://* || "$1" == https://* ]]
}

# Retrying, resumable download
fetch() {
    local url="$1" dest="$2" try
    for try in 1 2 3 4 5; do
        if curl -fL --retry 3 -C - --progress-bar -o "$dest" "$url"; then
            return 0
        fi
        echo "Download failed (attempt $try), retrying: $url" >&2
        sleep 2
    done
    echo "Error: could not download $url" >&2
    return 1
}

if [[ "$SOURCE" == *.manifest ]]; then
    # Manifest format: "sha256 <hash>" then one "part <filename>" line per part,
    # with parts hosted alongside the manifest.
    ZIP_NAME="$(basename "$SOURCE" .manifest)"
    ZIP_PATH="$BASE_DIR/$ZIP_NAME"
    MANIFEST="$BASE_DIR/$ZIP_NAME.manifest"

    if is_url "$SOURCE"; then
        fetch "$SOURCE" "$MANIFEST"
        PART_BASE="$(dirname "$SOURCE")"
    else
        cp "$SOURCE" "$MANIFEST"
        PART_BASE="$(cd "$(dirname "$SOURCE")" && pwd)"
    fi

    EXPECTED_SHA="$(awk '$1 == "sha256" {print $2}' "$MANIFEST")"
    PARTS="$(awk '$1 == "part" {print $2}' "$MANIFEST")"
    if [ -z "$EXPECTED_SHA" ] || [ -z "$PARTS" ]; then
        echo "Error: malformed manifest $MANIFEST" >&2
        exit 1
    fi

    if [ -f "$ZIP_PATH" ] && echo "$EXPECTED_SHA  $ZIP_PATH" | shasum -a 256 -c - >/dev/null 2>&1; then
        echo "Using already-assembled $ZIP_PATH (checksum OK)"
    else
        PARTS_DIR="$BASE_DIR/.$ZIP_NAME.parts"
        mkdir -p "$PARTS_DIR"
        for part in $PARTS; do
            echo "Fetching $part ..."
            if is_url "$SOURCE"; then
                fetch "$PART_BASE/$part" "$PARTS_DIR/$part"
            else
                cp "$PART_BASE/$part" "$PARTS_DIR/$part"
            fi
        done
        echo "Assembling $ZIP_NAME ..."
        for part in $PARTS; do cat "$PARTS_DIR/$part"; done > "$ZIP_PATH"
        echo "Verifying checksum ..."
        if ! echo "$EXPECTED_SHA  $ZIP_PATH" | shasum -a 256 -c -; then
            echo "Error: checksum mismatch for $ZIP_PATH — delete it and retry." >&2
            exit 1
        fi
        rm -rf "$PARTS_DIR"
    fi
elif is_url "$SOURCE"; then
    ZIP_PATH="$BASE_DIR/$(basename "$SOURCE")"
    if [ -f "$ZIP_PATH" ]; then
        echo "Using already-downloaded $ZIP_PATH"
    else
        echo "Downloading $SOURCE ..."
        fetch "$SOURCE" "$ZIP_PATH"
    fi
else
    ZIP_PATH="$SOURCE"
    if [ ! -f "$ZIP_PATH" ]; then
        echo "Error: zip not found: $ZIP_PATH" >&2
        exit 1
    fi
fi

# The bundle contains a single top-level dataset directory (e.g. investment_geometry/)
DATASET=$(unzip -Z1 "$ZIP_PATH" | cut -d/ -f1 | sort -u)
if [ -z "$DATASET" ] || [ "$(echo "$DATASET" | wc -l)" -ne 1 ]; then
    echo "Error: expected a single top-level dataset directory inside $ZIP_PATH, got:" >&2
    echo "$DATASET" >&2
    exit 1
fi

TARGET="$BASE_DIR/$DATASET"
if [ -e "$TARGET" ]; then
    echo "Dataset already present at $TARGET — not overwriting."
    echo "(Delete or rename it first if you want to re-extract the bundle.)"
else
    echo "Extracting $DATASET into $BASE_DIR ..."
    unzip -q "$ZIP_PATH" -d "$BASE_DIR"
fi

exec "$SCRIPT_DIR/run_geoapp.sh" "$DATASET"
