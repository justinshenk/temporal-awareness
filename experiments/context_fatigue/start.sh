#!/usr/bin/env bash
set -euo pipefail

# Ensure we always fall through to sleep so the user can SSH in to debug
trap 'echo "ERROR: startup failed — SSH in to debug"; exec sleep infinity' ERR

# --- Validate env vars ---
if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "WARNING: GITHUB_TOKEN not set — git push will not work"
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY not set — W&B logging will not work"
fi

# --- Configure git credentials ---
if [ -n "${GITHUB_TOKEN:-}" ]; then
    cat > ~/.netrc <<EOF
machine github.com
login x-access-token
password ${GITHUB_TOKEN}
EOF
    chmod 600 ~/.netrc
fi

# --- Configure git identity ---
git config --global user.name "${GIT_USER_NAME:-researcher}"
git config --global user.email "${GIT_USER_EMAIL:-researcher@runpod}"

# --- Clone repo (skip if already exists) ---
REPO_DIR="/workspace/temporal-awareness"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning temporal-awareness..."
    git clone --recurse-submodules -b context-fatigue-datasets \
        https://github.com/justinshenk/temporal-awareness.git "$REPO_DIR"
else
    echo "Repo already exists at $REPO_DIR, skipping clone"
fi

# --- Install Python dependencies ---
echo "Installing dependencies with uv..."
cd "$REPO_DIR/experiments/context_fatigue"
uv sync

# --- Verify GPU access (non-fatal) ---
echo "Checking GPU access..."
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected — CPU only')" || true

echo "Ready. SSH in and run experiments."
exec sleep infinity
