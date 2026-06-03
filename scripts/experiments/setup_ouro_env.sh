#!/usr/bin/bash
# ── One-time setup: create a separate venv for Ouro-2.6B ──
# Ouro requires transformers<4.56.0 but our main env has transformers==5.3.0.
# This creates a lightweight venv that inherits most packages from the main env
# but pins transformers==4.54.1.
#
# Run once on Sherlock login node:
#   bash scripts/experiments/setup_ouro_env.sh

set -euo pipefail

OURO_ENV_DIR="${OURO_ENV_DIR:-/home/groups/barbarae/molofsky/ouro-env}"

echo "Creating Ouro-compatible venv at: $OURO_ENV_DIR"

module load python/3.12.1

# Create venv with access to system/main env packages
python3 -m venv --system-site-packages "$OURO_ENV_DIR"

source "$OURO_ENV_DIR/bin/activate"

# Pin transformers to a version compatible with Ouro
pip install --break-system-packages "transformers==4.54.1" 2>/dev/null || \
    pip install "transformers==4.54.1"

echo ""
echo "Ouro venv ready at: $OURO_ENV_DIR"
echo "transformers version: $(python3 -c 'import transformers; print(transformers.__version__)')"
echo ""
echo "Submit scripts will auto-detect Ouro-2.6B and use this env."
