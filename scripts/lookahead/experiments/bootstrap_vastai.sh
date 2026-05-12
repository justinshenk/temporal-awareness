#!/bin/bash
# bootstrap_vastai.sh — one-shot environment setup for the staircase v2 pipeline.
#
# Usage on a fresh Vast.ai instance:
#   curl -sSL <raw-url-of-this-script> | bash -s -- A     # partition A
# OR
#   bash bootstrap_vastai.sh [partition]                  # if you've cloned manually
#
# Idempotent — safe to re-run after interrupted setup.
#
# What it does:
#   1. Clones the repo (skipped if already present)
#   2. Installs Python deps via pip (transformer dependencies + bitsandbytes
#      for INT8/INT4 quantization on partition A's 70B model)
#   3. Sets up HuggingFace authentication if HF_TOKEN env var is set
#   4. Downloads and extracts Maar et al. supplementary materials (data only)
#   5. Runs a 30-second smoke test on Pythia-410M to verify the pipeline
#   6. (optional) Launches the requested partition

set -euo pipefail

PARTITION="${1:-}"   # optional; if given, will launch partition after setup
WORKDIR="${WORKDIR:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/justinshenk/temporal-awareness.git}"
BRANCH="${BRANCH:-psycoplankton/rq4-lookahead-planning}"

# ──────────────────────────────────────────────────────────────────────
# 1. Clone repo
# ──────────────────────────────────────────────────────────────────────
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ ! -d temporal-awareness ]; then
    echo "[bootstrap] Cloning $REPO_URL..."
    git clone --branch "$BRANCH" "$REPO_URL" temporal-awareness
fi
cd temporal-awareness
echo "[bootstrap] Repo at $(pwd), branch $(git branch --show-current)"

# ──────────────────────────────────────────────────────────────────────
# 2. Install deps
# ──────────────────────────────────────────────────────────────────────
echo "[bootstrap] Installing Python dependencies..."
pip install --break-system-packages -q \
    "transformers>=4.45.0" \
    "torch>=2.1.0" \
    "scikit-learn>=1.3.0" \
    "scipy>=1.11.0" \
    "numpy>=1.24.0" \
    "tqdm>=4.65.0" \
    "huggingface_hub>=0.20.0" \
    "accelerate>=0.30.0" \
    "sentencepiece" \
    "protobuf" 2>&1 | tail -3

# bitsandbytes only when we need INT8/INT4 (mostly partition A's 70B model)
if [ "$PARTITION" = "A" ] || [ "${INSTALL_BNB:-0}" = "1" ]; then
    echo "[bootstrap] Installing bitsandbytes for quantization..."
    pip install --break-system-packages -q bitsandbytes 2>&1 | tail -3
fi

# ──────────────────────────────────────────────────────────────────────
# 3. HuggingFace auth
# ──────────────────────────────────────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
    echo "[bootstrap] Logging into HuggingFace..."
    python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"
elif [ -f ~/.huggingface/token ]; then
    echo "[bootstrap] Using existing HF token at ~/.huggingface/token"
else
    echo "[bootstrap] WARNING: no HF_TOKEN env var or ~/.huggingface/token."
    echo "[bootstrap]          Gated models (Llama, Gemma) will fail without auth."
    echo "[bootstrap]          Run: huggingface-cli login   OR   export HF_TOKEN=hf_..."
fi

# ──────────────────────────────────────────────────────────────────────
# 4. Maar data — download + extract (small, ~1.4MB after filter)
# ──────────────────────────────────────────────────────────────────────
MAAR_DIR="${WORKDIR}/temporal-awareness/data/maar_supplementary_material"
if [ ! -d "$MAAR_DIR/test" ]; then
    echo "[bootstrap] Maar data not found; expecting upload at $MAAR_DIR"
    echo "[bootstrap] Either:"
    echo "[bootstrap]   (a) upload the ZIP and extract to $MAAR_DIR, OR"
    echo "[bootstrap]   (b) set MAAR_DATA_ROOT to wherever you have it."
else
    echo "[bootstrap] Found Maar data at $MAAR_DIR"
fi
export MAAR_DATA_ROOT="$MAAR_DIR"

# ──────────────────────────────────────────────────────────────────────
# 5. Smoke test (30 seconds on Pythia-410M, 20 examples, trivia domain)
# ──────────────────────────────────────────────────────────────────────
echo "[bootstrap] Smoke-testing pipeline on Pythia-410M..."
mkdir -p results/v2_smoke logs/v2_smoke
if python3 scripts/lookahead/experiments/run_staircase_v2.py \
    --model EleutherAI/pythia-410m-deduped \
    --domain trivia \
    --output_dir results/v2_smoke \
    --max_examples 25 \
    --layer_mode workshop_6 \
    --quantization bf16 \
    --overwrite 2>&1 | tail -20
then
    echo "[bootstrap] ✓ Smoke test passed"
    echo "[bootstrap] Output: results/v2_smoke/"
    ls -la results/v2_smoke/
else
    echo "[bootstrap] ✗ Smoke test FAILED — fix before launching partition"
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────
# 6. (Optional) launch the partition
# ──────────────────────────────────────────────────────────────────────
if [ -n "$PARTITION" ]; then
    echo ""
    echo "[bootstrap] Launching partition $PARTITION..."
    echo "[bootstrap] (To run manually: python3 scripts/lookahead/experiments/launch_partition.py --partition $PARTITION)"
    echo ""
    python3 scripts/lookahead/experiments/launch_partition.py \
        --partition "$PARTITION" \
        --output_dir results/v2 \
        --log_dir logs/v2
fi

echo "[bootstrap] Done."
