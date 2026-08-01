#!/usr/bin/env bash
#
# bootstrap_box.sh — BOX-SIDE. Bring a freshly rented box to the point where
# the geometry driver can run. Run it after sync_up.sh has put the repo at
# $REMOTE_ROOT:
#
#   RUN=llama-health bash cloud/at_box.sh 'bash cloud/bootstrap_box.sh'
#
# git is not optional here. pyproject pins two dependencies as git+https, so
# `uv sync` fails on the stock image, which ships without git.
#
# Every step is idempotent, so re-running after a dropped SSH session is safe
# and cheap.

# shellcheck source=cloud/_config.sh
source "$(cd "$(dirname "$0")" && pwd)/_config.sh"
# shellcheck source=cloud/_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/_lib.sh"
set -uo pipefail

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

export PATH="$HOME/.local/bin:$PATH"
export DEBIAN_FRONTEND=noninteractive
LOG="${LOG:-/root/bootstrap_box.log}"

say() { printf '%s [bootstrap] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
die() { say "FAILED: $*"; exit 1; }

cd "$REPO_ROOT" || die "repo not found at $REPO_ROOT; run sync_up.sh first"
say "repo=$REPO_ROOT host=$(hostname)"

# --- 1. git ------------------------------------------------------------------
if command -v git >/dev/null; then
  say "git already present: $(git --version)"
else
  say "installing git"
  apt-get update -qq >>"$LOG" 2>&1 || die "apt-get update"
  apt-get install -y -qq git >>"$LOG" 2>&1 || die "apt-get install git"
  command -v git >/dev/null || die "git still missing after install"
  say "installed $(git --version)"
fi

# --- 2. uv -------------------------------------------------------------------
if command -v uv >/dev/null; then
  say "uv already present: $(uv --version)"
else
  say "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >>"$LOG" 2>&1 || die "uv install"
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || die "uv not on PATH after install"
  say "installed $(uv --version)"
fi

# --- 3. dependencies ---------------------------------------------------------
# The image already carries a CUDA torch. `uv sync` builds its own venv, so the
# resolved torch is the one that must see the GPU; that is what step 4 checks.
say "uv sync (this pulls the git+https dependencies)"
uv sync >>"$LOG" 2>&1 || { tail -n 40 "$LOG"; die "uv sync"; }

# --- 4. the GPU is real ------------------------------------------------------
# An instance can be "running" with no usable GPU (driver mismatch, a container
# that lost its device). Extraction would then fall back to CPU and take days
# while billing GPU rates, so this is a hard gate, not a warning.
say "checking torch sees a GPU"
uv run python - <<'PY' 2>&1 | tee -a "$LOG"
import sys, torch, transformers, transformer_lens
print(f"python              {sys.version.split()[0]}")
print(f"torch               {torch.__version__}")
print(f"transformers        {transformers.__version__}")
print(f"transformer_lens    {transformer_lens.__version__}")
print(f"cuda available      {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("NO GPU VISIBLE TO TORCH")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"gpu[{i}]              {p.name}  {p.total_memory / 1024**3:.1f} GB")
PY
[ "${PIPESTATUS[0]}" = "0" ] || die "torch cannot see a GPU"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | tee -a "$LOG"
df -h "$REPO_ROOT" | tail -n 1 | tee -a "$LOG"

say "OK — box is ready. Next: push_secrets, then cloud/run_geometry.sh"
