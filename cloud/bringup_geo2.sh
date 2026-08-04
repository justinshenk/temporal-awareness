#!/usr/bin/env bash
#
# bringup_geo2.sh — HOST-SIDE. Take one rented box from bare to running geometry
# extraction: clone the branch, push secrets, install dependencies, start the job.
#
#   RUN=geo2-llama-health MODEL=meta-llama/Llama-3.1-8B-Instruct DOMAIN=health \
#     RUN_NAME=llama31_8b_health_v2 MAX_SAMPLES=3000 bash cloud/bringup_geo2.sh
#
# The code arrives by git clone, not by rsync, so the box's HEAD is a commit id we
# can check. This campaign exists because a defect shipped in cached activations;
# "the box has the fix" has to be a fact read off the box, not an assumption.
#
# Every step is idempotent, so re-running after a dropped session is safe.
#
# Environment:
#   RUN         fleet run name; the box is already labelled ta-tp-<RUN>   (required)
#   MODEL       HF model id                                              (required)
#   RUN_NAME    artifact name, e.g. llama31_8b_health_v2                 (required)
#   DOMAIN      prompt-dataset domain, used to find the config           (required)
#   MAX_SAMPLES cap on samples                                     (default: all)
#   BRANCH      branch to clone            (default exp/turn-geometry-llama-gemma)
#   GATE_ONLY=1 run the gate on the box and stop

# shellcheck source=cloud/_config.sh
source "$(cd "$(dirname "$0")" && pwd)/_config.sh"
# shellcheck source=cloud/_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/_lib.sh"
set -euo pipefail

case "${1:-}" in -h|--help) usage; exit 0 ;; esac

: "${RUN:?set RUN=<fleet run name>}"
: "${MODEL:?set MODEL=<hf model id>}"
: "${RUN_NAME:?set RUN_NAME=<artifact name>}"
: "${DOMAIN:?set DOMAIN=<domain>}"
: "${HF_TOKEN:?set HF_TOKEN locally; the gated models 401 without it}"

BRANCH="${BRANCH:-exp/turn-geometry-llama-gemma}"
CLONE_URL="${CLONE_URL:-https://github.com/justinshenk/temporal-awareness.git}"
DATASET="data/intertemporal/${DOMAIN}/${DOMAIN}_geometry.json"
WANT_COMMIT="$(git -C "$REPO_ROOT" rev-parse "origin/$BRANCH")"

IID="$(run_instance "$RUN")"
echo "[bringup] run=$RUN instance=$IID model=$MODEL artifact=$RUN_NAME"
echo "[bringup] branch=$BRANCH expected commit=$WANT_COMMIT"

at() { RUN="$RUN" bash "$CLOUD_DIR/at_box.sh" "$@"; }

# --- 1. git, then the code ---------------------------------------------------
at "command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }; git --version"

at "set -e
    if [ -d $REMOTE_ROOT/.git ]; then
      cd $REMOTE_ROOT && git fetch --depth 1 origin $BRANCH && git checkout -f FETCH_HEAD
    else
      rm -rf $REMOTE_ROOT
      git clone --depth 1 --branch $BRANCH $CLONE_URL $REMOTE_ROOT
    fi
    cd $REMOTE_ROOT && git rev-parse HEAD"

GOT="$(at "cd $REMOTE_ROOT && git rev-parse HEAD" 2>/dev/null | tr -d '[:space:]')"
if [ "$GOT" != "$WANT_COMMIT" ]; then
  echo "[bringup] REFUSING: box HEAD is $GOT, expected $WANT_COMMIT" >&2
  exit 1
fi
echo "[bringup] box HEAD verified: $GOT"

# The fix itself, checked by content rather than by commit id alone.
at "cd $REMOTE_ROOT && grep -c 'token_ids=traj.token_ids' src/intertemporal/preference/preference_querier.py && grep -c 'hook_embed' src/inference/backends/backend_huggingface.py"

# --- 2. secrets and dependencies ---------------------------------------------
RUN="$RUN" HF_TOKEN="$HF_TOKEN" bash "$CLOUD_DIR/push_secrets.sh"
at "bash cloud/bootstrap_box.sh" | tail -20

# --- 3. the job ---------------------------------------------------------------
JOB_ENV="RUN_NAME=$RUN_NAME MODEL=$MODEL DATASET=$DATASET"
[ -n "${MAX_SAMPLES:-}" ] && JOB_ENV="$JOB_ENV MAX_SAMPLES=$MAX_SAMPLES"
[ "${GATE_ONLY:-0}" = "1" ] && JOB_ENV="$JOB_ENV GATE_ONLY=1"

at ". /root/.ta_env && cd $REMOTE_ROOT && mkdir -p out/logs && \
    $JOB_ENV nohup setsid bash cloud/jobs/geo2_job.sh > out/logs/geo2_${RUN_NAME}.nohup 2>&1 < /dev/null & \
    sleep 3; echo started"

echo "[bringup] started. Follow it with:"
echo "  RUN=$RUN bash cloud/at_box.sh \"grep '^PHASE ' out/logs/geo2_${RUN_NAME}.log\""
