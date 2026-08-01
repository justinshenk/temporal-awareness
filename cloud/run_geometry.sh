#!/usr/bin/env bash
#
# run_geometry.sh — BOX-SIDE driver for one scoped geometry run.
#
#   RUN=llama-health bash cloud/at_box.sh \
#     '. /root/.ta_env && RUN=llama-health MODEL=meta-llama/Llama-3.1-8B-Instruct \
#      DOMAIN=health nohup setsid bash cloud/run_geometry.sh &'
#
# Two things make this run affordable.
#
# Scope. We extract ONLY the change-of-turn window, the chat template's
# generation suffix, not all 17 semantic positions. That is ~3 GB of
# activations per model instead of ~112 GB.
#
# Streaming. sync_geometry_to_hf.py runs in the BACKGROUND for the whole
# extraction, so activations reach the Hub while the GPU is still working. The
# box's disk is the only copy of anything that has not been pushed yet, and a
# box can vanish, so nothing waits for the end of the run to be uploaded.
#
# Progress is written as PHASE lines the orchestrator greps:
#   grep '^PHASE ' out/logs/run_geometry_<RUN>.log
#
# Environment:
#   RUN, MODEL, DOMAIN            required
#   POSITIONS   semantic positions to extract  (default: chat_suffix chat_suffix_tail)
#   COMPONENTS  components to extract          (default: the script's own list)
#   DTYPE       storage dtype for activations  (default: the generator's float32)
#   MAX_SAMPLES cap on samples                 (default: all)
#   HF_REPO     dataset repo to stream into    (default: unrulyabstractions/temporal-awareness)
#   N_LAYERS    override the model's depth instead of reading its config
#   PY_CMD      interpreter                    (default: uv run python)
#   PREFLIGHT_ONLY=1  run every guard, then stop before loading the model

# shellcheck source=cloud/_config.sh
source "$(cd "$(dirname "$0")" && pwd)/_config.sh"
# shellcheck source=cloud/_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/_lib.sh"
set -uo pipefail

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

export PATH="$HOME/.local/bin:$PATH"
# shellcheck source=/dev/null  # written on the box by push_secrets.sh
[ -f /root/.ta_env ] && . /root/.ta_env

: "${RUN:?set RUN=<run name>}"
: "${MODEL:?set MODEL=<hf model id>}"
: "${DOMAIN:?set DOMAIN=<domain>}"

# The interpreter. `uv run python` on a bootstrapped box; overridable so the
# driver's guards can be exercised locally against an existing venv.
# shellcheck disable=SC2206  # deliberate word splitting into a command array
PY=(${PY_CMD:-uv run python})

POSITIONS="${POSITIONS:-chat_suffix chat_suffix_tail}"
HF_REPO="${HF_REPO:-unrulyabstractions/temporal-awareness}"
HF_DRAIN_SECONDS="${HF_DRAIN_SECONDS:-180}"

cd "$REPO_ROOT" || { echo "repo not found at $REPO_ROOT" >&2; exit 1; }

GEN="scripts/intertemporal/generate_geometry_samples.py"
HF_SYNC="scripts/intertemporal/sync_geometry_to_hf.py"
DATASET_CFG="data/intertemporal/${DOMAIN}/${DOMAIN}_geometry.json"

# A deterministic output directory, obtained through --resume rather than a
# timestamp. The background uploader has to know the path before extraction
# starts, and a re-run after a dropped session has to land in the same place
# instead of starting a second copy beside it.
RUN_DIR="$REPO_ROOT/out/geo/$RUN"
LOG="$REPO_ROOT/out/logs/run_geometry_${RUN}.log"
HF_LOG="$REPO_ROOT/out/logs/hf_sync_${RUN}.log"
LOCK="$REPO_ROOT/out/logs/.lock_${RUN}"
mkdir -p "$RUN_DIR" "$(dirname "$LOG")"

ts()    { date -u +%Y-%m-%dT%H:%M:%SZ; }
log()   { printf '%s %s\n' "$(ts)" "$*" | tee -a "$LOG"; }
phase() { printf 'PHASE %-18s ts=%s %s\n' "$1" "$(ts)" "${2:-}" | tee -a "$LOG"; }
die()   { phase FAILED "reason=$*"; exit 1; }

# Two drivers on one GPU means OOM or corrupt half-written activations, so an
# in-flight run wins and a second invocation stands down.
if ! mkdir "$LOCK" 2>/dev/null; then
  if pgrep -f "generate_geometry_samples.py.*$RUN_DIR" >/dev/null; then
    phase ALREADY_RUNNING "run=$RUN lock=$LOCK"
    exit 0
  fi
  log "stale lock $LOCK with no live extractor; taking it over"
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

phase START "run=$RUN model=$MODEL domain=$DOMAIN dir=$RUN_DIR"

[ -f "$DATASET_CFG" ] || die "dataset_config_missing:$DATASET_CFG"
[ -f "$GEN" ]         || die "generator_missing:$GEN"

# --- Layers ------------------------------------------------------------------
# The hardcoded layer lists were written for Qwen3-4B (36 layers) and contain 34
# and 35, which do not exist on a 32-layer Llama. Truncating them would silently
# drop the late band the analysis rests on, so they are projected by fractional
# depth and then validated.
LAYERS="$(MODEL="$MODEL" N_LAYERS="${N_LAYERS:-}" "${PY[@]}" - <<'PY'
import os
from src.intertemporal.common.model_layers import scale_layers, validate_layers
from src.intertemporal.geometry.geometry_utils import LAYERS

n = os.environ.get("N_LAYERS") or ""
if not n:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(os.environ["MODEL"])
    n = cfg.num_hidden_layers
n = int(n)
print(" ".join(str(x) for x in validate_layers(scale_layers(LAYERS, n), n)))
PY
)" || die "layer_projection"
[ -n "$LAYERS" ] || die "layer_projection_empty"
phase LAYERS_RESOLVED "model=$MODEL layers=[${LAYERS// /,}] positions=[${POSITIONS// /,}]"

# --- Pre-flight the scoped flags --------------------------------------------
# Passing the full 17-position default would extract ~112 GB and fill the disk.
# If the generator cannot be scoped, refuse: a run that silently ignores the
# scope is worse than no run, because it looks like it worked.
HELP="$("${PY[@]}" "$GEN" --help 2>&1)" || { printf '%s\n' "$HELP" | tail -20; die "generator_help_failed"; }
MISSING=""
for f in --layers --positions; do
  case "$HELP" in *"$f"*) ;; *) MISSING="$MISSING $f" ;; esac
done
[ -z "$MISSING" ] || die "generator_lacks_scope_flags:${MISSING// /,}"

# shellcheck disable=SC2206  # deliberate word splitting into an argument list
GEN_ARGS=(--config "$DATASET_CFG" --model "$MODEL" --resume "$RUN_DIR"
          --layers ${LAYERS} --positions ${POSITIONS})
# shellcheck disable=SC2206  # deliberate word splitting into an argument list
[ -n "${COMPONENTS:-}" ]  && GEN_ARGS+=(--components ${COMPONENTS})
[ -n "${MAX_SAMPLES:-}" ] && GEN_ARGS+=(--max-samples "$MAX_SAMPLES")
# Left unset on purpose: the generator's own default (float32) decides the
# stored precision, so the driver never silently halves it behind the analysis.
[ -n "${DTYPE:-}" ]       && GEN_ARGS+=(--dtype "$DTYPE")

# --- The GPU is real ---------------------------------------------------------
# A box can lose its device after bootstrap. Extraction would then fall back to
# CPU, take days, and bill GPU rates the whole time, so this is a gate.
if "${PY[@]}" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  phase GPU_OK ""
elif [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  phase GPU_MISSING "preflight_only=1, not fatal here"
else
  die "no_gpu_visible_to_torch"
fi

# PREFLIGHT_ONLY exercises every guard above without loading a model. Use it to
# check a box, or this driver, without spending GPU time or pulling weights.
if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  phase PREFLIGHT_OK "args=${GEN_ARGS[*]}"
  exit 0
fi

# --- Background upload -------------------------------------------------------
HF_PID=""
if [ ! -f "$HF_SYNC" ]; then
  phase HF_WATCH_SKIPPED "reason=missing:$HF_SYNC"
elif ! "${PY[@]}" "$HF_SYNC" --help 2>&1 | grep -q -- --watch; then
  phase HF_WATCH_SKIPPED "reason=no_watch_flag:$HF_SYNC"
elif pgrep -f "sync_geometry_to_hf.py.*$RUN_DIR" >/dev/null; then
  HF_PID="$(pgrep -f "sync_geometry_to_hf.py.*$RUN_DIR" | head -1)"
  phase HF_WATCH_REUSED "pid=$HF_PID log=$HF_LOG"
else
  nohup setsid "${PY[@]}" "$HF_SYNC" \
    --watch --run-dir "$RUN_DIR" --repo "$HF_REPO" --prefix "geometry/$RUN" \
    </dev/null >>"$HF_LOG" 2>&1 &
  HF_PID=$!
  sleep 5
  if kill -0 "$HF_PID" 2>/dev/null; then
    phase HF_WATCH_STARTED "pid=$HF_PID repo=$HF_REPO log=$HF_LOG"
  else
    # Without the uploader the box's disk is the only copy of a multi-hour run,
    # and a box can vanish. A bad flag or missing Hub token dies here in five
    # seconds, which is far cheaper than discovering it after the GPU time is
    # spent, so this is fatal rather than a skip.
    HF_PID=""
    tail -20 "$HF_LOG" >&2 || true
    die "hf_watch_died_on_start:see:$HF_LOG"
  fi
fi

# --- Extraction --------------------------------------------------------------
phase EXTRACT_START "cmd=$GEN"
log "args: ${GEN_ARGS[*]}"
"${PY[@]}" "$GEN" "${GEN_ARGS[@]}" 2>&1 | tee -a "$LOG"
RC="${PIPESTATUS[0]}"
if [ "$RC" != "0" ]; then
  phase EXTRACT_FAILED "rc=$RC"
else
  N_NPY="$(find "$RUN_DIR/data" -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')"
  BYTES="$(du -sh "$RUN_DIR" 2>/dev/null | cut -f1)"
  phase EXTRACT_DONE "rc=0 npy=$N_NPY size=$BYTES dir=$RUN_DIR"
fi

# --- Drain the uploader ------------------------------------------------------
# The watcher is behind the extractor by whatever it has not pushed yet. Give it
# time to finish before stopping it, then say what is still on the box.
if [ -n "$HF_PID" ]; then
  phase HF_DRAIN "pid=$HF_PID seconds=$HF_DRAIN_SECONDS"
  sleep "$HF_DRAIN_SECONDS"
  kill "$HF_PID" 2>/dev/null
  phase HF_WATCH_STOPPED "pid=$HF_PID log=$HF_LOG"
fi

[ "$RC" = "0" ] || exit "$RC"
phase DONE "run=$RUN dir=$RUN_DIR"
