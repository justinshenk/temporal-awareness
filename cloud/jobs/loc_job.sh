#!/usr/bin/env bash
#
# loc_job.sh — BOX-SIDE. One coarse activation-patching localization sweep,
# streamed to the Hugging Face dataset while it runs.
#
#   . /root/.ta_env && RUN_NAME=loc_llama_investment \
#     MODEL=meta-llama/Llama-3.1-8B-Instruct \
#     DATASET=data/intertemporal/investment/investment_local.json \
#     nohup setsid bash cloud/jobs/loc_job.sh &
#
# The box's disk is the only copy of anything not yet pushed, and a box can
# vanish without warning, so the uploader follows the run directory from the
# moment it appears instead of waiting for the sweep to finish.
#
# Progress is written as PHASE lines:
#   grep '^PHASE ' out/logs/loc_<RUN_NAME>.log
#
# Environment:
#   RUN_NAME       output directory name under out/experiments   (required)
#   MODEL          HF model id                                   (required)
#   DATASET        dataset config path                           (required)
#   N_PAIRS        contrastive pairs to sweep                    (default 24)
#   COMPONENTS     JSON list of components                       (default the three coarse ones)
#   HF_REPO        dataset repo to stream into                   (default unrulyabstractions/temporal-awareness)
#   HF_PREFIX      path inside the repo                          (default localization/<RUN_NAME>)
#   SYNC_INTERVAL  seconds between upload passes                 (default 120)
#   PY             interpreter                                   (default /venv/main/bin/python)

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/root/temporal-awareness}"
cd "$REPO_ROOT" || { echo "repo not found at $REPO_ROOT" >&2; exit 1; }
# shellcheck source=/dev/null
[ -f /root/.ta_env ] && . /root/.ta_env

: "${RUN_NAME:?set RUN_NAME}"
: "${MODEL:?set MODEL}"
: "${DATASET:?set DATASET}"
N_PAIRS="${N_PAIRS:-24}"
COMPONENTS="${COMPONENTS:-[\"resid_post\", \"attn_out\", \"mlp_out\"]}"
HF_REPO="${HF_REPO:-unrulyabstractions/temporal-awareness}"
HF_PREFIX="${HF_PREFIX:-localization/$RUN_NAME}"
SYNC_INTERVAL="${SYNC_INTERVAL:-120}"
PY="${PY:-/venv/main/bin/python}"

RUN_DIR="$REPO_ROOT/out/experiments/$RUN_NAME"
LOG="$REPO_ROOT/out/logs/loc_${RUN_NAME}.log"
HF_LOG="$REPO_ROOT/out/logs/hf_sync_${RUN_NAME}.log"
SYNC="scripts/intertemporal/sync_geometry_to_hf.py"
mkdir -p "$(dirname "$LOG")"

ts()    { date -u +%Y-%m-%dT%H:%M:%SZ; }
phase() { printf 'PHASE %-16s ts=%s %s\n' "$1" "$(ts)" "${2:-}" | tee -a "$LOG"; }

phase START "run=$RUN_NAME model=$MODEL dataset=$DATASET n_pairs=$N_PAIRS"
[ -f "$DATASET" ] || { phase FAILED "dataset_missing:$DATASET"; exit 1; }
"$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
  || { phase FAILED "no_gpu_visible_to_torch"; exit 1; }
phase GPU_OK ""

COARSE_JSON="{\"enabled\": true, \"components\": $COMPONENTS, \"layer_steps\": [1], \"pos_steps\": []}"
phase SWEEP_START "coarse=$COARSE_JSON"
# TransformerLens weight processing materializes fp32 copies and peaks at
# several times model size in host RAM, which SIGKILLs 4-8B loads. The stored
# campaign runs used process_weights=False, so matching it also keeps this
# sweep comparable to them.
export TA_TL_NO_PROCESS=1
"$PY" scripts/intertemporal/run_intertemporal_experiment.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --backend transformerlens \
  --n_pairs "$N_PAIRS" \
  --out "$RUN_NAME" \
  --only-viz-agg \
  --disable \
  --coarse "$COARSE_JSON" >>"$LOG" 2>&1 &
EXP_PID=$!

# The uploader can only start once the run directory exists, and it must not
# create it first: run_intertemporal_experiment.py moves an existing output
# directory aside, so a pre-created empty one would push the real run elsewhere.
for _ in $(seq 1 720); do
  [ -d "$RUN_DIR" ] && break
  kill -0 "$EXP_PID" 2>/dev/null || break
  sleep 5
done
if [ -d "$RUN_DIR" ]; then
  nohup setsid "$PY" "$SYNC" --watch --run-dir "$RUN_DIR" --repo "$HF_REPO" \
    --prefix "$HF_PREFIX" --done-file RUN_COMPLETE --interval "$SYNC_INTERVAL" \
    </dev/null >>"$HF_LOG" 2>&1 &
  HF_PID=$!
  sleep 5
  if kill -0 "$HF_PID" 2>/dev/null; then
    phase HF_WATCH_STARTED "pid=$HF_PID prefix=$HF_PREFIX log=$HF_LOG"
  else
    HF_PID=""
    tail -20 "$HF_LOG" | tee -a "$LOG"
    phase HF_WATCH_DIED "see:$HF_LOG"
  fi
else
  HF_PID=""
  phase HF_WATCH_SKIPPED "run_dir_never_appeared:$RUN_DIR"
fi

wait "$EXP_PID"
RC=$?
phase SWEEP_END "rc=$RC dir=$RUN_DIR"

# A sweep that dies still has to release the watcher, so the marker is written
# either way. It means "no more data is coming", never "the run succeeded".
touch "$RUN_DIR/RUN_COMPLETE"

# rc=137 is a SIGKILL, which on these boxes means the loader was OOM-killed.
# Such a run leaves configs and logs and no pairs at all, and publishing that
# under the normal name puts a 5 KB archive on the Hub that reads as a result.
NPAIRS_DONE=$(ls "$RUN_DIR/pairs" 2>/dev/null | wc -l | tr -d ' ')
if [ "$RC" != "0" ] || [ "${NPAIRS_DONE:-0}" -eq 0 ]; then
  phase RUN_FAILED "rc=$RC pairs=$NPAIRS_DONE — publishing diagnostics only"
  ARCHIVE_PREFIX="FAILED_"
else
  ARCHIVE_PREFIX=""
fi
if [ -n "$HF_PID" ]; then
  for _ in $(seq 1 180); do
    kill -0 "$HF_PID" 2>/dev/null || break
    sleep 10
  done
  kill "$HF_PID" 2>/dev/null
fi

phase HF_FINAL_PASS ""
"$PY" "$SYNC" --run-dir "$RUN_DIR" --repo "$HF_REPO" --prefix "$HF_PREFIX" \
  --once --verify >>"$HF_LOG" 2>&1
VERIFY_RC=$?
phase HF_VERIFY "rc=$VERIFY_RC log=$HF_LOG"

ARCHIVE="$REPO_ROOT/out/${ARCHIVE_PREFIX}${RUN_NAME}.tar.gz"
tar czf "$ARCHIVE" -C "$REPO_ROOT/out/experiments" "$RUN_NAME"
ARCHIVE="$ARCHIVE" REPO_ID="$HF_REPO" RUN_NAME="${ARCHIVE_PREFIX}${RUN_NAME}" "$PY" - <<'PY' >>"$LOG" 2>&1
import os
from huggingface_hub import HfApi
api = HfApi()
path = f"localization/{os.environ['RUN_NAME']}.tar.gz"
api.upload_file(
    path_or_fileobj=os.environ["ARCHIVE"],
    path_in_repo=path,
    repo_id=os.environ["REPO_ID"],
    repo_type="dataset",
)
info = api.get_paths_info(os.environ["REPO_ID"], [path], repo_type="dataset")
print("ARCHIVE_ON_HUB", path, [f.size for f in info])
PY
phase ARCHIVE_UPLOADED "$(du -h "$ARCHIVE" | cut -f1) $ARCHIVE"

[ "$RC" = "0" ] || exit "$RC"
phase DONE "run=$RUN_NAME"
