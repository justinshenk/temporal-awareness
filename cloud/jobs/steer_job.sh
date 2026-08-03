#!/usr/bin/env bash
#
# steer_job.sh — BOX-SIDE. Re-score the best steering cell of every model with
# per-prompt logging, then bootstrap confidence intervals from those scores.
#
#   . /root/.ta_env && nohup setsid bash cloud/jobs/steer_job.sh &
#
# The stored campaign vectors come from the Hub, and each model's re-score is
# pushed back as soon as it exists, so a box that dies costs one model rather
# than the whole set.
#
# steer_ci_rescore.py asserts that the recomputed S, S_ctrl and baseline match
# the stored summary. A model that fails that gate stops here and is reported;
# confidence intervals from a mismatched setup would describe a different run.
#
# Progress is written as PHASE lines:
#   grep '^PHASE ' out/logs/steer_ci.log
#
# Environment:
#   MODELS         space-separated HF model ids   (default: the four campaign models)
#   HF_REPO        dataset repo                   (default unrulyabstractions/temporal-awareness)
#   HF_PREFIX      path inside the repo           (default steering/ci_bootstrap)
#   PY             interpreter                    (default /venv/main/bin/python)

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/root/temporal-awareness}"
cd "$REPO_ROOT" || { echo "repo not found at $REPO_ROOT" >&2; exit 1; }
# shellcheck source=/dev/null
[ -f /root/.ta_env ] && . /root/.ta_env

MODELS="${MODELS:-Qwen/Qwen3-4B-Instruct-2507 meta-llama/Llama-3.1-8B-Instruct google/gemma-2-9b-it mistralai/Mistral-7B-Instruct-v0.3}"
HF_REPO="${HF_REPO:-unrulyabstractions/temporal-awareness}"
HF_PREFIX="${HF_PREFIX:-steering/ci_bootstrap}"
# The stored campaign sweep scored with steer_turn_preference.py's own default
# of 16. Batch composition changes padding, and padding changes bfloat16
# reduction order, so a re-score at another batch size is not the same setup.
BATCH_SIZE="${BATCH_SIZE:-16}"
PY="${PY:-/venv/main/bin/python}"

OUT_DIR="$REPO_ROOT/out/steering_ci"
MIRROR="$REPO_ROOT/out/hf_new"
LOG="$REPO_ROOT/out/logs/steer_ci.log"
SYNC="scripts/intertemporal/sync_geometry_to_hf.py"
mkdir -p "$(dirname "$LOG")" "$OUT_DIR"

ts()    { date -u +%Y-%m-%dT%H:%M:%SZ; }
phase() { printf 'PHASE %-16s ts=%s %s\n' "$1" "$(ts)" "${2:-}" | tee -a "$LOG"; }

push() {  # push <label> — one upload pass over everything written so far
  "$PY" "$SYNC" --run-dir "$OUT_DIR" --repo "$HF_REPO" --prefix "$HF_PREFIX" \
    --once >>"$LOG" 2>&1
  phase PUSHED "after=$1 rc=$?"
}

phase START "models=$MODELS batch_size=$BATCH_SIZE"
"$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
  || { phase FAILED "no_gpu_visible_to_torch"; exit 1; }
phase GPU_OK ""

REPO_ID="$HF_REPO" MIRROR="$MIRROR" "$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id=os.environ["REPO_ID"],
    repo_type="dataset",
    allow_patterns="steering/extreme_sweep/*",
    local_dir=os.environ["MIRROR"],
)
print("MIRROR_AT", path)
PY
phase MIRROR_READY "$(find "$MIRROR/steering/extreme_sweep" -type f | wc -l | tr -d ' ') files"

FAILED=""
for model in $MODELS; do
  short="${model##*/}"
  phase RESCORE_START "model=$model"
  "$PY" scripts/scratch/steer_ci_rescore.py --model "$model" --batch-size "$BATCH_SIZE" >>"$LOG" 2>&1
  rc=$?
  if [ "$rc" != "0" ]; then
    phase RESCORE_FAILED "model=$model rc=$rc (gate or load); see $LOG"
    FAILED="$FAILED $short"
    continue
  fi
  phase RESCORE_OK "model=$model $(grep "^GATE $short" "$LOG" | tail -1)"
  push "$short"
done

phase BOOTSTRAP_START ""
"$PY" scripts/scratch/steer_ci_bootstrap.py >>"$LOG" 2>&1
BOOT_RC=$?
phase BOOTSTRAP_END "rc=$BOOT_RC"
push bootstrap

"$PY" "$SYNC" --run-dir "$OUT_DIR" --repo "$HF_REPO" --prefix "$HF_PREFIX" \
  --once --verify >>"$LOG" 2>&1
phase HF_VERIFY "rc=$? log=$LOG"

[ -z "$FAILED" ] || { phase FAILED "models:$FAILED"; exit 1; }
[ "$BOOT_RC" = "0" ] || exit "$BOOT_RC"
phase DONE ""
