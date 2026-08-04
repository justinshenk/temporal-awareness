#!/usr/bin/env bash
#
# geo2_job.sh — BOX-SIDE. One turn-geometry extraction, gated on proof that the
# saved activations sit at the positions they claim, and streamed to the Hub in
# parts while it runs.
#
#   . /root/.ta_env && RUN_NAME=llama31_8b_health_v2 \
#     MODEL=meta-llama/Llama-3.1-8B-Instruct \
#     DATASET=data/intertemporal/health/health_geometry.json \
#     MAX_SAMPLES=3000 \
#     nohup setsid bash cloud/jobs/geo2_job.sh &
#
# The gate is the reason this job exists. The previous campaign cached
# activations over a re-templated prompt+response, so every position labelled
# chat_suffix held a response token and nothing errored. verify_turn_positions.py
# compares the cache against embedding ground truth and refuses to let a box that
# cannot prove its positions produce data.
#
# Delivery is in parts. A file-level sync of one run is about 450,000 tiny files,
# and the dataset repo already carries 92,000 from the last campaign. Re-uploading
# a whole 3 GB tarball every 15 minutes instead would keep 60 GB of dead LFS
# history per run. So each pass archives only the samples finished since the last
# one, uploads that part, and the run ends with a single consolidated archive.
# Nothing older than SNAP_INTERVAL ever exists only on the rented disk.
#
# Progress is written as PHASE lines:
#   grep '^PHASE ' out/logs/geo2_<RUN_NAME>.log
#
# Environment:
#   RUN_NAME       run + artifact name, e.g. llama31_8b_health_v2   (required)
#   MODEL          HF model id                                      (required)
#   DATASET        dataset config path                              (required)
#   MAX_SAMPLES    cap on samples                                   (default: all)
#   COMPONENTS     components to extract          (default: resid_post attn_out)
#   DTYPE          storage dtype                                    (default float16)
#   HF_REPO        dataset repo                   (default unrulyabstractions/temporal-awareness)
#   SNAP_INTERVAL  seconds between part uploads                     (default 900)
#   PY_CMD         interpreter                                      (default: uv run python)
#   GATE_ONLY=1    run the gate, then stop before extraction

set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
REPO_ROOT="${REPO_ROOT:-/root/temporal-awareness}"
cd "$REPO_ROOT" || { echo "repo not found at $REPO_ROOT" >&2; exit 1; }
# shellcheck source=/dev/null  # written on the box by push_secrets.sh
[ -f /root/.ta_env ] && . /root/.ta_env

: "${RUN_NAME:?set RUN_NAME}"
: "${MODEL:?set MODEL}"
: "${DATASET:?set DATASET}"

# shellcheck disable=SC2206  # deliberate word splitting into a command array
PY=(${PY_CMD:-uv run python})
COMPONENTS="${COMPONENTS:-resid_post attn_out}"
DTYPE="${DTYPE:-float16}"
HF_REPO="${HF_REPO:-unrulyabstractions/temporal-awareness}"
SNAP_INTERVAL="${SNAP_INTERVAL:-900}"

RUN_DIR="$REPO_ROOT/out/geo/$RUN_NAME"
SAMPLES_DIR="$RUN_DIR/data/samples"
PARTS_DIR="$REPO_ROOT/out/parts/$RUN_NAME"
CURSOR="$RUN_DIR/.part_cursor"
LOG="$REPO_ROOT/out/logs/geo2_${RUN_NAME}.log"
GATE_JSON="$REPO_ROOT/out/logs/gate_${RUN_NAME}.json"
mkdir -p "$RUN_DIR" "$PARTS_DIR" "$(dirname "$LOG")"

ts()    { date -u +%Y-%m-%dT%H:%M:%SZ; }
log()   { printf '%s %s\n' "$(ts)" "$*" | tee -a "$LOG"; }
phase() { printf 'PHASE %-16s ts=%s %s\n' "$1" "$(ts)" "${2:-}" | tee -a "$LOG"; }

phase START "run=$RUN_NAME model=$MODEL dataset=$DATASET dir=$RUN_DIR"
log "commit $(git rev-parse HEAD 2>/dev/null || echo unknown) on $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

[ -f "$DATASET" ] || { phase FAILED "dataset_missing:$DATASET"; exit 1; }
"${PY[@]}" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
  || { phase FAILED "no_gpu_visible_to_torch"; exit 1; }
phase GPU_OK ""

# --- Hub helper --------------------------------------------------------------
# Uploads one file and prints the size the Hub reports beside the local size, so
# a truncated transfer is visible rather than assumed away.
hub_put() {  # hub_put <local file> <path in repo>
  LOCAL="$1" INREPO="$2" REPO_ID="$HF_REPO" "${PY[@]}" - <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi()
local, path, repo = os.environ["LOCAL"], os.environ["INREPO"], os.environ["REPO_ID"]
size = os.path.getsize(local)
api.upload_file(path_or_fileobj=local, path_in_repo=path, repo_id=repo, repo_type="dataset")
info = api.get_paths_info(repo, [path], repo_type="dataset")
remote = info[0].size if info else None
print(f"HUB_PUT path={path} local_bytes={size} hub_bytes={remote} "
      f"match={remote == size}")
sys.exit(0 if remote == size else 1)
PY
}

# --- The gate ----------------------------------------------------------------
# Expectations live here, in code, so what counts as this model's turn tokens is
# reviewable rather than typed at launch time.
case "$MODEL" in
  *Qwen*|*qwen*)
    EXPECT=(--expect '<|im_end|>' --expect '<|im_start|>' --expect 'assistant') ;;
  *Llama*|*llama*)
    EXPECT=(--expect '<|eot_id|>' --expect '<|start_header_id|>'
            --expect 'assistant' --expect '<|end_header_id|>') ;;
  *gemma*|*Gemma*)
    EXPECT=(--expect '<end_of_turn>' --expect '<start_of_turn>' --expect 'model') ;;
  *Mistral*|*mistral*)
    EXPECT=(--expect '[/INST]') ;;
  *)
    phase FAILED "no_turn_token_expectation_for:$MODEL"; exit 1 ;;
esac

phase GATE_START "expect=${EXPECT[*]}"
"${PY[@]}" scripts/intertemporal/verify_turn_positions.py \
  --config "$DATASET" --model "$MODEL" --n-samples 2 \
  "${EXPECT[@]}" --out "$GATE_JSON" 2>&1 | tee -a "$LOG"
GATE_RC="${PIPESTATUS[0]}"

# The report is uploaded either way. A failed gate is evidence, not something to
# leave on a disk that gets destroyed.
[ -f "$GATE_JSON" ] && hub_put "$GATE_JSON" "geometry/${RUN_NAME}_gate.json" 2>&1 | tee -a "$LOG"

if [ "$GATE_RC" != "0" ]; then
  phase GATE_FAILED "rc=$GATE_RC report=geometry/${RUN_NAME}_gate.json"
  exit 1
fi
phase GATE_PASS "report=geometry/${RUN_NAME}_gate.json"
[ "${GATE_ONLY:-0}" = "1" ] && { phase GATE_ONLY_DONE ""; exit 0; }

# --- Part uploads ------------------------------------------------------------
# Sample directories are written in order (sample_0, sample_1, ...), so a single
# integer cursor records what is already on the Hub. The highest-numbered
# directory may be half written, so a pass always stops one short of it.
[ -f "$CURSOR" ] || echo 0 > "$CURSOR"

# Returns 0 on a successful upload, 2 when there is nothing new to send, and 1
# when a pass failed and must be retried. The final flush distinguishes the last
# two: "nothing left" is the success it is waiting for.
flush_part() {  # flush_part [final]
  local final="${1:-}" cursor last upto part list tarball
  cursor="$(cat "$CURSOR" 2>/dev/null || echo 0)"
  last="$(ls "$SAMPLES_DIR" 2>/dev/null | sed -n 's/^sample_\([0-9]*\)$/\1/p' | sort -n | tail -1)"
  [ -n "$last" ] || return 2
  if [ "$final" = "final" ]; then upto="$last"; else upto=$((last - 1)); fi
  [ "$upto" -ge "$cursor" ] || return 2

  part="$(printf 'part_%04d' "$cursor")"
  list="$PARTS_DIR/$part.list"
  : > "$list"
  if [ "$cursor" = "0" ]; then
    for f in config.json summary.json data/metadata.json data/prompt_dataset.json; do
      [ -f "$RUN_DIR/$f" ] && echo "$f" >> "$list"
    done
  fi
  local i
  for i in $(seq "$cursor" "$upto"); do
    [ -d "$SAMPLES_DIR/sample_$i" ] && echo "data/samples/sample_$i" >> "$list"
  done
  [ -s "$list" ] || return 2

  tarball="$PARTS_DIR/$part.tar.gz"
  tar czf "$tarball" -C "$RUN_DIR" -T "$list" || { log "tar failed for $part"; return 1; }
  if hub_put "$tarball" "geometry/${RUN_NAME}_parts/$part.tar.gz" 2>&1 | tee -a "$LOG"; then
    echo $((upto + 1)) > "$CURSOR"
    phase PART_UPLOADED "$part samples=$cursor..$upto bytes=$(wc -c < "$tarball" | tr -d " ")"
    rm -f "$tarball"
    return 0
  fi
  log "part upload FAILED for $part; keeping it on disk and retrying next pass"
  return 1
}

part_loop() {
  while :; do
    sleep "$SNAP_INTERVAL"
    flush_part
  done
}

part_loop &
LOOP_PID=$!
phase PART_LOOP_STARTED "pid=$LOOP_PID interval=${SNAP_INTERVAL}s"

# --- Extraction --------------------------------------------------------------
GEN_ARGS=(--config "$DATASET" --model "$MODEL" --resume "$RUN_DIR"
          --turn-only --dtype "$DTYPE")
# shellcheck disable=SC2206  # deliberate word splitting into an argument list
GEN_ARGS+=(--components ${COMPONENTS})
[ -n "${MAX_SAMPLES:-}" ] && GEN_ARGS+=(--max-samples "$MAX_SAMPLES")

phase EXTRACT_START "args=${GEN_ARGS[*]}"
"${PY[@]}" scripts/intertemporal/generate_geometry_samples.py "${GEN_ARGS[@]}" 2>&1 | tee -a "$LOG"
RC="${PIPESTATUS[0]}"

kill "$LOOP_PID" 2>/dev/null
pkill -P "$LOOP_PID" 2>/dev/null
sleep 10
phase PART_LOOP_STOPPED "pid=$LOOP_PID"

N_SAMPLES="$(ls "$SAMPLES_DIR" 2>/dev/null | grep -c '^sample_')"
N_NPY="$(find "$RUN_DIR" -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')"
if [ "$RC" != "0" ]; then
  phase EXTRACT_FAILED "rc=$RC samples=$N_SAMPLES npy=$N_NPY"
else
  phase EXTRACT_DONE "rc=0 samples=$N_SAMPLES npy=$N_NPY size=$(du -sh "$RUN_DIR" | cut -f1)"
fi

# Whatever the extraction did, everything on disk goes up before anything else.
# Keep flushing until a pass reports nothing left, so a run that finished many
# samples inside the last interval does not lose them.
FLUSH_OK=0
for _ in $(seq 1 20); do
  flush_part final
  case "$?" in
    0) continue ;;
    2) FLUSH_OK=1; break ;;
    *) sleep 30 ;;
  esac
done
phase PARTS_FLUSHED "complete=$FLUSH_OK cursor=$(cat "$CURSOR" 2>/dev/null)"
[ "$FLUSH_OK" = "1" ] || phase PARTS_INCOMPLETE "some samples are only on this box"

# --- Consolidated archive ----------------------------------------------------
ARCHIVE="$REPO_ROOT/out/${RUN_NAME}.tar.gz"
tar czf "$ARCHIVE" -C "$REPO_ROOT/out/geo" "$RUN_NAME" || { phase FAILED "tar_archive"; exit 1; }
phase ARCHIVE_BUILT "bytes=$(wc -c < "$ARCHIVE" | tr -d " ") path=$ARCHIVE"
if hub_put "$ARCHIVE" "geometry/${RUN_NAME}.tar.gz" 2>&1 | tee -a "$LOG"; then
  phase ARCHIVE_UPLOADED "geometry/${RUN_NAME}.tar.gz"
else
  phase ARCHIVE_UPLOAD_FAILED "geometry/${RUN_NAME}.tar.gz"
  exit 1
fi

[ "$RC" = "0" ] || exit "$RC"
phase DONE "run=$RUN_NAME samples=$N_SAMPLES"
