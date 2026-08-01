#!/usr/bin/env bash
#
# sync_up.sh — push this repo UP to a box (local -> cloud). The only direction
# that writes to the remote.
#
# THE EXCLUDES ARE THE WHOLE POINT. out/ alone is 156 GB of activations and
# figures; paper/ is a separate git repo; worktrees/ carries its own .venv.
# Without these, a "sync the code" call uploads the entire history of the
# project over a rented box's uplink and bills for every hour of it. Verify a
# change to this list with DRY_RUN=1 before trusting it.
#
# Usage:
#   RUN=llama-health bash cloud/sync_up.sh
#   RUN=llama-health DRY_RUN=1 bash cloud/sync_up.sh
#   DEST=/tmp/x bash cloud/sync_up.sh          # local dry-run target, no box

# shellcheck source=cloud/_config.sh
source "$(cd "$(dirname "$0")" && pwd)/_config.sh"
# shellcheck source=cloud/_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/_lib.sh"
set -euo pipefail

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  "") ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

# Excludes are anchored to the repo root with a leading slash where the name is
# a top-level directory, so a nested data/results/ is never silently skipped.
# The unanchored entries catch caches that appear at any depth.
EXCLUDES=(
  --exclude='/out/'
  --exclude='/paper/'
  --exclude='/site/'
  --exclude='/notebooks/'
  --exclude='/worktrees/'
  --exclude='/worktree/'
  --exclude='/temp/'
  --exclude='/tmp/'
  --exclude='/results/'
  --exclude='/reports/'
  --exclude='/.backups/'
  --exclude='/cloud/.run_*'
  --exclude='/cloud/.instances_*'
  --exclude='/cloud/.fleet_audit.log'
  --exclude='.git/'
  --exclude='.venv/'
  # The geoapp frontend's node_modules is 262 MB in 10,553 tiny files, 92% of
  # everything a naive sync would send, and nothing on a GPU box reads it.
  --exclude='node_modules/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='*.egg-info/'
  --exclude='.pytest_cache/'
  --exclude='.ruff_cache/'
  --exclude='.mypy_cache/'
  --exclude='.DS_Store'
)

# A dry run itemises every path it would send. A summary would hide the one
# thing worth checking, which is whether a 156 GB directory slipped back in.
DRY=()
if [ "${DRY_RUN:-0}" = "1" ]; then DRY=(--dry-run --stats --itemize-changes); fi

# Retry transient drops; --partial resumes rather than restarting the transfer.
rsync_retry() {
  local n=0
  until rsync -ah ${DRY[@]+"${DRY[@]}"} --timeout=120 --partial "$@"; do
    n=$((n + 1))
    [ "$n" -ge 6 ] && { echo "[sync_up] rsync FAILED after 6 attempts" >&2; return 1; }
    echo "[sync_up] rsync retry $n/6 (transient drop); resuming" >&2
    sleep 8
  done
}

# DEST short-circuits the box entirely. That is how the exclude list is proved
# correct without renting anything.
if [ -n "${DEST:-}" ]; then
  echo "[sync_up] local target: $REPO_ROOT/ -> $DEST/"
  rsync_retry "${EXCLUDES[@]}" "$REPO_ROOT/" "$DEST/"
  echo "[sync_up] done."
  exit 0
fi

: "${RUN:?set RUN=<run name> (or DEST=<dir> for a local dry run)}"
IID="$(run_instance "$RUN")"
resolve_ssh "$IID"

echo "[sync_up] run=$RUN instance=$IID  $REPO_ROOT/ -> $SSH_HOST:$REMOTE_ROOT/"
rsync_retry \
  -e "ssh $SSH_EPHEMERAL_OPTS -i $SSH_KEY -p $SSH_PORT" \
  --rsync-path="mkdir -p $REMOTE_ROOT && rsync" \
  "${EXCLUDES[@]}" \
  "$REPO_ROOT/" "$SSH_USER@$SSH_HOST:$REMOTE_ROOT/"

echo "[sync_up] done."
echo "[sync_up] next: RUN=$RUN bash cloud/at_box.sh 'bash cloud/bootstrap_box.sh'"
