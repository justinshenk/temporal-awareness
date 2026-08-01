#!/usr/bin/env bash
#
# at_box.sh — run one command on a run's box over SSH and stream its output.
#
# Pushing code is sync_up.sh; this only sends a command. Secrets are redacted
# from the echoed line, but the command still travels as SSH argv and is visible
# in the remote process list, so pass keys by sourcing the env file that
# push_secrets.sh wrote, never inline.
#
# Usage:
#   RUN=llama-health bash cloud/at_box.sh 'nvidia-smi'
#   RUN=llama-health bash cloud/at_box.sh '. /root/.ta_env && bash cloud/run_geometry.sh'
#   INSTANCE=12345678 bash cloud/at_box.sh 'uptime'

# shellcheck source=cloud/_config.sh
source "$(cd "$(dirname "$0")" && pwd)/_config.sh"
# shellcheck source=cloud/_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/_lib.sh"
set -euo pipefail

case "${1:-}" in
  -h|--help|"") usage; exit 0 ;;
esac

: "${RUN:?set RUN=<run name> (or INSTANCE=<id>)}"
CMD="$*"

IID="$(run_instance "$RUN")"
resolve_ssh "$IID"

echo "[at_box] run=$RUN instance=$IID host=$SSH_HOST:$SSH_PORT" >&2
printf '[at_box] $ %s\n' "$(printf '%s' "$CMD" | redact)" >&2

# cd only if the repo is already there; before the first sync_up it is not.
# shellcheck disable=SC2086  # SSH_EPHEMERAL_OPTS is a deliberate option list
exec ssh $SSH_EPHEMERAL_OPTS -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" \
  "cd $REMOTE_ROOT 2>/dev/null || true; $CMD"
