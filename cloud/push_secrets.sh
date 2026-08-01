#!/usr/bin/env bash
#
# push_secrets.sh — put API keys on a box without exposing them.
#
# Setting a key inline (`at_box.sh "HF_TOKEN=hf_... python ..."`) redacts it
# from the echoed line but still delivers it as SSH argv, so it lands in the
# remote process list, the sshd command log, and the remote shell history. A
# rented box is someone else's hardware; treat anything in argv as published.
#
# The keys travel over STDIN into a 0600 file instead, so no key is ever an
# argument on either side. umask makes the file 0600 at creation rather than
# briefly world-readable before a chmod. Run the pipeline by sourcing it:
#   RUN=x bash cloud/at_box.sh '. /root/.ta_env && bash cloud/run_geometry.sh'
#
# The keys still live on a rented box. Rotate anything pushed here when the
# campaign ends.
#
# Usage:
#   RUN=llama-health bash cloud/push_secrets.sh
#   RUN=llama-health bash cloud/push_secrets.sh --check   # names and lengths only

# shellcheck source=cloud/_config.sh
source "$(cd "$(dirname "$0")" && pwd)/_config.sh"
# shellcheck source=cloud/_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/_lib.sh"
set -euo pipefail

MODE=push
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  --check) MODE=check ;;
  "") ;;
  *) echo "unknown flag: $1" >&2; exit 2 ;;
esac

REMOTE_ENV="${REMOTE_ENV:-/root/.ta_env}"

: "${RUN:?set RUN=<run name> (or INSTANCE=<id>)}"
IID="$(run_instance "$RUN")"
resolve_ssh "$IID"
# shellcheck disable=SC2206  # SSH_EPHEMERAL_OPTS is a deliberate option list
SSH=(ssh $SSH_EPHEMERAL_OPTS -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST")

if [ "$MODE" = "check" ]; then
  # Prints variable NAMES and value LENGTHS. Never the values: this output is
  # meant to be safe to paste into a log.
  "${SSH[@]}" "if [ -f $REMOTE_ENV ]; then
      stat -c '%a %n' $REMOTE_ENV
      while IFS= read -r l; do
        case \"\$l\" in export*) n=\${l#export }; n=\${n%%=*}; v=\${l#*=};
          echo \"  \$n: set, length \${#v}\";; esac
      done < $REMOTE_ENV
    else echo '  $REMOTE_ENV absent'; exit 1; fi"
  exit 0   # set -e already propagated a failing check
fi

: "${HF_TOKEN:?set HF_TOKEN locally first; both models are gated and will 401 without it}"

# Optional keys are written only when set, so a re-push never blanks a key that
# an earlier push put there under a different local environment.
{
  echo "export HF_TOKEN='${HF_TOKEN}'"
  [ -n "${OPENAI_API_KEY:-}" ]    && echo "export OPENAI_API_KEY='${OPENAI_API_KEY}'"
  [ -n "${ANTHROPIC_API_KEY:-}" ] && echo "export ANTHROPIC_API_KEY='${ANTHROPIC_API_KEY}'"
  [ -n "${GEMINI_API_KEY:-}" ]    && echo "export GEMINI_API_KEY='${GEMINI_API_KEY}'"
  echo "export HF_HOME='/root/hf_cache'"
  echo "export TOKENIZERS_PARALLELISM=false"
  # HF's Xet chunked-transfer backend hangs with no timeout when a rented box
  # has a throttled route to the CDN: the load sits at "Fetching N files" with
  # the GPU idle, never errors, and bills the whole time. Plain HTTPS plus a
  # per-request timeout turns that stall into a retry.
  echo "export HF_HUB_DISABLE_XET=1"
  echo "export HF_HUB_DOWNLOAD_TIMEOUT=20"
  true
} | "${SSH[@]}" "umask 077 && cat > $REMOTE_ENV"

echo "[push_secrets] wrote $REMOTE_ENV (0600) over stdin; no key touched argv."
RUN="$RUN" INSTANCE="$IID" SSH_HOST="$SSH_HOST" SSH_PORT="$SSH_PORT" \
  bash "$CLOUD_DIR/push_secrets.sh" --check
