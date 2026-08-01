#!/usr/bin/env bash
#
# _lib.sh — helpers shared by every script in cloud/. Sourced, not executed.
#
# usage() prints a script's own leading comment block, so the header IS the
# help text and the two can never drift apart.
#
# A run name is the unit the orchestrator thinks in ("llama-health"), so every
# host-side script takes RUN and resolves the id here. The id lives in
# cloud/.run_<RUN>.id, written by launch_run.sh at creation time.
#
# vastai 1.0.13 removed `ssh-url` and `show instances` (both hit the v0 API and
# now return HTTP 410), so the endpoint is read from the paginated v1 listing.
# That listing pages at 25: a box on page 2 resolves to an EMPTY endpoint unless
# every page is walked, and every sync/at_box against it then fails forever.

# Discard host keys. Vast reuses the same [sshN.vast.ai]:PORT endpoints across
# different boxes with different host keys, so a persistent known_hosts
# guarantees "REMOTE HOST IDENTIFICATION HAS CHANGED" rejections that hang as
# "waiting for sshd". Acceptable for throwaway GPU boxes.
# shellcheck disable=SC2034  # consumed by the scripts that source this
SSH_EPHEMERAL_OPTS="-F /dev/null -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

# usage — the calling script's header comment, shebang and hashes stripped.
usage() {
  awk 'NR > 1 { if (/^#/) { sub(/^#[ ]?/, ""); print; next } exit }' "$0"
}

run_id_file() {  # run_id_file <run>
  printf '%s/.run_%s.id' "$CLOUD_DIR" "$1"
}

run_env_file() {  # run_env_file <run> — model/domain recorded at launch
  printf '%s/.run_%s.env' "$CLOUD_DIR" "$1"
}

# run_instance — the instance id for a run. INSTANCE in the environment wins, so
# a box can be addressed before its id file exists.
run_instance() {  # run_instance <run>
  local run="$1" f
  if [ -n "${INSTANCE:-}" ]; then printf '%s' "$INSTANCE"; return 0; fi
  f="$(run_id_file "$run")"
  [ -s "$f" ] || { echo "No instance recorded for run '$run' ($f). Launch it first." >&2; return 1; }
  tr -d '[:space:]' < "$f"
}

# instance_field — one top-level field of an instance row, walking every page.
instance_field() {  # instance_field <id> <field>
  IID="$1" FIELD="$2" python3 -c '
import json, os, subprocess, sys
iid, field = int(os.environ["IID"]), os.environ["FIELD"]
token = None
for _ in range(40):  # hard page cap so a runaway token can never hang us
    cmd = ["vastai", "show", "instances-v1", "--raw"]
    if token:
        cmd += ["--next-token", token]
    try:
        d = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    except Exception:
        break
    rows = d if isinstance(d, list) else d.get("instances", d.get("results", []))
    for r in rows:
        if isinstance(r, dict) and r.get("id") == iid:
            print(r.get(field) if r.get(field) is not None else "")
            sys.exit(0)
    token = d.get("next_token") if isinstance(d, dict) else None
    if not token or not rows:
        break
sys.exit(1)
'
}

# resolve_ssh — set SSH_USER/SSH_HOST/SSH_PORT for an instance id. Exported
# SSH_HOST+SSH_PORT short-circuit the API walk, so a caller that already
# resolved the endpoint does not pay for it again.
resolve_ssh() {  # resolve_ssh <instance-id>
  local iid="$1" line
  if [ -n "${SSH_HOST:-}" ] && [ -n "${SSH_PORT:-}" ]; then
    SSH_USER="${SSH_USER:-root}"; return 0
  fi
  line="$(IID="$iid" python3 -c '
import json, os, subprocess, sys
iid = int(os.environ["IID"])
token = None
for _ in range(40):
    cmd = ["vastai", "show", "instances-v1", "--raw"]
    if token:
        cmd += ["--next-token", token]
    try:
        d = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    except Exception:
        break
    rows = d if isinstance(d, list) else d.get("instances", d.get("results", []))
    for r in rows:
        if isinstance(r, dict) and r.get("id") == iid:
            print((r.get("ssh_host") or ""), (r.get("ssh_port") or ""))
            sys.exit(0)
    token = d.get("next_token") if isinstance(d, dict) else None
    if not token or not rows:
        break
')"
  SSH_USER="root"
  SSH_HOST="${line%% *}"
  SSH_PORT="${line##* }"
  if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
    echo "Could not resolve an SSH endpoint for instance $iid." >&2
    echo "Is it running?  bash cloud/fleet_status.sh" >&2
    return 1
  fi
}

# redact — blank out anything that looks like a key before echoing a command.
# at_box.sh echoes what it runs, and those lines land in orchestrator logs.
redact() {
  sed -E 's/([A-Za-z_]*(KEY|TOKEN|SECRET|PASSWORD)[A-Za-z_]*=)[^ ]+/\1<redacted>/g'
}
