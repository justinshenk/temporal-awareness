#!/usr/bin/env bash
#
# _config.sh — shared config for the temporal-awareness vast.ai fleet.
# Source this from every other script in cloud/. It defines nothing that spends.
#
# OWNERSHIP IS THE POINT OF THIS FILE. Multiple agents share this vast account.
# We may only ever stop, destroy, or reap an instance that WE launched. Every
# other box on the account belongs to someone else's workstream and is off
# limits, no matter how idle it looks.

set -uo pipefail

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CLOUD_DIR/.." && pwd)"

export VASTAI_API_KEY="${VASTAI_API_KEY:-$(cat "$HOME/.config/vastai/vast_api_key" 2>/dev/null)}"

# --- Ownership ledgers -------------------------------------------------------
# OURS:     instance ids this project created. Only these may ever be destroyed.
# FOREIGN:  instance ids observed on the account that we did NOT create. Recorded
#           once at baseline and appended to on every scan. Never destroyed.
export LEDGER_OURS="$CLOUD_DIR/.instances_ours"
export LEDGER_FOREIGN="$CLOUD_DIR/.instances_foreign"
touch "$LEDGER_OURS" "$LEDGER_FOREIGN"

# Every instance we create is labelled with this prefix. The label is the
# cross-check on the ledger: if a box is in .instances_ours but its label does
# not start with this, something is wrong and we refuse to touch it.
export LABEL_PREFIX="${LABEL_PREFIX:-ta-tp}"

export REMOTE_ROOT="${REMOTE_ROOT:-/root/temporal-awareness}"

# --- Helpers -----------------------------------------------------------------

# vast_instances_json — every instance on the account, walking pagination.
# instances-v1 pages at 25; a box on page 2 reads as "missing" if you don't walk.
vast_instances_json() {
  python3 - <<'PY'
import json, subprocess, sys
out, token = [], None
for _ in range(40):
    cmd = ["vastai", "show", "instances-v1", "--raw"]
    if token:
        cmd += ["--next-token", token]
    p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        d = json.loads(p.stdout)
    except Exception:
        break
    rows = d if isinstance(d, list) else d.get("instances", d.get("results", []))
    out.extend(r for r in rows if isinstance(r, dict))
    token = d.get("next_token") if isinstance(d, dict) else None
    if not token or not rows:
        break
print(json.dumps(out))
PY
}

ledger_has() {  # ledger_has <file> <id>
  grep -qx -- "$2" "$1" 2>/dev/null
}

ledger_add_ours() {  # ledger_add_ours <id> <run>
  grep -qx -- "$1" "$LEDGER_OURS" 2>/dev/null || echo "$1" >> "$LEDGER_OURS"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) LAUNCH id=$1 run=${2:-?}" >> "$CLOUD_DIR/.fleet_audit.log"
}

# is_ours — the ONLY sanctioned ownership test. Requires BOTH:
#   (a) the id is recorded in .instances_ours, AND
#   (b) the live label starts with LABEL_PREFIX.
# Belt and braces on purpose: a stale ledger line must never authorise a destroy.
is_ours() {  # is_ours <id> [instances_json]
  local id="$1" js="${2:-}"
  ledger_has "$LEDGER_OURS" "$id" || return 1
  [ -n "$js" ] || js="$(vast_instances_json)"
  # Must export: an inline VAR=x assignment binds to the first command in the
  # pipeline (printf), not to the python3 that actually reads the environment.
  export ID="$id" PREFIX="$LABEL_PREFIX"
  printf '%s' "$js" | python3 -c '
import sys, json, os
iid = int(os.environ["ID"]); pref = os.environ["PREFIX"]
row = next((r for r in json.load(sys.stdin) if r.get("id") == iid), None)
# Gone from the account: nothing to destroy, treat as not-ours so callers no-op.
sys.exit(0 if row is not None and str(row.get("label") or "").startswith(pref) else 1)
'
}
