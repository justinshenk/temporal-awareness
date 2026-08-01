#!/usr/bin/env bash
#
# reap.sh — destroy instances THIS PROJECT launched. The only sanctioned destroy.
#
# Refuses, loudly, to touch anything that is not in .instances_ours AND does not
# carry our label prefix. Other agents share this account; their boxes are not
# ours to reclaim, however idle they look.
#
# Destroying is irreversible and takes the box's disk with it, so by default this
# also refuses unless the run's artifacts are already verified on local disk.
# --skip-verify exists for boxes that never produced artifacts (a failed launch,
# a bootstrap that died); it does not exist for boxes that ran an experiment.
#
#   bash cloud/reap.sh --list                       # what would be reaped
#   bash cloud/reap.sh --all --yes-i-am-really-sure
#   bash cloud/reap.sh 12345678 --yes-i-am-really-sure
#   SKIP_VERIFY=1 bash cloud/reap.sh 12345678 --yes-i-am-really-sure

source "$(cd "$(dirname "$0")" && pwd)/_config.sh"

CONFIRM=0; ALL=0; LIST=0; ids=()
for a in "$@"; do
  case "$a" in
    --yes-i-am-really-sure) CONFIRM=1 ;;
    --all)  ALL=1 ;;
    --list) LIST=1 ;;
    --skip-verify) SKIP_VERIFY=1 ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *)  ids+=("$a") ;;
  esac
done

JS="$(vast_instances_json)"

if [ "$ALL" = "1" ]; then
  mapfile_compat() { while IFS= read -r l; do [ -n "$l" ] && ids+=("$l"); done < "$1"; }
  mapfile_compat "$LEDGER_OURS"
fi

if [ "${#ids[@]}" -eq 0 ]; then
  echo "Nothing named. Use --all to reap every instance in .instances_ours, or pass ids." >&2
  bash "$CLOUD_DIR/fleet_status.sh"
  exit 2
fi

# ---- Gate 1: ownership ------------------------------------------------------
approved=()
for id in "${ids[@]}"; do
  if is_ours "$id" "$JS"; then
    approved+=("$id")
    echo "  [ours]     $id — eligible"
  elif ledger_has "$LEDGER_FOREIGN" "$id"; then
    echo "  [FOREIGN]  $id — another agent's instance. REFUSING." >&2
  elif ledger_has "$LEDGER_OURS" "$id"; then
    echo "  [stale]    $id — in our ledger but gone from the account (or label mismatch). Skipping." >&2
  else
    echo "  [UNKNOWN]  $id — not in either ledger. REFUSING; run fleet_status.sh first." >&2
  fi
done

[ "${#approved[@]}" -eq 0 ] && { echo "Nothing eligible to reap."; exit 0; }
[ "$LIST" = "1" ] && { echo "(--list, stopping before destroy)"; exit 0; }

# ---- Gate 2: artifacts captured --------------------------------------------
if [ "${SKIP_VERIFY:-0}" != "1" ]; then
  if [ -x "$REPO_ROOT/scripts/verify_experiment_output.py" ] || [ -f "$REPO_ROOT/scripts/verify_experiment_output.py" ]; then
    echo "[reap] Gate 2 — verifying pulled artifacts before teardown."
    if ! "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/verify_experiment_output.py" --pulled; then
      echo "[reap] REFUSED — artifacts are not verified on local disk." >&2
      echo "[reap] The box(es) are UNTOUCHED and still billing. Pull first." >&2
      exit 1
    fi
  else
    echo "[reap] REFUSED — scripts/verify_experiment_output.py does not exist yet," >&2
    echo "[reap] so 'artifacts are safe' cannot be established. Pass SKIP_VERIFY=1" >&2
    echo "[reap] only for boxes that never produced artifacts." >&2
    exit 1
  fi
fi

[ "$CONFIRM" = "1" ] || { echo "Refusing: pass --yes-i-am-really-sure (destroy is irreversible)." >&2; exit 2; }

for id in "${approved[@]}"; do
  echo "[reap] destroying $id"
  # `vastai destroy instance` prompts interactively; without an answer it aborts
  # while still exiting 0, so a pipeline exit code is NOT evidence of teardown.
  echo "y" | vastai destroy instance "$id" 2>&1 | sed 's/^/    /'

  # Confirm against the API that it is actually gone before touching the ledger
  # or writing an audit line. An audit log that records destroys that did not
  # happen is worse than no audit log.
  gone=0
  for _ in 1 2 3 4 5 6; do
    sleep 5
    if ! printf '%s' "$(vast_instances_json)" | ID="$id" python3 -c '
import sys, json, os
iid = int(os.environ["ID"])
sys.exit(0 if any(r.get("id") == iid for r in json.load(sys.stdin)) else 1)
'; then gone=1; break; fi
  done

  if [ "$gone" = "1" ]; then
    echo "    confirmed gone from the account"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DESTROY id=$id (confirmed)" >> "$CLOUD_DIR/.fleet_audit.log"
    if grep -vx -- "$id" "$LEDGER_OURS" > "$LEDGER_OURS.tmp"; then
      mv "$LEDGER_OURS.tmp" "$LEDGER_OURS"
    else
      # grep exits 1 when nothing remains; that is an empty ledger, not an error.
      : > "$LEDGER_OURS"
      rm -f "$LEDGER_OURS.tmp"
    fi
  else
    echo "[reap] destroy NOT confirmed for $id — still on the account. Leaving it" >&2
    echo "[reap] in the ledger so it stays reapable. Check it manually." >&2
  fi
done

echo
bash "$CLOUD_DIR/fleet_status.sh"
