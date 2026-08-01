#!/usr/bin/env bash
#
# fleet_status.sh — show every instance on the shared vast account, split into
# OURS (this project launched it) and FOREIGN (another agent's — do not touch).
#
# Spends nothing. Run it before and after anything that launches or destroys.
#
# Any instance seen that is not in .instances_ours is recorded in
# .instances_foreign, so the "not ours" set only ever grows. That is deliberate:
# a box can never be reclassified from foreign to ours by a later scan.
#
#   bash cloud/fleet_status.sh
#   bash cloud/fleet_status.sh --quiet     # exit 0/1 only, for gating

source "$(cd "$(dirname "$0")" && pwd)/_config.sh"

QUIET=0
[ "${1:-}" = "--quiet" ] && QUIET=1

JS="$(vast_instances_json)"

# Record anything not already claimed as ours into the foreign ledger.
printf '%s' "$JS" | python3 -c '
import sys, json, os
ours = set()
try:
    ours = {l.strip() for l in open(os.environ["LEDGER_OURS"]) if l.strip()}
except OSError:
    pass
seen = set()
try:
    seen = {l.strip() for l in open(os.environ["LEDGER_FOREIGN"]) if l.strip()}
except OSError:
    pass
new = [str(r["id"]) for r in json.load(sys.stdin)
       if str(r.get("id")) not in ours and str(r.get("id")) not in seen]
if new:
    with open(os.environ["LEDGER_FOREIGN"], "a") as f:
        for i in new:
            f.write(i + "\n")
    print("  [ledger] recorded %d new FOREIGN instance(s): %s" % (len(new), ", ".join(new)),
          file=sys.stderr)
'

[ "$QUIET" = "1" ] && exit 0

LEDGER_OURS="$LEDGER_OURS" LABEL_PREFIX="$LABEL_PREFIX" printf '%s' "$JS" | python3 -c '
import sys, json, os, datetime
rows = json.load(sys.stdin)
ours = set()
try:
    ours = {l.strip() for l in open(os.environ["LEDGER_OURS"]) if l.strip()}
except OSError:
    pass
pref = os.environ["LABEL_PREFIX"]
now = datetime.datetime.now(datetime.UTC)

def fmt(r):
    dph = float(r.get("dph_total") or 0)
    age = spent = ""
    if r.get("start_date"):
        st = datetime.datetime.fromtimestamp(float(r["start_date"]), datetime.UTC)
        h = (now - st).total_seconds() / 3600
        age, spent = "%.1fh" % h, "$%.2f" % (h * dph)
    return ("  id=%-10s %-10s %sx %-14s $%.3f/hr  %-6s %-7s label=%s"
            % (r.get("id"), r.get("actual_status"), r.get("num_gpus"),
               r.get("gpu_name"), dph, age, spent, r.get("label")))

mine  = [r for r in rows if str(r.get("id")) in ours]
theirs= [r for r in rows if str(r.get("id")) not in ours]

print("=== OURS (%d) — this project launched these; only these may be reaped ===" % len(mine))
for r in mine or []:
    line = fmt(r)
    if not str(r.get("label") or "").startswith(pref):
        line += "   << LABEL MISMATCH: will NOT be reaped"
    print(line)
if not mine:
    print("  (none)")

print()
print("=== FOREIGN (%d) — another agent owns these. DO NOT TOUCH. ===" % len(theirs))
for r in theirs or []:
    print(fmt(r))
if not theirs:
    print("  (none)")

burn_mine = sum(float(r.get("dph_total") or 0) for r in mine)
burn_all  = sum(float(r.get("dph_total") or 0) for r in rows)
print()
print("burn: ours $%.3f/hr  |  account total $%.3f/hr ($%.2f/day)"
      % (burn_mine, burn_all, burn_all * 24))
'
