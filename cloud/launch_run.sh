#!/usr/bin/env bash
#
# launch_run.sh — rent ONE vast.ai box for ONE geometry run.
#
# This is the only script here that spends money. Everything up to the create
# call is read-only, so --search-only exercises the whole selection path for
# free.
#
# The account is shared with other agents. Every box we create carries the
# LABEL_PREFIX label and is recorded in .instances_ours, because reap.sh will
# only destroy something that has BOTH. A box created without the label can
# never be reaped by us.
#
# Usage:
#   RUN=llama-health MODEL=meta-llama/Llama-3.1-8B-Instruct DOMAIN=health \
#     MIN_VRAM=40 DISK=120 MAX_PRICE=0.90 bash cloud/launch_run.sh
#
#   MIN_VRAM=40 DISK=120 MAX_PRICE=0.90 bash cloud/launch_run.sh --search-only
#   YES=1 ... bash cloud/launch_run.sh          # skip the interactive confirm
#
# Environment:
#   RUN         run name; the box is labelled "${LABEL_PREFIX}-${RUN}"   (required)
#   MODEL       HF model id, recorded for the driver                     (required)
#   DOMAIN      prompt-dataset domain, recorded for the driver           (required)
#   MIN_VRAM    minimum per-GPU VRAM in GB                               (default 40)
#   DISK        instance disk in GB; permanent after creation            (default 120)
#   MAX_PRICE   maximum $/hr                                             (default 1.20)
#   MIN_REL     minimum host reliability, filtered client-side           (default 0.98)
#   NUM_GPUS    GPUs per box                                             (default 1)
#   YES=1       do not prompt before creating
#   RELAUNCH=1  allow a new box when this run's recorded id is gone

# shellcheck source=cloud/_config.sh
source "$(cd "$(dirname "$0")" && pwd)/_config.sh"
# shellcheck source=cloud/_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/_lib.sh"
set -euo pipefail

SEARCH_ONLY=0
for a in "$@"; do
  case "$a" in
    -h|--help) usage; exit 0 ;;
    --search-only) SEARCH_ONLY=1 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

MIN_VRAM="${MIN_VRAM:-40}"
DISK="${DISK:-120}"
MAX_PRICE="${MAX_PRICE:-1.20}"
MIN_REL="${MIN_REL:-0.98}"
NUM_GPUS="${NUM_GPUS:-1}"
ORDER="${ORDER:-dph_total+}"
IMAGE="${IMAGE:-vastai/pytorch:@vastai-automatic-tag}"
PORTAL_ENV='-p 1111:1111 -p 8080:8080 -e OPEN_BUTTON_PORT="1111" -e OPEN_BUTTON_TOKEN="1" -e JUPYTER_DIR="/" -e DATA_DIRECTORY="/workspace/"'

command -v vastai >/dev/null || { echo "vastai not found. Run: pip install vastai" >&2; exit 1; }

if [ "$SEARCH_ONLY" = "0" ]; then
  : "${RUN:?set RUN=<run name>}"
  : "${MODEL:?set MODEL=<hf model id>}"
  : "${DOMAIN:?set DOMAIN=<domain>}"
fi
RUN="${RUN:-search-only}"

# --- Idempotency -------------------------------------------------------------
# Running this twice must never leave two boxes billing for one run.
if [ "$SEARCH_ONLY" = "0" ] && [ -s "$(run_id_file "$RUN")" ]; then
  EXISTING="$(tr -d '[:space:]' < "$(run_id_file "$RUN")")"
  if STATUS="$(instance_field "$EXISTING" actual_status)"; then
    echo "[launch] run '$RUN' already has instance $EXISTING (status: ${STATUS:-unknown}). Nothing to do."
    exit 0
  fi
  if [ "${RELAUNCH:-0}" != "1" ]; then
    echo "[launch] run '$RUN' recorded instance $EXISTING, which is gone from the account." >&2
    echo "[launch] Refusing to spend without RELAUNCH=1 (its disk, and anything on it, is already lost)." >&2
    exit 1
  fi
  echo "[launch] recorded instance $EXISTING is gone; RELAUNCH=1, renting a replacement."
fi

# --- Offer search (read-only) ------------------------------------------------
# gpu_ram in the QUERY DSL is GB; the --raw OUTPUT reports MB. Multiplying by
# 1024 here asks for "40960 GB" and matches nothing.
# MIN_CC defaults to 800 (Ampere). Llama-3.1 and Gemma-2 are bf16-trained, and
# Gemma-2 overflows in fp16, so a Turing card (Q RTX 8000, cc=750) is unusable
# however cheap its 48 GB looks. Without this clause the cheapest-first ordering
# picks exactly that card.
MIN_CC="${MIN_CC:-800}"
# Upload bandwidth is the real bottleneck: the run streams several GB to the HF
# dataset, which is our only durable copy. Cheapest-first once picked a box at
# 18.7 Mbps up, where a 3 GB sync costs ~20 minutes and any crash before it
# finishes loses everything. A few cents an hour buys two orders of magnitude.
MIN_UP="${MIN_UP:-500}"
QUERY="num_gpus=${NUM_GPUS} gpu_ram>=${MIN_VRAM} compute_cap>=${MIN_CC} inet_up>=${MIN_UP} verified=true rentable=true direct_port_count>=1 disk_space>=${DISK} dph_total<=${MAX_PRICE}"
echo "[launch] search: $QUERY"
echo "[launch] then reliability>=${MIN_REL}, applied here rather than in the query"
OFFERS_JSON="$(vastai search offers "$QUERY" -o "$ORDER" --raw)"

# Reliability is filtered HERE. Vast rejects a `reliability2` clause with
# "Unrecognized field" and then returns the UNFILTERED list, so putting it in
# the query looks like a guard while enforcing nothing. A host that dies
# mid-extraction costs far more than the cents saved, because extraction
# restarts from whatever survived on the box's disk.
OFFER_ID="$(printf '%s' "$OFFERS_JSON" | MIN_REL="$MIN_REL" python3 -c '
import sys, json, os
mr = float(os.environ["MIN_REL"] or 0)
offers = json.load(sys.stdin) or []
kept = [o for o in offers if float(o.get("reliability2") or 0) >= mr]
print("[launch] %d of %d offers meet reliability>=%.3f" % (len(kept), len(offers), mr), file=sys.stderr)
for o in offers[:5]:
    print("   %-10s $%.3f/hr  rel=%.3f  %sx %-16s %5.0f GB  up=%sMbps  %s" % (
        o.get("id"), o.get("dph_total") or 0.0, float(o.get("reliability2") or 0),
        o.get("num_gpus"), o.get("gpu_name"), (o.get("gpu_ram") or 0) / 1024.0,
        o.get("inet_up"), "KEPT" if o in kept else "rejected: reliability"), file=sys.stderr)
print(kept[0]["id"] if kept else "")
')"
[ -n "$OFFER_ID" ] || { echo "[launch] no offer met MIN_VRAM/DISK/MAX_PRICE/MIN_REL; loosen one." >&2; exit 1; }

printf '%s' "$OFFERS_JSON" | OFFER_ID="$OFFER_ID" python3 -c '
import sys, json, os
oid = int(os.environ["OFFER_ID"])
o = next(x for x in json.load(sys.stdin) if x.get("id") == oid)
print("[launch] CHOSEN offer %s | %sx %s | %.0f GB VRAM | $%.3f/hr | reliability=%.3f | up=%sMbps down=%sMbps | %s" % (
    o.get("id"), o.get("num_gpus"), o.get("gpu_name"), (o.get("gpu_ram") or 0) / 1024.0,
    o.get("dph_total") or 0.0, float(o.get("reliability2") or 0),
    o.get("inet_up"), o.get("inet_down"), o.get("geolocation")))
'

if [ "$SEARCH_ONLY" = "1" ]; then
  echo "[launch] --search-only: nothing was created, nothing was spent."
  exit 0
fi

LABEL="${LABEL_PREFIX}-${RUN}"
echo "[launch] would create: offer=$OFFER_ID disk=${DISK}GB label=$LABEL model=$MODEL domain=$DOMAIN"
if [ "${YES:-0}" != "1" ]; then
  read -r -p "[launch] create this instance? [y/N] " ans
  case "$ans" in y|Y) ;; *) echo "Aborted; nothing spent."; exit 0 ;; esac
fi

# --cancel-unavail: fail rather than leave a stopped instance whose disk bills.
CREATE_JSON="$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" \
  --env "$PORTAL_ENV" \
  --onstart-cmd 'entrypoint.sh' \
  --disk "$DISK" \
  --label "$LABEL" \
  --ssh --direct --cancel-unavail \
  --raw)"
INSTANCE_ID="$(printf '%s' "$CREATE_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("new_contract",""))')"
[ -n "$INSTANCE_ID" ] || { echo "[launch] create failed: $CREATE_JSON" >&2; exit 1; }

# Record BEFORE polling. A crash between here and "running" must never leave a
# billing box that no ledger knows about, because reap.sh cannot destroy what
# is not in .instances_ours.
printf '%s\n' "$INSTANCE_ID" > "$(run_id_file "$RUN")"
ledger_add_ours "$INSTANCE_ID" "$RUN"
cat > "$(run_env_file "$RUN")" <<EOF
RUN=$RUN
MODEL=$MODEL
DOMAIN=$DOMAIN
INSTANCE=$INSTANCE_ID
LABEL=$LABEL
CREATED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
echo "[launch] created instance $INSTANCE_ID, recorded as ours. Waiting for it to run."

for i in $(seq 1 80); do
  STATUS="$(instance_field "$INSTANCE_ID" actual_status || echo missing)"
  echo "   [$i] status: ${STATUS:-unknown}"
  [ "$STATUS" = "running" ] && break
  sleep 15
done

echo "[launch] next: RUN=$RUN bash cloud/sync_up.sh && RUN=$RUN bash cloud/push_secrets.sh"
bash "$CLOUD_DIR/fleet_status.sh"
