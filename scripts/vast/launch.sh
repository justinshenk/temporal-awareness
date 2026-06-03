#!/usr/bin/env bash
# Generic vast.ai launcher for GPU experiments.
#
# Provisions a single vast.ai instance, rsyncs the current repo to it, runs a
# user-supplied command on the remote, pulls a results tarball back, and
# destroys the instance it created.
#
# CRITICAL SAFETY INVARIANT — shared accounts:
#   This script ONLY destroys the instance it created itself. It never issues
#   `destroy all`, never iterates over existing instances, and never touches
#   any instance ID it didn't produce from its own `create instance` call.
#   If the create step fails to return an ID, the script aborts WITHOUT calling
#   destroy. The cleanup trap re-checks that INSTANCE_ID is set and non-empty.
#
# Usage:
#   scripts/vast/launch.sh <remote-command>
#
# Example:
#   REMOTE_CMD='python scripts/my_experiment/run.py --smoke' \
#       scripts/vast/launch.sh
#
#   GPU_QUERY='gpu_name=H100_SXM num_gpus=1 disk_space>=200' \
#   BUDGET_USD=50 \
#       scripts/vast/launch.sh "python scripts/my_experiment/run.py"
#
# Env vars:
#   REMOTE_CMD        command string to execute on the remote (required, or $1)
#   VAST_API_KEY      (required) — read from .env if not set
#   GPU_QUERY         default 'gpu_name=RTX_5090 num_gpus=1 disk_space>=120 reliability>=0.97 rentable=True'
#   IMAGE             default pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel (bootstrap.sh upgrades torch for Blackwell)
#   MAX_DPH_USD       default 3.00 — refuse offers above this total price/hour
#   BUDGET_USD        default 50 — informational, surfaced at launch time
#   MIN_DISK_GB       default 120 — minimum instance disk (and --disk requested)
#   EXCLUDES          default excludes for rsync (.git, venvs, caches, results)
#   RESULTS_REMOTE    default /workspace/results.tar.gz
#   RESULTS_LOCAL     default $PWD/vast_results/results_<instance_id>.tar.gz
#   INSTALL_DEPS      default 1 — run bootstrap.sh to install deps before REMOTE_CMD
#   BOOTSTRAP_EXTRA   extra pip packages to install (space-separated)
#   PREDOWNLOAD_HF    space-separated HF model IDs to pre-download before REMOTE_CMD
#   LABEL             default "vast-launch" — prefix for result files
#   LEAVE_INSTANCE    default 0 — if 1, skip destroy at end (useful for debugging)
#   VAST_USER         default $USER — tag included in the instance label so
#                     other users on a shared account can attribute running
#                     instances. Falls back to "unknown" if neither is set.
#
# Exits:
#   0 on success (instance destroyed)
#   non-zero on any failure (cleanup trap still runs; instance still destroyed
#   if it was created, unless LEAVE_INSTANCE=1)

set -euo pipefail

REMOTE_CMD="${REMOTE_CMD:-${1:-}}"
if [ -z "$REMOTE_CMD" ]; then
    echo "ERROR: REMOTE_CMD (or first positional arg) required" >&2
    echo "  example: $0 'python scripts/my_experiment/run.py --smoke'" >&2
    exit 2
fi

REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_DIR"

# ---- api key ----
if [ -z "${VAST_API_KEY:-}" ]; then
    ENV_FILE="$REPO_DIR/.env"
    if [ ! -f "$ENV_FILE" ] && [ -f "$HOME/projects/temporal/temporal-awareness/.env" ]; then
        ENV_FILE="$HOME/projects/temporal/temporal-awareness/.env"
    fi
    if [ -f "$ENV_FILE" ]; then
        VAST_API_KEY="$(grep -E '^VASTAI_API_KEY=' "$ENV_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'" | head -1)"
    fi
fi
if [ -z "${VAST_API_KEY:-}" ]; then
    echo "ERROR: VAST_API_KEY not set and not found in .env" >&2
    exit 2
fi
export VAST_API_KEY

# ---- defaults ----
GPU_QUERY="${GPU_QUERY:-gpu_name=RTX_5090 num_gpus=1 disk_space>=120 reliability>=0.97 rentable=True}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel}"
MAX_DPH_USD="${MAX_DPH_USD:-3.00}"
BUDGET_USD="${BUDGET_USD:-50}"
MIN_DISK_GB="${MIN_DISK_GB:-120}"
EXCLUDES="${EXCLUDES:---exclude=.git --exclude=.venv* --exclude=venv --exclude=__pycache__ --exclude=*.tar.gz --exclude=results --exclude=spd_repo}"
RESULTS_REMOTE="${RESULTS_REMOTE:-/workspace/results.tar.gz}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
BOOTSTRAP_EXTRA="${BOOTSTRAP_EXTRA:-}"
PREDOWNLOAD_HF="${PREDOWNLOAD_HF:-}"
LABEL="${LABEL:-vast-launch}"
LEAVE_INSTANCE="${LEAVE_INSTANCE:-0}"
VAST_USER="${VAST_USER:-${USER:-unknown}}"

# ---- vastai CLI ----
VASTAI_VENV="${VASTAI_VENV:-$HOME/.cache/vastai-venv}"
if [ ! -x "$VASTAI_VENV/bin/vastai" ]; then
    python3 -m venv "$VASTAI_VENV"
    "$VASTAI_VENV/bin/pip" install --quiet --upgrade pip
    "$VASTAI_VENV/bin/pip" install --quiet vastai
fi
VASTAI="$VASTAI_VENV/bin/vastai"
"$VASTAI" set api-key "$VAST_API_KEY" >/dev/null

# ---- safety: track instance we create; cleanup only destroys that one ----
INSTANCE_ID=""
cleanup() {
    local rc=$?
    if [ -z "$INSTANCE_ID" ]; then
        echo "[launch] cleanup: no instance was created, nothing to destroy."
        return $rc
    fi
    if [ "$LEAVE_INSTANCE" = "1" ]; then
        echo "[launch] cleanup: LEAVE_INSTANCE=1, keeping instance $INSTANCE_ID"
        echo "[launch]   destroy manually: $VASTAI destroy instance $INSTANCE_ID"
        return $rc
    fi
    echo "[launch] cleanup: destroying instance $INSTANCE_ID (and only this one)"
    # Explicit by-ID destroy. NEVER use `destroy all` or any variant that
    # would touch other instances on this (possibly shared) account.
    "$VASTAI" destroy instance "$INSTANCE_ID" 2>&1 || \
        echo "[launch] WARN: destroy instance $INSTANCE_ID returned non-zero; verify manually"
    return $rc
}
trap cleanup EXIT INT TERM

# ---- search offers ----
echo "[launch] searching offers: $GPU_QUERY"
OFFERS_RAW="$("$VASTAI" search offers "$GPU_QUERY" -o dph_total --raw 2>&1 || true)"
CHOSEN="$(python3 - <<'PY' <<<"$OFFERS_RAW"
import json, os, sys
try:
    offers = json.loads(sys.stdin.read())
except Exception as e:
    print(f"PARSE_ERR: {e}", file=sys.stderr); sys.exit(2)
max_dph = float(os.environ.get("MAX_DPH_USD", "3.0"))
best = None
for o in offers:
    dph = float(o.get("dph_total", 0) or 0)
    if dph <= 0 or dph > max_dph:
        continue
    if not o.get("rentable", True):
        continue
    if best is None or dph < float(best.get("dph_total", 0)):
        best = o
if not best:
    print("NONE")
else:
    print(f"{best['id']} {best.get('dph_total',0):.4f} {best.get('gpu_name','?')} {best.get('disk_space',0)}")
PY
)"

if [ "$CHOSEN" = "NONE" ] || [ -z "$CHOSEN" ]; then
    echo "ERROR: no offers under \$$MAX_DPH_USD/hr matching: $GPU_QUERY" >&2
    exit 1
fi

OFFER_ID="$(echo "$CHOSEN" | awk '{print $1}')"
OFFER_DPH="$(echo "$CHOSEN" | awk '{print $2}')"
OFFER_GPU="$(echo "$CHOSEN" | awk '{print $3}')"
OFFER_DISK="$(echo "$CHOSEN" | awk '{print $4}')"
echo "[launch] picked offer $OFFER_ID: \$${OFFER_DPH}/hr, $OFFER_GPU, ${OFFER_DISK}GB disk"
python3 -c "
dph=float('$OFFER_DPH'); budget=float('$BUDGET_USD')
print(f'[launch] budget=\${budget:.2f} -> {budget/dph:.1f}h of runtime headroom')
"

# ---- create instance ----
VAST_INSTANCE_LABEL="${VAST_USER}/${LABEL}/$(date -u +%Y%m%dT%H%M%SZ)"
echo "[launch] creating instance (image=$IMAGE, disk=${MIN_DISK_GB}GB, label=$VAST_INSTANCE_LABEL)"
CREATE_OUT="$("$VASTAI" create instance "$OFFER_ID" \
    --image "$IMAGE" --disk "$MIN_DISK_GB" --ssh \
    --label "$VAST_INSTANCE_LABEL" 2>&1)"
echo "[launch] create output: $CREATE_OUT"
INSTANCE_ID="$(echo "$CREATE_OUT" | python3 -c "
import re, sys
text = sys.stdin.read()
m = re.search(r\"'new_contract': (\d+)\", text)
if m:
    print(m.group(1))
" || true)"

if [ -z "$INSTANCE_ID" ]; then
    echo "ERROR: could not parse new instance id from create output" >&2
    echo "  raw: $CREATE_OUT" >&2
    INSTANCE_ID=""  # ensure cleanup() sees empty and does NOT try to destroy
    exit 1
fi
echo "[launch] created instance $INSTANCE_ID"

# ---- wait for running status + SSH ----
echo "[launch] waiting for instance to reach running state..."
SSH_HOST=""; SSH_PORT=""
for i in $(seq 1 60); do
    STATUS_LINE="$("$VASTAI" show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    print(d.get('actual_status','unknown'), d.get('ssh_host','') or '', d.get('ssh_port','') or '')
except Exception:
    print('unknown  ')
" || echo 'unknown  ')"
    ACTUAL="$(echo "$STATUS_LINE" | awk '{print $1}')"
    SSH_HOST="$(echo "$STATUS_LINE" | awk '{print $2}')"
    SSH_PORT="$(echo "$STATUS_LINE" | awk '{print $3}')"
    echo "  [$i/60] actual_status=$ACTUAL ssh=$SSH_HOST:$SSH_PORT"
    if [ "$ACTUAL" = "running" ] && [ -n "$SSH_HOST" ] && [ "$SSH_HOST" != "None" ]; then
        break
    fi
    sleep 10
done

if [ "$ACTUAL" != "running" ] || [ -z "$SSH_HOST" ]; then
    echo "ERROR: instance $INSTANCE_ID did not become ready" >&2
    exit 1
fi

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -p $SSH_PORT"
SSH="ssh $SSH_OPTS root@$SSH_HOST"

# ---- wait for sshd actually accepting connections ----
for i in $(seq 1 12); do
    if $SSH -o ConnectTimeout=5 'echo ok' >/dev/null 2>&1; then
        break
    fi
    sleep 5
done

echo "[launch] rsyncing repo to instance"
$SSH "mkdir -p /workspace/repo"
rsync -az --delete -e "ssh $SSH_OPTS" $EXCLUDES "$REPO_DIR/" "root@$SSH_HOST:/workspace/repo/"

# ---- bootstrap (install deps) then run the user command ----
REMOTE_SCRIPT=/tmp/vast_remote_run.$$.sh
cat > "$REMOTE_SCRIPT" <<REMOTE_EOF
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/repo
export PYTHONPATH="/workspace/repo:\${PYTHONPATH:-}"
export HF_HOME="\${HF_HOME:-/workspace/hf_cache}"

if [ "${INSTALL_DEPS}" = "1" ] && [ -f scripts/vast/bootstrap.sh ]; then
    BOOTSTRAP_EXTRA='${BOOTSTRAP_EXTRA}' PREDOWNLOAD_HF='${PREDOWNLOAD_HF}' bash scripts/vast/bootstrap.sh
fi

echo "[remote] === running user command ==="
${REMOTE_CMD}
rc=\$?
echo "[remote] user command exited with rc=\$rc"

echo "[remote] === packaging results -> ${RESULTS_REMOTE} ==="
tar czf "${RESULTS_REMOTE}" \
    results 2>/dev/null || echo "[remote] (no results/ dir)"
ls -la "${RESULTS_REMOTE}" 2>&1 || true
exit \$rc
REMOTE_EOF
chmod +x "$REMOTE_SCRIPT"
scp -q $SSH_OPTS "$REMOTE_SCRIPT" "root@$SSH_HOST:/workspace/remote_run.sh"
rm -f "$REMOTE_SCRIPT"

echo "[launch] executing remote command: $REMOTE_CMD"
set +e
$SSH "bash /workspace/remote_run.sh" 2>&1
REMOTE_RC=$?
set -e
echo "[launch] remote command rc=$REMOTE_RC"

# ---- pull results ----
LOCAL_RESULTS_DIR="${RESULTS_LOCAL_DIR:-$REPO_DIR/vast_results}"
mkdir -p "$LOCAL_RESULTS_DIR"
LOCAL_TAR="$LOCAL_RESULTS_DIR/${LABEL}_${INSTANCE_ID}.tar.gz"
echo "[launch] pulling $RESULTS_REMOTE -> $LOCAL_TAR"
scp -q $SSH_OPTS "root@$SSH_HOST:$RESULTS_REMOTE" "$LOCAL_TAR" || \
    echo "[launch] WARN: could not pull results tarball from remote"
ls -la "$LOCAL_TAR" 2>&1 || true

echo "[launch] done (rc=$REMOTE_RC). Cleanup trap will destroy instance $INSTANCE_ID."
exit $REMOTE_RC
