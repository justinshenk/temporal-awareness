#!/bin/bash
# Memory-pressure gate. Exits 0 when it is safe to load a multi-GB model.
#
# A machine crash on 2026-08-03 (swap 58.4/59.4 GB, an ollama process holding
# 20.8 GB RSS beside a ~16 GB Llama) destroyed a running sweep. Never launch a
# big model into that pressure. This gate never kills anyone else's process; it
# only refuses to add load.
#
#   bash scripts/scratch/mem_gate.sh          # check, print, exit 0/1
#
# Thresholds: foreign process >= 4 GB RSS, or swap used >= 50% of swap total,
# or free memory < 8 GB.

set -u

FOREIGN_GB_MAX=4
SWAP_PCT_MAX=50
FREE_GB_MIN=8

fail=0

echo "=== memory gate $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

ollama_procs=$(pgrep -fl ollama || true)
if [ -n "$ollama_procs" ]; then
  echo "BLOCK: ollama is running:"
  echo "$ollama_procs"
  fail=1
else
  echo "ok: no ollama process"
fi

# Largest non-Claude, non-kernel user process, in GB.
read -r foreign_kb foreign_cmd < <(
  ps -Ao rss,comm -r | awk 'NR>1 && $2 !~ /claude|mds|kernel_task/ {print $1, $2; exit}'
)
foreign_gb=$(echo "$foreign_kb" | awk '{printf "%.2f", $1/1048576}')
if awk "BEGIN{exit !($foreign_gb >= $FOREIGN_GB_MAX)}"; then
  echo "BLOCK: foreign process ${foreign_gb} GB >= ${FOREIGN_GB_MAX} GB: $foreign_cmd"
  fail=1
else
  echo "ok: largest foreign process ${foreign_gb} GB ($foreign_cmd)"
fi

swap_line=$(sysctl -n vm.swapusage)
swap_total=$(echo "$swap_line" | awk '{gsub(/M/,"",$3); print $3}')
swap_used=$(echo "$swap_line" | awk '{gsub(/M/,"",$6); print $6}')
if awk "BEGIN{exit !($swap_total > 0)}"; then
  swap_pct=$(awk "BEGIN{printf \"%.1f\", 100*$swap_used/$swap_total}")
else
  swap_pct=0
fi
if awk "BEGIN{exit !($swap_pct >= $SWAP_PCT_MAX)}"; then
  echo "BLOCK: swap ${swap_used}M/${swap_total}M = ${swap_pct}% >= ${SWAP_PCT_MAX}%"
  fail=1
else
  echo "ok: swap ${swap_used}M/${swap_total}M = ${swap_pct}%"
fi

free_gb=$(vm_stat | awk '/Pages free/ {printf "%.2f", $3*16384/1073741824}')
if awk "BEGIN{exit !($free_gb < $FREE_GB_MIN)}"; then
  echo "BLOCK: only ${free_gb} GB free (< ${FREE_GB_MIN} GB)"
  fail=1
else
  echo "ok: ${free_gb} GB free"
fi

if [ "$fail" -ne 0 ]; then
  echo "GATE: BLOCKED — do not launch; wait and report."
  exit 1
fi
echo "GATE: CLEAR — safe to load one model."
exit 0
