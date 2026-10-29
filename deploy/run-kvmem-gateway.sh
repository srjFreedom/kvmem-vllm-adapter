#!/usr/bin/env bash
# ============================================================================
# KVMem 入口网关：选 B、把 max_tokens 定到 R、H/D 分层缓存历史块
#   日志固定写到 ~/kvmem-gateway.log
# ============================================================================
set -uo pipefail
LOG=${LOG:-$HOME/kvmem-gateway.log}
HOST=${HOST:-0.0.0.0}          # 与引擎一致：局域网可访问；只本机用 127.0.0.1
UPSTREAM=${UPSTREAM:-http://127.0.0.1:8001}
PORT=${PORT:-8100}
[ -f "$HOME/kvmem-params.env" ] && . "$HOME/kvmem-params.env"
CTX=${CTX:-524288}
B=${B:-65536}
R=${R:-32768}
# sink/recent 必须与引擎一致：同一个 kvmem 参数不能两处取不同值
SINK=${SINK:-8192}
RECENT=${RECENT:-2048}
SELECT=${SELECT:-retrieval}
HIST_KIB=${HIST_KIB:-64}
HIST_DISK_MIB=${HIST_DISK_MIB:-256}

if [ "${FOREGROUND:-0}" != "1" ]; then
  for p in $(ps -eo pid=,args= | awk '/kvmem_gateway/ && !/awk/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
  sleep 1
fi
{ echo "==================== $(date '+%F %T') KVMem gateway start ===================="
  echo "listen=$HOST:$PORT  CTX=$CTX B=$B R=$R sink=$SINK recent=$RECENT select=$SELECT"
        echo "history cache: H=${HIST_KIB}KiB D=${HIST_DISK_MIB}MiB"
} >> "$LOG"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GW=("${VLLM_PY:-${VLLM_ROOT:-$HERE/../engines/vllm}/.venv/bin/python}" \
  "$HERE/../scripts/kvmem_gateway.py" \
    --host "$HOST" --upstream "$UPSTREAM" --port "$PORT" \
    --kvmem-ctx "$CTX" --kvmem-budget "$B" --kvmem-gen-reserve "$R" \
    --kvmem-sink-tokens "$SINK" --kvmem-recent-tokens "$RECENT" \
    --kvmem-select "$SELECT" \
    --kvmem-history-kib "$HIST_KIB" --kvmem-history-disk-mib "$HIST_DISK_MIB" \
    --kvmem-history-disk-path /tmp/kvmem_history_tier)

if [ "${FOREGROUND:-0}" = "1" ]; then
  echo "foreground mode (systemd); log: $LOG"
  exec "${GW[@]}" >> "$LOG" 2>&1
fi

setsid "${GW[@]}" >> "$LOG" 2>&1 < /dev/null &
sleep 3
echo "gateway: $(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")  (tail -f $LOG)"
