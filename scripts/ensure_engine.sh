#!/usr/bin/env bash
# Reuse the running engine unless something it loads has changed.
#
# Starting vLLM means a full torch.compile + graph capture pass (~4-5 min on this box), so a
# restart is only justified when the code or config the workers already imported has changed.
#   usage: scripts/ensure_engine.sh [port]      -> prints REUSE or RESTART
set -u
PORT=${1:-8001}
ADAPTER=/home/srj/vllm-kvmem-adapter
LAUNCH=/home/srj/run-kvmem-sparse.sh
CFG=/tmp/kvmem_plugin.json
LOG=/tmp/sparse_server.log

health=$(curl -s -m 3 "http://127.0.0.1:${PORT}/health" -o /dev/null -w '%{http_code}' || echo 000)
if [[ "$health" != "200" ]]; then
  echo "RESTART (no healthy engine on :${PORT})"; exit 0
fi

# The engine's start time comes from the pid file the launcher writes.
pidfile=$(ls -t /home/srj/vLLM-2080Ti-Definitive/run-logs/*.pid 2>/dev/null | head -1)
started=0
if [[ -n "$pidfile" ]]; then
  started=$(stat -c %Y "$pidfile" 2>/dev/null || echo 0)
fi
if (( started == 0 )); then
  echo "REUSE (engine healthy; start time unknown)"; exit 0
fi

newer=$(find "$ADAPTER/src" "$ADAPTER/plugin" -name '*.py' -newermt "@${started}" 2>/dev/null | head -3)
cfg_newer=""
[[ -f "$CFG" && $(stat -c %Y "$CFG") -gt $started ]] && cfg_newer="$CFG"

if [[ -n "$newer" || -n "$cfg_newer" ]]; then
  echo "RESTART (changed since start):"
  [[ -n "$newer" ]] && printf '  %s\n' $newer
  [[ -n "$cfg_newer" ]] && echo "  $cfg_newer"
  exit 0
fi
echo "REUSE (engine healthy, no code/config change since start)"
