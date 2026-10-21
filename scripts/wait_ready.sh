#!/usr/bin/env bash
# Fast readiness check: never sleep in fixed chunks, never wait for a timeout when the answer is
# already in the log.
#
#   scripts/wait_ready.sh [port] [timeout_s] [--smoke] [--log PATH]
#
# Success is whichever comes first:
#   * the log says "Application startup complete"
#   * GET /health returns 200
# Failure is immediate (exit 2) when the log shows a known startup error, so a broken start costs
# seconds instead of minutes. Exit 1 on timeout.
set -uo pipefail

PORT=${1:-${PORT:-18200}}
TIMEOUT=${2:-900}
[ "${2:-}" = "--smoke" ] && { TIMEOUT=900; SMOKE_ONLY=1; }
shift 2 2>/dev/null || true
SMOKE=${SMOKE_ONLY:-0}
LOG=""
while (($#)); do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --log) LOG=${2:-}; shift ;;
    *) ;;
  esac
  shift
done

: "${SMOKE:=0}"
LOG_DIR=${LOG_DIR:-/home/srj/vLLM-2080Ti-Definitive/run-logs}
[ -n "$LOG" ] || LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)

FAIL_PATTERN='initialization failed|Address already in use|Engine core initialization failed|To serve at least one request|Traceback (most recent call last)|IndentationError|NameError'

START_MS=$(date +%s%3N)
deadline=$(( START_MS + TIMEOUT * 1000 ))
health_ok=0
while :; do
  now=$(date +%s%3N)
  (( now > deadline )) && { echo "TIMEOUT after ${TIMEOUT}s (log: ${LOG:-none})" >&2; exit 1; }

  if curl -s -m 1 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '^200$'; then
    health_ok=1
    break
  fi
  if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    if grep -aq "Application startup complete" "$LOG" 2>/dev/null; then
      health_ok=1
      break
    fi
    if match=$(grep -aoE "$FAIL_PATTERN" "$LOG" 2>/dev/null | head -1); [ -n "$match" ]; then
      echo "ENGINE FAILED: ${match} (log: $LOG)" >&2
      grep -aE "$FAIL_PATTERN" "$LOG" 2>/dev/null | head -3 >&2
      exit 2
    fi
  fi
  sleep 0.5
done

elapsed=$(( $(date +%s%3N) - START_MS ))
echo "ready on :${PORT} after $(( elapsed / 1000 )).$(( elapsed % 1000 / 100 ))s${LOG:+  (log: $LOG)}"

if [ "$SMOKE" = "1" ]; then
  MODEL=$(curl -s -m 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
          | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
  if [ -z "$MODEL" ]; then echo "smoke: SKIPPED (no model id from /v1/models)" >&2; exit 0; fi
  t0=$(date +%s%N)
  code=$(curl -s -m 120 -o /tmp/kvmem_smoke_resp.json -w '%{http_code}' \
    "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}")
  t1=$(date +%s%N)
  echo "smoke: ${code} in $(( (t1 - t0) / 1000000 ))ms (model=${MODEL})"
  [ "$code" = "200" ] || exit 3
fi
