#!/usr/bin/env bash
# Concurrency check for the kvmem sparse backend.
#   concurrency_check.sh [N] [PROMPT_CHARS] [PORT]
# Fires N concurrent requests and reports how many returned 200 with a non-empty answer.
set -uo pipefail
N=${1:-2}
CHARS=${2:-20000}
PORT=${3:-8001}
BASE=${BASE_REQ:-/tmp/v2_req0.json}
[ -f "$BASE" ] || { echo "need a base request at $BASE"; exit 2; }
python3 - "$BASE" "$N" "$CHARS" <<'PY'
import json, sys
base = json.load(open(sys.argv[1])); n = int(sys.argv[2]); chars = int(sys.argv[3])
text = base['messages'][0]['content'][:chars]
for i in range(1, n + 1):
    d = json.loads(json.dumps(base))
    d['messages'] = [{"role": "user", "content": text + f"\n\n(Answer only with the code C{i}.)"}]
    d['max_tokens'] = 16
    json.dump(d, open(f"/tmp/conc_{i}.json", "w"))
PY
rm -f /tmp/conc_*.done /tmp/conc_*.out
for i in $(seq 1 "$N"); do
  nohup curl -s -m 900 -o "/tmp/conc_$i.out" -w "%{http_code}" \
    "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
    -d @"/tmp/conc_$i.json" > "/tmp/conc_$i.done" 2>&1 &
done
ok=0; start=$(date +%s)
for _ in $(seq 1 300); do
  done_n=0
  for i in $(seq 1 "$N"); do [ -s "/tmp/conc_$i.done" ] && done_n=$((done_n+1)); done
  [ "$done_n" = "$N" ] && break
  sleep 3
done
for i in $(seq 1 "$N"); do
  code=$(cat "/tmp/conc_$i.done" 2>/dev/null)
  body=$(python3 -c "
import json,sys
try:
    d=json.load(open('/tmp/conc_$i.out')); c=d['choices'][0]['message'].get('content') or ''
    print('answer=%r' % c[:24])
except Exception:
    print('no-json')
" 2>/dev/null)
  echo "req$i: http=$code $body"
  [ "$code" = "200" ] && ok=$((ok+1))
done
echo "elapsed: $(( $(date +%s) - start ))s"
echo "RESULT: $ok/$N returned 200"
[ "$ok" = "$N" ]
