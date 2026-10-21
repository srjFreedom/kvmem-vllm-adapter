#!/usr/bin/env bash
# Reproducible check for the ALLOCATION-SIDE half of the loop: the scheduler hands the middle
# aligned blocks of a live request back to vLLM's pool, and the answers stay correct because the
# worker only ever reads the compacted view.
#
# Usage: scripts/free_loop_check.sh [port]
#   Requires an engine started with free_path=true and max_num_seqs>=2 (the pool must actually be
#   under pressure, otherwise nothing is freed).
set -u
PORT=${1:-8001}
LOG=${KVMEM_PROBE_LOG:-/tmp/kvmem_sparse.jsonl}
CFG=/tmp/kvmem_plugin.json

health=$(curl -s -m 3 "http://127.0.0.1:${PORT}/health" -o /dev/null -w '%{http_code}' || echo 000)
[[ "$health" == "200" ]] || { echo "no engine on :${PORT}"; exit 1; }
python3 - "$CFG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
missing = [k for k in ("free_path",) if not cfg.get(k)]
if missing:
    print(f"WARNING: {missing} are off in {sys.argv[1]}; nothing will be freed")
print(f"free_path={cfg.get('free_path')} low_watermark={cfg.get('free_low_watermark')} "
      f"per_step={cfg.get('free_per_step')} sink_blocks={cfg.get('sink_blocks')} "
      f"recent_blocks={cfg.get('recent_blocks')}")
PY

before=$(grep -c '"kind": "freed_blocks"' "$LOG" 2>/dev/null | head -1)
before=${before:-0}
echo "freed_blocks records before: $before"
for tag in 0 1 2 3; do
  rm -f "/tmp/fl_resp${tag}.json"
  nohup bash -c "curl -s -m 600 'http://127.0.0.1:${PORT}/v1/chat/completions' \
    -H 'Content-Type: application/json' -d @/tmp/w4_req${tag}.json \
    -o /tmp/fl_resp${tag}.json -w 'http %{http_code}\n' > /tmp/fl_curl${tag}.log 2>&1" \
    > /dev/null 2>&1 &
done
echo "4 concurrent requests fired; waiting for completion..."
for _ in $(seq 1 60); do
  done_count=$(ls /tmp/fl_resp*.json 2>/dev/null | wc -l | head -1)
  (( done_count >= 4 )) && break
  sleep 10
done

python3 - "$before" "$LOG" <<'PY'
import json, random, sys, glob
before = int(sys.argv[1]); log = sys.argv[2]
def key(i, tag):
    r = random.Random(i * 7 + tag)
    return "".join(r.choice("BCDFGHJKLMNPQRSTVWXYZ23456789") for _ in range(6))
ok = 0
for tag in range(4):
    try:
        got = (json.load(open(f"/tmp/fl_resp{tag}.json"))["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        print(f"  req{tag}: no response"); continue
    exp = f"{key(50, tag)} {key(1260, tag)}"
    good = got == exp
    ok += good
    print(f"  req{tag}: {'OK' if good else 'MISMATCH ' + repr(got)}")
free_records = [json.loads(l) for l in open(log) if '"kind": "freed_blocks"' in l]
total = sum(r.get("freed", 0) for r in free_records)
print(f"answers correct: {ok}/4")
print(f"freed_blocks records: {len(free_records)} (before: {before})  blocks released: {total}")
if free_records:
    last = free_records[-1]
    print(f"last: freed={last.get('freed')} free_now={last.get('free_now')} "
          f"held={last.get('held')} pool_total={last.get('pool_total')}")
print("PASS free_loop_check" if ok == 4 and total > 0 else "INCOMPLETE free_loop_check")
PY
