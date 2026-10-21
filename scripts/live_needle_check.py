#!/usr/bin/env python3
"""Assert the live-engine smoke result: on every TP rank a real block was evicted to the host
pool and restored, and the live KV stayed byte-identical (the needle is alive)."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kvmem_smoke.jsonl"
rows = [json.loads(line) for line in open(path)]
errors = [r for r in rows if r["kind"].endswith("error")]
needles = [r for r in rows if r["kind"] == "needle"]
geom = [r for r in rows if r["kind"] == "geometry"]

if not geom:
    print("FAIL: the adapter never attached (no geometry record)")
    sys.exit(1)
if errors:
    print(f"FAIL: {len(errors)} error record(s); first: {errors[0].get('error', '')[:400]}")
    sys.exit(1)
if not needles:
    print("FAIL: no needle record (the prompt may be too short to span two blocks)")
    sys.exit(1)

ok = True
for r in needles:
    good = r["bytes_identical"] and r["evicted"]
    ok &= bool(good)
    print(f"rank pid={r['pid']}: evicted={r['evicted']} restored_blocks={r['restored_blocks']} "
          f"bytes_identical={r['bytes_identical']} -> {'OK' if good else 'FAIL'}")

print(f"geometry: {geom[0]['paged_tensors']} paged layers, block={geom[0]['block_tokens']} tokens, "
      f"heads={geom[0]['num_kv_heads']}, head_dim={geom[0]['head_dim']}, "
      f"record_bytes={geom[0]['record_bytes']}")
print("PASS live_needle_check" if ok else "FAIL live_needle_check")
sys.exit(0 if ok else 1)
