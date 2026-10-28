"""Host-side check of the mean-K statistics store (no engine, no GPU).

Verifies the three properties the recall path depends on, against kvmem's own `RawKvStore`
semantics rather than against our own bookkeeping:

  1. a bucket's `mean_k` is the average of the tokens written into it;
  2. a bucket split across two steps accumulates (never resets) when the second write is not
     bucket-aligned — a resumed/adopted request depends on this;
  3. a step that starts on a bucket boundary resets that bucket (a replay over the same position
     must not double-count).

Run:  PYTHONPATH=build/python:src python tests/test_kvmem_statistics.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "python"))

from kvmem_statistics import KvmemStatistics  # noqa: E402

BLOCK = 64
LAYERS = 2
WIDTH = 4


def _rows(values, first_pos, tokens):
    """Bucket sums of `values[i]` (a float per token) as (span, WIDTH) rows, like the engine's op."""
    base = first_pos // BLOCK
    span = tokens // BLOCK + 2
    out = np.zeros((span, WIDTH), dtype=np.float32)
    for i in range(tokens):
        pos = first_pos + i
        local = pos // BLOCK - base
        out[local] += values[i]
    return out


def main():
    stats = KvmemStatistics(LAYERS, kv_heads=1, head_dim=WIDTH, block_tokens=BLOCK)
    failures = []

    def check(name, got, want):
        if not np.allclose(np.asarray(got, dtype=np.float64),
                           np.asarray(want, dtype=np.float64), atol=1e-5):
            failures.append(f"{name}: got {got} want {want}")

    # 1. values 0..99 across two buckets
    first = list(range(100))
    sums = {layer: _rows(first, 0, 100) for layer in range(LAYERS)}
    written = stats.write_step(0, 100, sums)
    check("buckets written (100 tokens, 2 buckets, 2 layers)", written, 4)
    check("bucket0 mean", stats.mean_k(0, 0), np.full(WIDTH, np.mean(first[0:64])))
    check("bucket1 mean (partial)", stats.mean_k(1, 0), np.full(WIDTH, np.mean(first[64:100])))

    # 2. fill bucket1 up to 64 tokens; the write is NOT bucket-aligned (pos 100)
    second = list(range(100, 128))
    stats.write_step(100, 28, {layer: _rows(second, 100, 28) for layer in range(LAYERS)})
    check("bucket1 mean after accumulate", stats.mean_k(1, 0),
          np.full(WIDTH, np.mean(first[64:100] + second)))

    # 3. re-write bucket2 from its boundary: a replay must reset, not double
    third = list(range(128, 160))
    stats.write_step(128, 32, {layer: _rows(third, 128, 32) for layer in range(LAYERS)})
    check("bucket2 mean", stats.mean_k(2, 0), np.full(WIDTH, np.mean(third)))
    stats.write_step(128, 32, {layer: _rows(third, 128, 32) for layer in range(LAYERS)})
    check("bucket2 mean after replay", stats.mean_k(2, 0), np.full(WIDTH, np.mean(third)))

    # 4. truncation keeps the prefix' statistics and drops the rest
    stats.truncate_to(128)
    check("bucket2 gone after truncate", stats.mean_k(2, 0), np.zeros(WIDTH))
    check("bucket1 intact after truncate", stats.mean_k(1, 0),
          np.full(WIDTH, np.mean(first[64:100] + second)))

    info = stats.stats()
    print("stats:", info)
    if failures:
        print("\nFAIL")
        for line in failures:
            print("  -", line)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
