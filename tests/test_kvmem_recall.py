"""Numeric check of kvmem's mean-K scorer against an independent reference.

This is the gate for P1: the adapter is about to rank pages with these numbers instead of its old
Python index, so the semantics have to be pinned down exactly, not approximately. The reference
below mirrors `kvmem/src/host/mean_key_retrieval.cpp` line by line:

    group       = query_heads / kv_heads
    Mean mode   : query sums divided by (tokens * group)
    Max mode    : the RAW token sum, undivided
    per block/layer: for each kv head, max over that head's probes of
                     dot(q, mean_k) / (|q| * |mean_k| + 1e-6)
    layer_score = sum(over kv heads) / kv_heads
    score       = sum(over ACTIVE layers) / active_layers

Run:  PYTHONPATH=build/python:src python tests/test_kvmem_recall.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "build", "python"))

from kvmem_recall import KvmemRecall  # noqa: E402
from kvmem_statistics import KvmemStatistics  # noqa: E402

BLOCK = 64


def reference_scores(stats, keys, q_rows, *, kv_heads, head_dim, query_heads, blocks,
                     aggregation):
    """Independent implementation of mean_key_retrieval_scores."""
    group = query_heads // kv_heads
    tokens = q_rows.shape[0]
    sums = q_rows.reshape(tokens, query_heads, head_dim).sum(axis=0).astype(np.float64)
    if aggregation == "mean":
        # Mean: the group's sums are averaged, then divided by the token count -> one probe per kv
        # head, i.e. means has kv_heads rows.
        means = sums.reshape(kv_heads, group, head_dim).mean(axis=1) / tokens
    else:
        # Max: raw sums, undivided, one probe PER QUERY HEAD (kv_heads * group rows).
        means = sums.reshape(query_heads, head_dim)
    q_norms = np.linalg.norm(means, axis=-1)
    out = []
    for block in range(blocks):
        per_layer = []
        for layer in range(stats.layers):
            key = stats.mean_k(block, layer).reshape(kv_heads, head_dim).astype(np.float64)
            best_sum = 0.0
            for head in range(kv_heads):
                k_norm = np.linalg.norm(key[head])
                best = -np.inf
                probes = (range(group) if aggregation == "max" else [None])
                for g in probes:
                    probe = head * group + g if g is not None else head
                    dot = float(np.dot(means[probe], key[head]))
                    best = max(best, dot / (q_norms[probe] * k_norm + 1e-6))
                best_sum += best
            per_layer.append(best_sum / kv_heads)
        out.append(float(np.mean(per_layer)))
    return np.asarray(out)


def main():
    rng = np.random.default_rng(11)
    failures = []
    for case, (layers, kv_heads, group, head_dim, tokens, blocks) in enumerate([
            (3, 1, 1, 4, 1, 2),
            (4, 2, 3, 8, 5, 3),
            (2, 4, 2, 4, 17, 4),
    ]):
        query_heads = kv_heads * group
        stats = KvmemStatistics(layers, kv_heads=kv_heads, head_dim=head_dim)
        width = kv_heads * head_dim
        keys = rng.normal(size=(layers, blocks, width)).astype(np.float32)
        # Write each block as `n` tokens whose SUM is the intended vector: the store keeps sums and
        # divides by mean_tokens, so this yields mean_k == keys/64 while keeping the sums exact.
        for layer in range(layers):
            for block in range(blocks):
                stats.write_step(block * BLOCK, BLOCK, {layer: keys[layer, block][None, :]})
        q_rows = rng.normal(size=(tokens, query_heads, head_dim)).astype(np.float32)
        layer_rows = {layer: q_rows for layer in range(layers)}
        for aggregation in ("mean", "max"):
            recall = KvmemRecall(stats, block_tokens=BLOCK, aggregation=aggregation)
            got = recall.scores(layer_rows, rows=tokens, query_heads=query_heads,
                                kv_heads=kv_heads, head_dim=head_dim, blocks=blocks)
            want = reference_scores(stats, keys, q_rows, kv_heads=kv_heads, head_dim=head_dim,
                                    query_heads=query_heads, blocks=blocks,
                                    aggregation=aggregation)
            if got is None:
                failures.append(f"case{case}/{aggregation}: no scores ({recall.last_error})")
                continue
            err = float(np.max(np.abs(np.asarray(got) - want)))
            status = "ok" if err < 1e-5 else "MISMATCH"
            print(f"case{case} {aggregation:4s} layers={layers} kv={kv_heads} group={group} "
                  f"tokens={tokens} blocks={blocks} max_err={err:.2e} {status}")
            if err >= 1e-5:
                failures.append(f"case{case}/{aggregation}: err {err:.2e} got {got} want {want}")
    # An ABSENT layer must not change the score: skipping layer 2 in a 3-layer store gives the same
    # numbers as the same 3 active layers (that is what 'absent, never zero' has to mean).
    stats = KvmemStatistics(3, kv_heads=1, head_dim=4)
    keys = rng.normal(size=(3, 2, 4)).astype(np.float32)
    for layer in range(3):
        for block in range(2):
            stats.write_step(block * BLOCK, BLOCK, {layer: keys[layer, block][None, :]})
    q_rows = rng.normal(size=(1, 1, 4)).astype(np.float32)
    recall = KvmemRecall(stats, block_tokens=BLOCK, aggregation="max")
    all_layers = recall.scores({0: q_rows, 1: q_rows, 2: q_rows}, rows=1, query_heads=1,
                               kv_heads=1, head_dim=4, blocks=2)
    two_layers = recall.scores({0: q_rows, 1: q_rows}, rows=1, query_heads=1,
                               kv_heads=1, head_dim=4, blocks=2)
    # The honest check: an ABSENT layer must score like a store that never had it, i.e. the mean is
    # taken over the REMAINING active layers (not over 3 with a zero contribution).
    small = KvmemStatistics(2, kv_heads=1, head_dim=4)
    for layer in range(2):
        for block in range(2):
            small.write_step(block * BLOCK, BLOCK, {layer: keys[layer, block][None, :]})
    small_scores = KvmemRecall(small, block_tokens=BLOCK, aggregation="max").scores(
        {0: q_rows, 1: q_rows}, rows=1, query_heads=1, kv_heads=1, head_dim=4, blocks=2)
    print(f"absent-layer check: 3 active={np.round(all_layers, 5)} 2 active="
          f"{np.round(two_layers, 5)} 2-layer store={np.round(small_scores, 5)}")
    if not np.allclose(two_layers, small_scores, atol=1e-5):
        failures.append("an ABSENT layer did not score like a store that never had it")
    if failures:
        print("\nFAIL")
        for line in failures:
            print("  -", line)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
