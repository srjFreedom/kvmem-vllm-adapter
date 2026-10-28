"""Per-sequence mean-K statistics, owned by kvmem's `RawKvStore` (v0.19 recall design).

Why this module exists
----------------------
Before v0.19 this adapter kept its own mean-key store in Python (`kvmem_retrieval_index.py`):
per-block torch tensors, hand-rolled cosine, hand-rolled layer averaging. v0.19 makes that
statistics layer part of the core contract:

    RawKvStore(n_layer = full-attention layers,
               n_embd_k = kv_heads * head_dim,      # the KEY width
               block_tokens = 64)                    # the statistics bucket
    write_layer_mean_sum(pos0, n, layer, sum_f32)    # ordered FP32 token sums per bucket
    mean_k(block, layer)                             # normalised mean, what the scorers read

The reference product program fills exactly this store from **device-side bucket sums**
(`backends/ninfer/src/models/qwen3_5/program/storage/memory_statistics.cpp`,
`commit_memory_statistics`), and the scorers (`mean_key_retrieval_scores`, `page_softmax_scores`)
read nothing else. So the engine reduces K while it still owns it (see `kvmem_bridge.capture_k`)
and this class is only the host-side bookkeeping that feeds the core store.

Bucket semantics (must match the store, not be "close enough")
--------------------------------------------------------------
`write_layer_mean_sum` fills the bucket at `pos0 // block_tokens`, adds the supplied sum, and sets
`mean_tokens`:
  * `pos0 % block_tokens == 0` -> the bucket starts here, so its sum is RESET first;
  * otherwise the sum is ACCUMULATED onto the earlier tokens of that bucket.
A step therefore has to be written bucket by bucket with the true first position of the part of
the bucket it covers — never as one span — or a resumed/adopted request silently mixes two
contexts into one bucket mean.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

BLOCK_TOKENS = 64


def _ext():
    import vllm_kvmem_ext as ext
    return ext


class KvmemStatistics:
    """The sequence's mean-K statistics: one kvmem `RawKvStore` plus write bookkeeping."""

    def __init__(self, layers: int, kv_heads: int, head_dim: int,
                 block_tokens: int = BLOCK_TOKENS):
        self.layers = int(layers)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.width = self.kv_heads * self.head_dim
        self.block_tokens = int(block_tokens or BLOCK_TOKENS)
        self._store = _ext().RawKvStore(self.layers, self.width, self.block_tokens)
        self.frontier = 0            # first token position no statistics have been written for
        self.steps = 0
        self.bucket_writes = 0
        self.skipped = 0             # token ranges refused because they would go backwards
        self.last_error: Optional[str] = None

    # -- write path -------------------------------------------------------------------------
    def write_step(self, first_pos: int, tokens: int,
                   sums: Dict[int, "np.ndarray"]) -> int:
        """Write one engine step's per-layer bucket sums.

        first_pos: TRUE (original, un-compacted) position of the step's first token.
        tokens:    how many contiguous tokens the step covered.
        sums:      {attention-layer ordinal: float32 array [span, width]} where row `b` is the
                   sum of the tokens of absolute bucket `first_pos//block + b` that fall inside
                   this step.

        Returns the number of bucket writes performed.
        """
        first_pos = int(first_pos)
        tokens = int(tokens)
        if tokens <= 0 or not sums:
            return 0
        if first_pos < 0:
            self.last_error = "negative position"
            self.skipped += 1
            return 0
        block = self.block_tokens
        end = first_pos + tokens
        # Refuse to rewrite history: a step that starts before the frontier would reset buckets
        # that later steps already depend on. Resume/replay paths reset the store explicitly.
        if first_pos < self.frontier and self.frontier - first_pos >= block:
            self.last_error = f"step starts at {first_pos} < frontier {self.frontier}"
            self.skipped += 1
            return 0
        base = first_pos // block
        written = 0
        for layer, rows in sums.items():
            layer = int(layer)
            if rows is None or layer < 0 or layer >= self.layers:
                continue
            arr = np.ascontiguousarray(rows, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != self.width:
                self.last_error = f"layer {layer} sums shape {arr.shape} != (?, {self.width})"
                continue
            span = min(int(arr.shape[0]), (end - first_pos) // block + 2)
            for local in range(span):
                bucket = base + local
                lo = max(bucket * block, first_pos)
                hi = min(bucket * block + block, end)
                if hi <= lo:
                    continue
                # A bucket the step only partly covers must keep the rest of its tokens; the store
                # accumulates when pos0 is not bucket-aligned and resets when it is.
                self._store.write_layer_mean_sum(int(lo), int(hi - lo), layer, arr[local])
                written += 1
        self.bucket_writes += written
        self.steps += 1
        if end > self.frontier:
            self.frontier = end
        return written

    # -- read path (P1 uses these; kept here so the store has one owner) ----------------------
    def mean_k(self, block_id: int, layer: int) -> "np.ndarray":
        return np.asarray(self._store.mean_k(int(block_id), int(layer)), dtype=np.float32)

    def has_block(self, block_id: int) -> bool:
        return bool(self._store.has_block(int(block_id)))

    def n_tokens(self, block_id: int) -> int:
        return int(self._store.n_tokens(int(block_id)))

    # -- lifecycle ---------------------------------------------------------------------------
    def truncate_to(self, token_pos: int) -> None:
        self._store.truncate_to(int(token_pos))
        self.frontier = min(self.frontier, int(token_pos))

    def clear(self) -> None:
        self._store.clear()
        self.frontier = 0

    def mean_checkpoint(self, token_pos: int):
        return self._store.mean_checkpoint(int(token_pos))

    def restore_mean_checkpoint(self, token_pos: int, state) -> None:
        self._store.restore_mean_checkpoint(int(token_pos), state)

    def bytes(self) -> int:
        return int(self._store.allocated_bytes())

    def stats(self) -> dict:
        return {"layers": self.layers, "kv_heads": self.kv_heads, "head_dim": self.head_dim,
                "width": self.width, "block_tokens": self.block_tokens,
                "frontier": self.frontier, "steps": self.steps,
                "bucket_writes": self.bucket_writes, "skipped": self.skipped,
                "bytes": self.bytes(), "error": self.last_error}


def stats_from_bridge(bridge, layers: Iterable[int], first_pos: int, tokens: int) -> Dict[int, "np.ndarray"]:
    """Drain one step's bucket sums out of the engine bridge.

    `first_pos`/`tokens` describe the step: the bridge indexed its rows by
    `position // 64 - first_pos // 64`, so the span is derivable without a second host copy of the
    positions. Layers the engine marked unusable (recurrent layers, the MTP layer) report slot -1
    and are skipped.
    """
    block = BLOCK_TOKENS
    span = int(tokens) // block + 2
    out: Dict[int, "np.ndarray"] = {}
    for layer in layers:
        layer = int(layer)
        if layer < 0:
            continue
        rows = bridge.read_key_buckets(layer, span)
        if rows is None:
            continue
        out[layer] = np.ascontiguousarray(rows, dtype=np.float32)
    return out
