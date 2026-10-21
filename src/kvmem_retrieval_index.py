"""Per-block pre-RoPE key statistics + query sums for KVMem's mean-K retrieval.

KVMem's contract asks the backend for *pre-RoPE statistics* (`capabilities.pre_rope_statistics`):
per-block key information in a position-invariant frame, plus the current query summary. This class
accumulates exactly that while the engine runs, so a reselect can rank blocks by content instead of
by recency.

Scoring follows kvmem's mean-K semantics (`mean_key_retrieval_scores`):
  * mean over the tokens of a block, over the participating layers and over the KV heads,
  * cosine similarity against the mean query,
  * scores are returned per block, in block order.

Memory: one float32 vector per (block, layer). With B=32768, block=128 there are 256 blocks; at
17 layers and 256 channels that is ~4.5 MB - negligible next to the KV payload itself.
"""
from __future__ import annotations

import torch


class KvmemRetrievalIndex:
    def __init__(self, block_tokens, page_tokens, head_dim, layers, dtype=torch.float32):
        if block_tokens % page_tokens:
            raise ValueError("a selection block must be a whole number of pages")
        self.block_tokens = int(block_tokens)
        self.page_tokens = int(page_tokens)
        self.head_dim = int(head_dim)
        self.layers = int(layers)
        self.dtype = dtype
        self._sums: dict[int, torch.Tensor] = {}     # block -> (layers, head_dim) float32
        self._counts: dict[int, torch.Tensor] = {}   # block -> (layers,) float32
        self._query: torch.Tensor | None = None      # (layers, head_dim) float32

    # --- accumulation -----------------------------------------------------------------
    def block_of(self, page):
        return int(page) * self.page_tokens // self.block_tokens

    def add(self, page, layer, k_pre, weights=None):
        """Accumulate one page's pre-RoPE keys for `layer`.

        k_pre: (tokens, head_dim) - mean over the KV heads is taken here, because KVMem averages
        across KV heads when it turns keys into a score.
        """
        block = self.block_of(page)
        if k_pre.numel() == 0:
            return
        vec = k_pre.to(self.dtype)
        if vec.ndim == 3:                            # (heads, tokens, head_dim)
            vec = vec.mean(dim=0)
        if vec.ndim == 2:
            vec = vec.mean(dim=0)
        # Page bytes arrive from the host while the index may live on the GPU (the query path runs
        # on CUDA), so pin everything to the index's own device.
        device = getattr(self, "device", None)
        if device is None:
            device = self.device = vec.device
        vec = vec.to(device)
        sums = self._sums.get(block)
        if sums is None:
            sums = torch.zeros(self.layers, self.head_dim, dtype=self.dtype, device=device)
            self._counts[block] = torch.zeros(self.layers, dtype=self.dtype, device=device)
        sums[layer] += vec
        self._counts[block][layer] += 1.0
        self._sums[block] = sums

    def begin_query(self, layer):
        """Reset this layer's query accumulator (called once per step, before its chunks)."""
        if self._query is None:
            _dev = getattr(self, "device", None)
            self._query = torch.zeros(self.layers, self.head_dim, dtype=self.dtype,
                                      device=_dev) if _dev is not None else \
                torch.zeros(self.layers, self.head_dim, dtype=self.dtype)
        self._query[layer] = 0.0
        _dev = getattr(self, "device", None)
        self._query_counts = getattr(self, "_query_counts",
                                     torch.zeros(self.layers, dtype=self.dtype,
                                                 device=_dev) if _dev is not None else
                                     torch.zeros(self.layers, dtype=self.dtype))
        self._query_counts[layer] = 0.0

    def add_query(self, layer, q_pre):
        """Accumulate one chunk's pre-RoPE query into the running per-layer sum."""
        if q_pre.numel() == 0:
            return
        vec = q_pre.to(self.dtype)
        if vec.ndim == 3:
            vec = vec.mean(dim=1)
        if vec.ndim == 2:
            vec = vec.mean(dim=0)
        if self._query is None:
            _dev = getattr(self, "device", None)
            self._query = torch.zeros(self.layers, self.head_dim, dtype=self.dtype,
                                      device=_dev) if _dev is not None else \
                torch.zeros(self.layers, self.head_dim, dtype=self.dtype)
        if not hasattr(self, "_query_counts"):
            self._query_counts = torch.zeros(self.layers, dtype=self.dtype)
        device = getattr(self, "device", None)
        vec = vec.to(device) if device is not None else vec
        self._query[layer] += vec
        self._query_counts[layer] += 1.0

    def set_query(self, layer, q_pre):
        """Record this step's pre-RoPE query summary for `layer` (mean over its heads/tokens)."""
        if q_pre.numel() == 0:
            return
        vec = q_pre.to(self.dtype)
        if vec.ndim == 3:
            vec = vec.mean(dim=1)
        if vec.ndim == 2:
            vec = vec.mean(dim=0)
        if self._query is None:
            _dev = getattr(self, "device", None)
            self._query = torch.zeros(self.layers, self.head_dim, dtype=self.dtype,
                                      device=_dev) if _dev is not None else \
                torch.zeros(self.layers, self.head_dim, dtype=self.dtype)
        self._query[layer] = vec

    # --- scoring ----------------------------------------------------------------------
    def scores(self, blocks=None):
        """Cosine similarity per block (mean over the layers that actually contributed)."""
        if self._query is None:
            raise RuntimeError("no query captured: retrieval needs a query summary")
        blocks = sorted(self._sums) if blocks is None else list(blocks)
        q = self._query.mean(dim=0)
        counts = getattr(self, "_query_counts", None)
        if counts is not None:
            # Layers that never contributed a chunk must not dilute the mean.
            active = counts > 0
            if bool(active.any()):
                q = self._query[active].mean(dim=0)
        qn = q.norm().clamp_min(1e-6)
        out = []
        for block in blocks:
            sums = self._sums[block]
            counts = self._counts[block].clamp_min(1e-6).unsqueeze(-1)
            k = (sums / counts).mean(dim=0)
            out.append(float((k @ q) / (k.norm().clamp_min(1e-6) * qn)))
        return out

    def covered_blocks(self):
        return sorted(self._sums)

    def bytes_used(self):
        return (len(self._sums) * self.layers * self.head_dim * 4
                + (self._query.numel() * 4 if self._query is not None else 0))
