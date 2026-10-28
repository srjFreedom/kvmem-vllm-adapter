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
        # A non-finite page contribution must never enter the accumulator: one NaN poisons the
        # block's mean key, and a poisoned key makes `scores()` unusable for the whole request.
        # Device-side replace, no host sync.
        try:
            vec = torch.where(torch.isfinite(vec), vec, torch.zeros_like(vec))
        except Exception:
            pass
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
        try:
            vec = torch.where(torch.isfinite(vec), vec, torch.zeros_like(vec))
        except Exception:
            pass
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
        try:
            vec = torch.where(torch.isfinite(vec), vec, torch.zeros_like(vec))
        except Exception:
            pass
        self._query[layer] = vec

    # --- scoring ----------------------------------------------------------------------
    def scores(self, blocks=None):
        """Cosine similarity per block (mean over the layers that actually contributed).

        A single non-finite layer or block would otherwise poison the ENTIRE ranking: one NaN query
        layer makes the layer-mean NaN, so every cosine is NaN and the selection degenerates to
        whatever order the blocks happen to be in (measured: `q_norm=NaN`, `nonfinite=190`, all
        scores flat and the question block dropped). So non-finite contributors are EXCLUDED here
        instead of being averaged in, and a block that carries non-finite key statistics returns
        NaN, which the caller turns into its "never pick" floor.
        """
        if self._query is None:
            raise RuntimeError("no query captured: retrieval needs a query summary")
        blocks = sorted(self._sums) if blocks is None else list(blocks)
        counts = getattr(self, "_query_counts", None)
        finite_layers = torch.isfinite(self._query).all(dim=1)
        if counts is not None:
            # Layers that never contributed a chunk must not dilute the mean.
            finite_layers = finite_layers & (counts > 0)
        active = finite_layers
        dropped = int((counts > 0).sum() - finite_layers.sum()) if counts is not None else 0
        if bool(finite_layers.any()):
            q = self._query[finite_layers].mean(dim=0)
        else:
            q = self._query.mean(dim=0)
        qn = q.norm().clamp_min(1e-6)
        # Measurement for a scoring that comes back all-zero: with cosine == 0 for EVERY block
        # exactly one side must be a zero vector, and nothing else in the pipeline can tell them
        # apart. Record both magnitudes plus a few raw scores.
        _diag = {"q_norm": float(q.norm()),
                 "active_layers": int(active.sum()),
                 "dropped_layers": int(dropped),
                 "layers": int(self.layers)}
        out = []
        bad_blocks = 0
        for block in blocks:
            sums = self._sums[block]
            counts = self._counts[block].clamp_min(1e-6).unsqueeze(-1)
            k = (sums / counts).mean(dim=0)
            if "k_norm" not in _diag:
                _diag["k_norm"] = float(k.norm())
                _diag["k_norm_raw"] = float(sums.norm())
            if not bool(torch.isfinite(k).all()):
                # A block whose key statistics carry NaN/inf must never win a ranking slot: report
                # NaN so the caller leaves it on the "never pick" floor.
                bad_blocks += 1
                out.append(float("nan"))
                continue
            out.append(float((k @ q) / (k.norm().clamp_min(1e-6) * qn)))
        _diag["raw"] = [round(float(v), 4) for v in out[:5]]
        _diag["bad_blocks"] = int(bad_blocks)
        self.last_diag = _diag
        return out

    def covered_blocks(self):
        return sorted(self._sums)

    def bytes_used(self):
        return (len(self._sums) * self.layers * self.head_dim * 4
                + (self._query.numel() * 4 if self._query is not None else 0))
