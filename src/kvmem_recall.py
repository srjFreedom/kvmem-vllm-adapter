"""Query frame + mean-K scoring, taken from kvmem's own recall surface (v0.19 design).

Reference: `backends/ninfer/.../storage/memory_statistics.cpp: score_memory_query`.

    frame = query_score_frame(frontier, 64, budget_blocks, sink_blocks, recent_blocks, fixed)
    view  = PageQueryView{layer_rows, rows, query_heads, kv_heads, head_dim, layer_data}
    scores = mean_key_retrieval_scores(statistics, blocks, {mean-q sums ...})   # this P1 path
          or PageSoftmaxCuda().score(statistics, view, frame)                   # P5 path
    propagate_query_scores(scores, frame, graph, alpha)                        # P3 path

Two things this module deliberately does NOT do:

* it does not own the statistics (kvmem's `RawKvStore` does: `kvmem_statistics.py`), and
* it does not decide the frame's sink/recent/budget in tokens - those arrive already expressed in
  statistics BLOCKS, because that is what `query_score_frame` and `pick_topk_blocks` speak.

A layer with no captured query rows is passed as `None`, which the binding turns into an ABSENT
layer (`layer_rows[layer] = 0`): upstream is explicit that such a layer "is absent, never a
zero-valued sample", and feeding zeros is what diluted our layer mean before.

Exact scoring semantics (read from `kvmem/src/host/mean_key_retrieval.cpp`, verified against an
independent reference to 1e-6 in tests/test_kvmem_recall.py) - re-derive nothing:

    group      = query_heads / kv_heads
    Mean: sums are divided by (tokens * group)      -> per-(kv-head, channel) mean query vector
    Max : the RAW token sum is used, undivided      -> one probe per query head
    per block, per layer:
        for each kv head:
            key_norm = |mean_k(block, layer, kv_head)|
            best     = max over the head's query probes of dot(q, key) / (|q|*key_norm + 1e-6)
        layer_score = sum(best over kv heads) / kv_heads
    score(block) = sum(layer_score over ACTIVE layers) / active_layers
Blocks with no statistics score 0.0 (they are skipped, not penalised), and a layer's absence is
skipped in both the sum and the denominator.
"""
from __future__ import annotations

import numpy as np

SCORERS = ("meanq-cosine", "page-softmax")

# Sink/recent/budget in TOKENS, published by the plugin from the kvmem config
# (--additional-config {"kvmem": {...}}). The split into blocks happens per store, because the
# statistics bucket (64) need not equal the selection block (32). Without it `query_score_frame`
# rejects the call outright ("invalid Query score budget or block size") - the selector's C++ config
# is not exposed to Python, so this is the channel.
_DEFAULTS = {"sink_tokens": 0, "recent_tokens": 0, "budget_tokens": 0, "block_tokens": 64}


def set_default_blocks(sink_tokens=0, recent_tokens=0, budget_tokens=0, block_tokens=64):
    """Publish the budget bands (tokens). Called once per process by the plugin."""
    _DEFAULTS.update({"sink_tokens": int(sink_tokens or 0),
                      "recent_tokens": int(recent_tokens or 0),
                      "budget_tokens": int(budget_tokens or 0),
                      "block_tokens": int(block_tokens or 64)})
    return dict(_DEFAULTS)


def default_blocks():
    return dict(_DEFAULTS)


def _ext():
    import vllm_kvmem_ext as ext
    return ext


class KvmemRecall:
    """Frame + scorer over one sequence's mean-K statistics."""

    def __init__(self, statistics, block_tokens: int = 64, sink_blocks: int = 0,
                 recent_blocks: int = 0, budget_blocks: int = 0,
                 aggregation: str = "mean", scorer: str = "page-softmax"):
        self.statistics = statistics
        self.block_tokens = int(block_tokens or 64)
        if not (sink_blocks or recent_blocks or budget_blocks):
            # Fall back to the published bands: the selector's C++ config is not visible here, so
            # the plugin publishes them once per process.
            sink_blocks = _DEFAULTS["sink_tokens"] // self.block_tokens
            recent_blocks = _DEFAULTS["recent_tokens"] // self.block_tokens
            budget_blocks = _DEFAULTS["budget_tokens"] // self.block_tokens
        self.sink_blocks = int(sink_blocks or 0)
        self.recent_blocks = int(recent_blocks or 0)
        self.budget_blocks = int(budget_blocks or 0)
        self.max_aggregation = str(aggregation).lower() != "mean"
        self.scorer = scorer if scorer in SCORERS else "page-softmax"
        self.last_frame = None
        self.last_error = None
        self.calls = 0
        self._softmax = None          # kvmem::PageSoftmaxCuda, created on first use
        self.device_bytes = 0

    # -- frame -----------------------------------------------------------------------------
    def frame(self, accepted_through: int, fixed_blocks=()):
        """The candidate/required page frame for this selection round."""
        frame = _ext().query_score_frame(
            int(accepted_through), self.block_tokens, self.budget_blocks,
            self.sink_blocks, self.recent_blocks, [int(b) for b in (fixed_blocks or ())])
        self.last_frame = frame
        return frame

    def prefix_required(self, prefix_tokens: int, media=()):
        """Sink + recent + the partial page replay appends into (upstream `query_prefix_required`)."""
        return list(_ext().query_prefix_required(
            int(prefix_tokens), self.block_tokens, self.sink_blocks, self.recent_blocks,
            [int(b) for b in (media or ())]))

    # -- score -----------------------------------------------------------------------------
    def scores(self, layer_rows, rows: int, query_heads: int, kv_heads: int, head_dim: int,
               blocks: int = None, frame=None):
        """One score per statistics block.

        layer_rows: {attention-layer ordinal: float32 array [rows, query_heads, head_dim]}.
                    Ordinals missing from the dict are ABSENT layers, not zero rows.
        """
        self.calls += 1
        stats = self.statistics
        if stats is None:
            self.last_error = "no statistics store"
            return None
        if int(rows or 0) <= 0:
            self.last_error = "no query rows"
            return None
        layers = int(getattr(stats, "layers", 0) or 0)
        if layers <= 0:
            self.last_error = "statistics store has no layers"
            return None
        ordered = []
        for layer in range(layers):
            rows_for_layer = (layer_rows or {}).get(layer)
            if rows_for_layer is None:
                ordered.append(None)                      # absent, not zero-filled
                continue
            array = np.ascontiguousarray(rows_for_layer, dtype=np.float32)
            if array.shape[1] != int(query_heads) or array.shape[2] != int(head_dim):
                self.last_error = f"layer {layer} rows shape {array.shape} != (*,{query_heads},{head_dim})"
                return None
            ordered.append(array[: int(rows)])
        count = int(blocks if blocks is not None else getattr(stats, "frontier", 0) // self.block_tokens)
        if count <= 0:
            self.last_error = "no blocks to score"
            return None
        try:
            if self.scorer == "page-softmax":
                # v0.19 default: softmax over the candidate frame's logits, per query row and head.
                # It takes the FRAME, so the candidate set, the score mask and the accepted extent
                # all come from query_score_frame instead of a bare block count.
                if frame is None:
                    frame = self.frame(int(getattr(stats, "frontier", 0)))
                if frame is None or int(frame.accepted_through) <= 0:
                    self.last_error = "no query frame"
                    return None
                if self._softmax is None:
                    try:
                        import torch
                        device = int(torch.cuda.current_device())
                    except Exception:
                        device = 0
                    self._softmax = _ext().PageSoftmaxCuda(device)
                out = self._softmax.score(stats._store, ordered, int(rows), int(query_heads),
                                          int(kv_heads), int(head_dim), frame,
                                          bool(self.max_aggregation))
                try:
                    self.device_bytes = int(self._softmax.device_bytes())
                except Exception:
                    pass
            else:
                out = _ext().query_scores(stats._store, ordered, int(rows), int(query_heads),
                                          int(kv_heads), int(head_dim), count,
                                          bool(self.max_aggregation), None)
        except Exception as exc:                          # never take the selection down with it
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
        self.last_error = None
        return out

    def stats(self) -> dict:
        return {"scorer": self.scorer, "block_tokens": self.block_tokens,
                "sink_blocks": self.sink_blocks, "recent_blocks": self.recent_blocks,
                "budget_blocks": self.budget_blocks,
                "aggregation": "max" if self.max_aggregation else "mean",
                "calls": self.calls, "error": self.last_error}


def compare_scores(core, legacy, blocks=None, top: int = 10) -> dict:
    """How two score vectors agree: a rewrite has to be checked, not assumed.

    Reports the ranges, the top-`top` block overlap and the rank correlation over the blocks both
    sides scored, so "the numbers moved" and "the ranking collapsed" cannot look the same.
    """
    out = {}
    try:
        a = np.asarray(core, dtype=np.float64)
        b = np.asarray(legacy, dtype=np.float64)
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
        if n == 0:
            return {"comparable": False}
        finite = np.isfinite(a) & np.isfinite(b)
        out.update({"comparable": True, "blocks": int(n),
                    "core_min": float(np.nanmin(a)), "core_max": float(np.nanmax(a)),
                    "legacy_min": float(np.nanmin(b)), "legacy_max": float(np.nanmax(b)),
                    "finite_both": int(finite.sum()),
                    "core_nonfinite": int((~np.isfinite(a)).sum()),
                    "legacy_nonfinite": int((~np.isfinite(b)).sum())})
        if finite.sum() >= 3:
            af = a[finite]
            bf = b[finite]
            # Pearson on ranks == Spearman, without scipy.
            ra = np.argsort(np.argsort(af))
            rb = np.argsort(np.argsort(bf))
            if ra.std() > 0 and rb.std() > 0:
                out["rank_corr"] = float(np.corrcoef(ra, rb)[0, 1])
            k = min(int(top), af.size)
            top_a = set(np.argsort(-af)[:k].tolist())
            top_b = set(np.argsort(-bf)[:k].tolist())
            out["top_overlap"] = len(top_a & top_b) / float(k)
            out["top_k"] = k
            if blocks is not None and len(blocks) >= n:
                out["core_top"] = [int(blocks[i]) for i in np.argsort(-af)[:k]]
                out["legacy_top"] = [int(blocks[i]) for i in np.argsort(-bf)[:k]]
    except Exception as exc:
        return {"comparable": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


def align_scores(core, core_block_tokens: int, legacy_blocks: int,
                 legacy_block_tokens: int):
    """Resample core scores onto the LEGACY block grid before comparing them.

    The two sides do not score the same token spans: kvmem's statistics bucket is 64 tokens while
    the shipped selection block is 32 (`--kvmem-block-tokens 32`), so comparing element i to element
    i compared different halves of the context - which is what produced rank_corr ~ -0.02 and a
    meaningless top_overlap of 0.
    """
    core = list(core)
    if not core or legacy_blocks <= 0:
        return []
    core_block_tokens = int(core_block_tokens or 0)
    legacy_block_tokens = int(legacy_block_tokens or 0)
    if not core_block_tokens or not legacy_block_tokens or core_block_tokens == legacy_block_tokens:
        return core[:legacy_blocks]
    if core_block_tokens % legacy_block_tokens == 0:
        ratio = core_block_tokens // legacy_block_tokens
        return [core[min(len(core) - 1, j // ratio)] for j in range(legacy_blocks)]
    if legacy_block_tokens % core_block_tokens == 0:
        ratio = legacy_block_tokens // core_block_tokens
        out = []
        for j in range(legacy_blocks):
            part = core[j * ratio:(j + 1) * ratio]
            out.append(float(np.mean(part)) if part else 0.0)
        return out
    # Coarse grids only: map each legacy block's midpoint to the core block containing it.
    out = []
    for j in range(legacy_blocks):
        mid = (j + 0.5) * legacy_block_tokens
        index = int(mid // core_block_tokens)
        out.append(core[min(len(core) - 1, max(0, index))])
    return out
