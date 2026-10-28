# The sparse orchestrator (KVMem "A mode" on vLLM): ties the kvmem selector, the sparse-view
# contract, the re-RoPE and the host pool together, one call per engine step.
#
# Invariants (they are what make the sparse view exact and non-drifting):
#   1. The host page store holds the CANONICAL bytes of every written page at its TRUE position.
#   2. A device page is always derived from canonical by exactly ONE rotation hop:
#      rotate(canonical[page], (slot - page) * page_tokens).
#   3. New tokens are written by the engine at their COMPACTED positions (because the caller
#      overrides `positions`), so commit_new_tokens() un-rotates them back to canonical once.
#   4. Reselect never rotates a device page that already carries a rotation: it always rebuilds
#      from canonical. This is why the fp8 re-quantization error cannot compound.
import torch

from kvmem_sparse_rope import FP8_DTYPE, _dequant, _quantize, delta_rotate, rotary_inv_freq
from kvmem_sparse_view import build_sparse_view, rotation_deltas


def rotate_page_batch(pages, deltas, inv_freq, rotary_dim, k_slice):
    """Rotate a STACK of pages by a per-page delta in one shot.

    pages:  [n, heads, tokens, content] fp8 bytes; deltas: [n] token offsets.
    Doing this per page cost ~17 tiny kernels per page (34k launches per reselect); one batched
    pass per layer replaces that with a single kernel.
    """
    if not len(pages):
        return pages
    out = pages.clone()
    half = rotary_dim // 2
    device = pages.device
    # Everything must live on the pages' device (the store is host-resident, the KV is not).
    angle = (deltas.to(device=device, dtype=torch.float64)[:, None, None, None] *
             inv_freq.to(device=device, dtype=torch.float64)[None, None, None, :])
    cos = torch.cos(angle).to(torch.float32)
    sin = torch.sin(angle).to(torch.float32)
    k = _dequant(out[..., k_slice])
    x1 = k[..., :half]
    x2 = k[..., half:rotary_dim]
    rotated = torch.empty_like(k)
    rotated[..., :half] = x1 * cos - x2 * sin
    rotated[..., half:rotary_dim] = x1 * sin + x2 * cos
    rotated[..., rotary_dim:] = k[..., rotary_dim:]
    out[..., k_slice] = _quantize(rotated)
    return out


class HostPageStore:
    """Canonical (true-position) bytes per physical page."""

    def __init__(self):
        self._pages = {}

    def __len__(self):
        return len(self._pages)

    @property
    def pages(self):
        return self._pages

    def has(self, page):
        return int(page) in self._pages

    def get(self, page):
        return self._pages[int(page)]

    def put(self, page, tensor):
        # The store is HOST memory: keeping GPU clones here exhausts VRAM (a 39K-token request
        # OOMed the engine by holding 2,000 pages x 17 layers on the device).
        if isinstance(tensor, (list, tuple)):
            self._pages[int(page)] = [t.detach().to("cpu").clone() for t in tensor]
        else:
            self._pages[int(page)] = tensor.detach().to("cpu").clone()

    def drop(self, page):
        self._pages.pop(int(page), None)

    def __len__(self):
        return len(self._pages)


class KvmemHostStore:
    """Host page store backed by KVMem's RAM tier (the H budget is enforced by KVMem).

    One record per logical page. The record layout is discovered from the payload (per-layer
    shape/dtype), so the same store serves any layer geometry: the paged attention layers do not
    all have to share a shape. Capacity is charged by KVMem; over-subscription evicts the oldest
    page instead of letting the process RSS grow without bound.
    """

    def __init__(self, capacity_bytes, page, layers=None, record_bytes=None, tensor_shape=None,
                 disk_bytes=0, disk_dir=None):
        import vllm_kvmem_ext as ext
        self._ext = ext
        self._storage = ext.HostKvStorage(int(capacity_bytes))
        self._records = {}
        self._layouts = {}        # page -> [(shape, dtype, nbytes), ...]
        self._order = []          # insertion order, for eviction
        # Disk tier (D budget). When H is exhausted the oldest page is staged out here instead of
        # being dropped, and pulled back on demand - the same two-tier behaviour the llama/ninfer
        # backends get from kvmem's nvme_kv_tier.
        self._disk = None
        self._disk_slot = {}      # page -> slot in the tier
        self._disk_free = []
        self._disk_bytes = int(disk_bytes or 0)
        if self._disk_bytes > 0:
            try:
                import os as _os
                directory = disk_dir or _os.path.join("/tmp", "kvmem_disk_tier")
                _os.makedirs(directory, exist_ok=True)
                slot_bytes = None
                if record_bytes:
                    slot_bytes = int(record_bytes)
                elif tensor_shape and layers:
                    n = 1
                    for d in tensor_shape:
                        n *= int(d)
                    slot_bytes = n * int(layers)
                if slot_bytes:
                    self._disk = ext.NvmeKvTier(directory, self._disk_bytes, slot_bytes)
                    if not self._disk.enabled():
                        self._disk = None
            except Exception:
                self._disk = None
        self.page = int(page)
        self.layers = int(layers) if layers else 0
        self.record_bytes = int(record_bytes) if record_bytes else 0
        self.tensor_shape = tuple(tensor_shape) if tensor_shape else None

    # --- store protocol ---------------------------------------------------------------
    @property
    def pages(self):
        return self._records

    def has(self, page):
        return int(page) in self._records

    def __len__(self):
        return len(self._records)

    def put(self, page, tensor):
        import torch
        layers = list(tensor) if isinstance(tensor, (list, tuple)) else [tensor]
        arrays, layout = [], []
        for item in layers:
            cpu = item.detach().to("cpu").contiguous()
            arrays.append(cpu.numpy().tobytes())
            layout.append((tuple(cpu.shape), str(cpu.dtype).replace("torch.", ""), cpu.nbytes))
        buf = b"".join(arrays)
        while True:
            try:
                record = self._storage.allocate(len(buf))
                break
            except Exception:
                if not self.evict_oldest(stage_out=True):
                    raise
        record.write(buf)
        record.seal()
        page = int(page)
        if page in self._records:
            self._order.remove(page)
        self._records[page] = record
        self._layouts[page] = layout
        self._order.append(page)

    def get(self, page):
        import torch
        page = int(page)
        if page not in self._records:
            raw = self.disk_read(page)
            if raw is None:
                raise KeyError(page)
        else:
            raw = self._records[page].read()
        out, offset = [], 0
        for shape, dtype, nbytes in self._layouts[page]:
            chunk = torch.frombuffer(bytearray(raw[offset:offset + nbytes]), dtype=torch.uint8)
            out.append(chunk.clone().view(getattr(torch, dtype)).reshape(shape))
            offset += nbytes
        return out

    # --- budget ------------------------------------------------------------------------
    def evict_oldest(self, stage_out=False):
        if not self._order:
            return False
        page = self._order.pop(0)
        record = self._records.pop(page, None)
        layout = self._layouts.get(page)
        if stage_out and self._disk is not None and record is not None:
            slot = self._disk_free.pop() if self._disk_free else None
            if slot is None:
                used = len(self._disk_slot)
                if used < int(self._disk.slot_count()):
                    slot = used
            if slot is not None:
                try:
                    self._disk.write_block(int(slot), record.read())
                    self._disk_slot[page] = int(slot)
                    self.disk_staged = int(getattr(self, "disk_staged", 0)) + 1
                    self._disk_layout = getattr(self, "_disk_layout", {})
                    self._disk_layout[page] = layout
                    return True
                except Exception:
                    pass
        self._layouts.pop(page, None)
        return True

    def disk_read(self, page):
        """Bytes staged out for `page`, or None."""
        if self._disk is None or page not in self._disk_slot:
            return None
        layout = getattr(self, "_disk_layout", {}).get(page) or self._layouts.get(page)
        total = sum(int(n) for _s, _d, n in layout) if layout else None
        if total is None:
            return None
        raw = self._disk.read_block(int(self._disk_slot[page]), int(total))
        self._layouts[page] = layout
        return raw

    def disk_stats(self):
        return {"staged": int(getattr(self, "disk_staged", 0)),
                "slots": len(getattr(self, "_disk_slot", {}) or {}),
                "enabled": self._disk is not None}

    def has_disk(self, page):
        return self._disk is not None and int(page) in self._disk_slot

    def charged_bytes(self):
        return int(self._storage.charge())

    def capacity_bytes(self):
        return int(self._storage.capacity())


def _probe_write(record) -> None:
    """Append one diagnostic record to the KVMem probe file (best effort, never raises).

    The orchestrator has no log handle of its own; the runtime/plugin own the probe path, and this
    file is the one every other layer writes to.
    """
    try:
        import json
        import os
        path = os.environ.get("KVMEM_PROBE_LOG") or "/tmp/kvmem_sparse.jsonl"
        record = dict(record)
        record.setdefault("pid", os.getpid())
        with open(path, "a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


class SparseOrchestrator:
    """Per-request sparse working-set manager for one TP rank.

    kv_tensor: [pages, kv_heads, page_tokens, content] uint8 (packed fp8 KV: K then V).
    """

    def __init__(self, selector, kv_tensors, page_tokens, head_dim, rotary_dim, base,
                 k_slice=None, store=None, reselect_interval=64):
        self.selector = selector
        # A logical KV page spans EVERY paged attention layer: the block table is shared, so all
        # layers must be moved and re-RoPE'd together. Wiring only layer 0 left the other 16 with
        # K at the wrong positions (that was the long-standing "gibberish" cause).
        if isinstance(kv_tensors, (list, tuple)):
            self.kvs = list(kv_tensors)
        else:
            self.kvs = [kv_tensors]
        self.page_tokens = int(page_tokens)
        self.head_dim = int(head_dim)
        self.rotary_dim = int(rotary_dim)
        self.inv_freq = rotary_inv_freq(self.rotary_dim, base)
        self.k_slice = k_slice if k_slice is not None else slice(0, self.head_dim)
        self.store = store if store is not None else HostPageStore()
        self.reselect_interval = int(reselect_interval)
        # A full rebuild costs pages x layers of H2D, so trigger it by NEW TOKENS rather than by
        # step count: decode then reselects every `reselect_token_budget` tokens, not every N
        # steps. 0 disables the token trigger.
        self.reselect_token_budget = 4096
        self._tokens_at_reselect = 0
        # When set, the attention view comes from KVMem's own planner (plan_working_set) instead of
        # being built here. tests/kvmem_view_test.py proves the two are equivalent.
        self.planner = None
        # Content-based ranking (mean-K over pre-RoPE keys). When present, reselect asks it for
        # scores and hands them to KVMem's selector, which is what makes Retrieval mode real.
        self.retrieval_index = None
        self._inv_freq_ready = True
        self.session_id = 1
        self.last_visible_tokens = 0
        # The caller owns slot allocation: every request gets a DISJOINT set of device slots,
        # otherwise two concurrent requests would compact into 0..|sel|-1 and clobber each other.
        self.slots = []
        self.resident = set()          # pages whose bytes currently live on the device
        self.slot_of = {}              # page -> compacted slot
        # The rotation already baked into the STORE's bytes for each page. The store is never
        # rewritten, so every device build is exactly one hop from it (no accumulation).
        self.store_delta = {}
        self.steps_since_reselect = 0
        self.total_tokens = 0
        self._tokens_at_reselect = 0
        self.last_view = None
        # The rotation delta currently baked into each resident device page.
        self.baked_delta = {}

    # --- conversation growth ---------------------------------------------------------
    def note_tokens(self, total_tokens, scores=None):
        """Register the conversation length (counts tokens the engine has computed)."""
        self.total_tokens = int(total_tokens)
        self.selector.set_context_tokens(self.total_tokens)
        if scores is not None:
            self.selector.set_block_scores(list(scores))

    def should_reselect(self):
        # Compaction rewrites the block table and the scheduler's token book together (three-book
        # reconciliation, see the BlockChannel compact message), so it is opt-in: the default keeps
        # vLLM's original monotone path and uses kvmem only for the H/D tiers and retrieval.
        if not bool(getattr(self, "inrequest_compaction", False)):
            return False
        # KVMem semantics: reselect once NEW tokens since the last reselect reach the generation
        # reserve R - that is the point where the pool (B history + R output) is full. The baseline
        # is set to the prompt length when prefill completes (see VllmSparseRuntime.after_step), so
        # R counts OUTPUT tokens, not the prefill.
        since = self.total_tokens - self._tokens_at_reselect
        if self.reselect_token_budget and since >= self.reselect_token_budget:
            return True
        # R is the authority. The planner carries it when the native extension is available; the
        # runtime copies the same R onto `gen_reserve` so the cadence still holds without it.
        reserve = 0
        if self.planner is not None and getattr(self.planner, "gen_reserve", 0):
            reserve = int(self.planner.gen_reserve)
        elif getattr(self, "gen_reserve", 0):
            reserve = int(self.gen_reserve)
        if reserve and since >= reserve:
            return True
        return self.steps_since_reselect >= self.reselect_interval

    # --- engine-side new tokens ------------------------------------------------------
    @property
    def kv(self):
        return self.kvs[0]

    def _baked_delta(self, baked, slot, page):
        """The RoPE rotation (`tokens`) the stored bytes carry, relative to the page's LOGICAL start.

        A page's tokens sit at true positions `page*page_tokens + i`. "Baked" is how far the bytes
        in hand were rotated away from that frame: 0 when the engine wrote them at their true
        positions (an uncompacted prefill), `(rank - page) * page_tokens` when they were written
        into a compacted view at `rank`.

        `baked` is authoritative and the caller passes it. The `slot - page` fallback only
        reproduces it while a request's engine page ids happen to run parallel to its compacted
        ranks -- true for the first request out of the pool and for the tests, NOT true across a
        resume or for a second concurrent request, whose pool pages start at a different base.
        Getting that wrong bakes a constant `(base_new - base_old)` rotation into the whole
        restored context, which silently degenerates the answer instead of failing.
        """
        if baked is not None:
            return int(baked)
        return (int(slot) - int(page)) * self.page_tokens

    def write_page(self, page, bytes_as_written, slot, baked=None):
        """Record a page whose bytes are `bytes_as_written`, baked `baked` tokens off its logical
        position (see `_baked_delta`).

        The store keeps those bytes verbatim; the device copy is one hop from them, so a page can
        be re-derived any number of times without compounding fp8 rounding.
        """
        delta = self._baked_delta(baked, slot, page)
        layers = (list(bytes_as_written) if isinstance(bytes_as_written, (list, tuple))
                  else [bytes_as_written] * len(self.kvs))
        # The store always holds one tensor per paged layer, so every consumer can index by layer.
        self.store.put(page, layers)
        self.store_delta[int(page)] = delta
        self._feed_retrieval(page, layers)
        for layer, kv in enumerate(self.kvs):
            kv[int(slot)].copy_(layers[layer])
        self.resident.add(int(page))
        self.slot_of[int(page)] = int(slot)
        self.baked_delta[int(page)] = delta

    def _feed_retrieval(self, page, layers):
        """Accumulate this page's pre-RoPE KEY statistics into the retrieval index.

        The bytes are already in hand when a page is committed, so this costs no extra I/O: only a
        pass over `page_tokens x head_dim` per layer. `store_delta[page]` is the rotation baked into
        them relative to the page's logical start, so un-rotating by its negative takes them back to
        the frame they were written in (the same key content at a fixed offset), which is what the
        mean-K index compares.
        """
        index = getattr(self, "retrieval_index", None)
        if index is None or not layers:
            return
        delta = int(self.store_delta.get(int(page), 0))
        if delta and not self._inv_freq_ready:
            return
        import torch
        block = int(page) * self.page_tokens // index.block_tokens
        for layer, page_bytes in enumerate(layers):
            try:
                k = _dequant(page_bytes[..., self.k_slice])        # (heads, tokens, dim)
            except Exception:
                return
            if delta:
                # Page bytes live on the host; inv_freq may have been built on the GPU by an earlier
                # device-side path, so align it with the tensor being rotated.
                inv = self.inv_freq.to(k.device) if hasattr(self.inv_freq, "to") else self.inv_freq
                k = delta_rotate(k, -float(delta), inv, self.rotary_dim)
            index.add(int(page), layer, k)
        return block

    def commit_written_page(self, page, slot, baked=None):
        """Record a page the ENGINE just wrote into device page `slot` (live path).

        `baked` is the rotation the engine actually applied (see `_baked_delta`): 0 while the
        request prefills at true positions, `(rank - page) * page_tokens` once a compacted view has
        re-based its positions. The caller knows which; without it the `slot - page` fallback bakes
        the engine's page-id base into the page, which only cancels inside one request.
        """
        delta = self._baked_delta(baked, slot, page)
        page_bytes = [kv[int(slot)].clone() for kv in self.kvs]
        self.store.put(page, page_bytes)
        self.store_delta[int(page)] = delta
        self._feed_retrieval(page, page_bytes)
        self.resident.add(int(page))
        self.slot_of[int(page)] = int(slot)
        self.baked_delta[int(page)] = delta

    def restore_page(self, page, slot=None, channel=None, request_id=None, rank=None):
        """Bring an evicted page back into `slot`, rotated to `rank` (or to `slot`) in the view."""
        if slot is None:
            if channel is None:
                raise RuntimeError("restore_page needs a slot or a block channel")
            slot = channel.request(request_id or "restore", 1)[0]
        # The store keeps the canonical bytes forever, so a restore is exactly ONE rotation hop by
        # `needed - store_delta` (never a rotation of a rotation). The control plane - asking the
        # scheduler for a free block - is the caller's job; this is the data plane.
        #
        # `needed` is the position the bytes must end up at, in the SAME semantic frame as
        # `store_delta` (relative to the page's logical start). Inside one request's own row the
        # compacted rank and the engine page id run parallel, so `slot - page` works; a resume
        # writes the page at a different base, and there only the RANK describes the new position.
        page, slot = int(page), int(slot)
        if not self.store.has(page):
            raise RuntimeError(f"cannot restore page {page}: no canonical bytes")
        if rank is not None:
            needed = (int(rank) - page) * self.page_tokens
        else:
            needed = (slot - page) * self.page_tokens
        step = needed - self.store_delta.get(page, 0)
        for layer, kv in enumerate(self.kvs):
            dev = self.store.get(page)[layer].to(kv.device, non_blocking=True)
            kv[slot].copy_(dev if step == 0 else self._rotate_page_bytes(dev, step))
        self.resident.add(page)
        self.slot_of[page] = slot
        self.baked_delta[page] = needed
        return needed

    # --- v0.19 scoring path ------------------------------------------------------------
    def core_selection_scores(self):
        """Page-softmax scores from kvmem's own scorer, resampled to the selection-block grid.

        This IS the ranking input now. Upstream's default configuration is page-softmax with LRFU
        off (resolve_query_scoring: softmax implies mean aggregation and no LRFU), so the scores
        here are attention-mass shares that are already comparable between rounds - the LRFU decay
        exists for the uncalibrated cosine scorer we no longer use.

        Returns None when there is nothing to score (no statistics / no query yet), which the caller
        treats as "keep the current working set" rather than as a ranking of zeros.
        """
        stats = getattr(self, "statistics", None)
        if stats is None:
            return None
        try:
            from vllm import kvmem_bridge as bridge
            from kvmem_recall import KvmemRecall, align_scores
        except Exception:
            return None
        if not getattr(bridge, "ARMED", False):
            return None
        layer_rows, rows = {}, 0
        try:
            for layer, (q, _pos) in bridge.read_all().items():
                if q is None or q.numel() == 0:
                    continue
                q = q.detach().to("cpu", non_blocking=False).float()
                layer_rows[int(layer)] = q
                rows = max(rows, int(q.shape[0]))
        except Exception:
            return None
        if rows <= 0 or not layer_rows:
            return None
        recall = getattr(self, "_recall", None)
        if recall is None:
            cfg = getattr(self.selector, "config_", None)
            recall = KvmemRecall(
                stats,
                block_tokens=int(getattr(stats, "block_tokens", 64) or 64),
                sink_blocks=int(getattr(cfg, "sink_tokens", 0) or 0) // 64,
                recent_blocks=int(getattr(cfg, "recent_tokens", 0) or 0) // 64,
                budget_blocks=int(getattr(cfg, "budget_tokens", 0) or 0) // 64,
                aggregation="mean")
            self._recall = recall
        frontier = int(getattr(stats, "frontier", 0))
        if frontier <= 0:
            return None
        query_heads = int(layer_rows[next(iter(layer_rows))].shape[1])
        core = recall.scores(layer_rows, rows, query_heads,
                             int(getattr(stats, "kv_heads", 0) or 0),
                             int(getattr(stats, "head_dim", 0) or 0),
                             frontier // int(recall.block_tokens))
        if core is None:
            self._core_error = recall.last_error
            return None
        self._core_error = None
        self._last_core = list(core)
        if not getattr(self, "_core_logged", False):
            self._core_logged = True
            try:
                finite = [float(v) for v in core if v == v]
                order = sorted(range(len(core)), key=lambda i: -float(core[i]))[:5]
                _probe_write({"kind": "core_scores", "scorer": recall.scorer,
                              "aggregation": "max" if recall.max_aggregation else "mean",
                              "blocks": len(core), "finite": len(finite),
                              "min": min(finite) if finite else None,
                              "max": max(finite) if finite else None,
                              "sum": round(sum(finite), 6) if finite else None,
                              "top": order, "device_bytes": getattr(recall, "device_bytes", 0),
                              "frontier": frontier, "rows": rows})
            except Exception:
                pass
        selection_blocks = int(getattr(self.selector, "selection_blocks", lambda: 0)() or 0)
        if selection_blocks <= 0:
            return None
        aligned = align_scores(core, int(recall.block_tokens), selection_blocks,
                              int(self.selector.tokens_per_selection_block()))
        return aligned or None

    # --- P1 audit: core mean-K scorer vs the legacy Python index ------------------------
    def _compare_core_scores(self, blocks, legacy):
        """Log one comparison record (env: KVMEM_SCORER_COMPARE=1). Never raises."""
        import os
        # OFF unless the v0.19 statistics are on: the audit exists to gate the P1 switch, and it must
        # never perturb the shipped 0.2.5 ranking. Gated on KVMEM_STATS (not on its own flag) because
        # vLLM forks the workers with a filtered environment, so a new variable may simply not
        # arrive here; this one is already known to survive it.
        if os.environ.get("KVMEM_STATS", "").strip().lower() not in ("core", "1", "on"):
            return
        if os.environ.get("KVMEM_SCORER_COMPARE", "1") != "1":
            return
        # The counter is per ORCHESTRATOR and an orchestrator can serve several requests (parked
        # sessions are reused), so a low cap silently ended the audit while everything else looked
        # healthy. Keep a high cap and record every refusal reason instead of returning blind.
        if getattr(self, "_compare_done", 0) >= 64:
            _probe_write({"kind": "scorer_compare_skipped", "reason": "cap",
                          "done": int(getattr(self, "_compare_done", 0))})
            return
        stats = getattr(self, "statistics", None)
        if stats is None:
            _probe_write({"kind": "scorer_compare_skipped", "reason": "no statistics"})
            return
        try:
            from vllm import kvmem_bridge as bridge
            from kvmem_recall import KvmemRecall, compare_scores
            if not getattr(bridge, "ARMED", False):
                _probe_write({"kind": "scorer_compare_skipped", "reason": "bridge not armed"})
                return
            layer_rows = {}
            rows = 0
            for layer, (q, _pos) in bridge.read_all().items():
                if q is None or q.numel() == 0:
                    continue
                q = q.detach().to("cpu", non_blocking=False).float()
                layer_rows[int(layer)] = q
                rows = max(rows, int(q.shape[0]))
            if rows <= 0:
                _probe_write({"kind": "scorer_compare_skipped", "reason": "no query rows",
                              "layers": len(layer_rows)})
                return
            kv_heads = int(getattr(stats, "kv_heads", 0) or 0)
            head_dim = int(getattr(stats, "head_dim", 0) or 0)
            query_heads = int(layer_rows[next(iter(layer_rows))].shape[1]) if layer_rows else 0
            recall = getattr(self, "_recall", None)
            if recall is None:
                cfg = getattr(self.selector, "config_", None)
                recall = KvmemRecall(
                    stats,
                    block_tokens=int(getattr(stats, "block_tokens", 64) or 64),
                    sink_blocks=int(getattr(cfg, "sink_tokens", 0) or 0) // 64,
                    recent_blocks=int(getattr(cfg, "recent_tokens", 0) or 0) // 64,
                    budget_blocks=int(getattr(cfg, "budget_tokens", 0) or 0) // 64,
                    aggregation="mean")
                self._recall = recall
            count = int(getattr(stats, "frontier", 0)) // int(recall.block_tokens)
            core = recall.scores(layer_rows, rows, query_heads, kv_heads, head_dim, count)
            if core is None:
                rec = {"kind": "scorer_compare_error", "error": recall.last_error,
                       "rows": rows, "layers": len(layer_rows)}
            else:
                rec = {"kind": "scorer_compare", "rows": rows, "layers": len(layer_rows),
                       "query_heads": query_heads, "kv_heads": kv_heads,
                       "core_blocks": len(core), "legacy_blocks": len(legacy)}
                from kvmem_recall import align_scores
                index = getattr(self, "retrieval_index", None)
                lb = int(getattr(index, "block_tokens", 0) or 0)
                aligned = align_scores(core, int(getattr(stats, "block_tokens", 64)),
                                       len(legacy), lb)
                rec["core_block_tokens"] = int(getattr(stats, "block_tokens", 64))
                rec["legacy_block_tokens"] = lb
                rec.update(compare_scores(aligned, legacy,
                                          blocks=range(len(aligned)), top=10))
            self._compare_done = getattr(self, "_compare_done", 0) + 1
            _probe_write(rec)
        except Exception:
            try:
                import traceback as _tb
                if getattr(self, "log", None) is not None:
                    self.log({"kind": "scorer_compare_crash",
                              "error": _tb.format_exc()[-500:]})
            except Exception:
                pass

    # --- reselect ---------------------------------------------------------------------
    def reselect(self, mandatory_pages=(), slots=None, page_to_slot=None,
                 prefill_pressure=False, forced_pages=()):
        # The selector works on the token frontier, which can include a partially-filled last
        # selection block; only pages that already hold written tokens may be selected.
        core = None
        try:
            core = self.core_selection_scores()
        except Exception as exc:
            _probe_write({"kind": "core_scores_error", "error": f"{type(exc).__name__}: {exc}"})
        if core is not None:
            blocks = list(range(len(core)))
            raw = [float(v) for v in core]
            # Cross-check record only: the legacy index never feeds the selector any more.
            if self.retrieval_index is not None:
                try:
                    legacy_blocks = self.retrieval_index.covered_blocks()
                    legacy = [float(v) for v in self.retrieval_index.scores(legacy_blocks)] \
                        if legacy_blocks else []
                    if legacy:
                        self._compare_core_scores(legacy_blocks, legacy)
                except Exception:
                    pass
        elif self.retrieval_index is not None:
            blocks = self.retrieval_index.covered_blocks()
            if blocks:
                try:
                    scores = self.retrieval_index.scores(blocks)
                    raw = [float(s) for s in scores]
                    _probe_write({"kind": "scores_fallback_legacy",
                                  "reason": getattr(self, "_core_error", None)})
                    # The selector requires exactly one score per selection block
                    # (`store_.block_count()`), while the index only covers blocks that actually
                    # carry statistics. Pad the rest BELOW every real score so unscored blocks are
                    # never chosen by relevance (sink/recent still enter via the mandatory set).
                    #
                    # EVERY value handed to the core must be FINITE. Two separate ways to violate
                    # that, both of which made the core reject the whole table with
                    # "nonfinite LFRU retrieval score" and silently drop retrieval for the request
                    # (the selection then degenerated to sink+recent and dropped the question):
                    #   1. padding with `-inf`;
                    #   2. writing a score that is itself NaN/inf straight through.
                    # So: compute a finite floor, then SKIP any non-finite score instead of
                    # believing it.
                    # The grid and the block count belong to the SELECTOR, not to the legacy index:
                    # `set_block_scores` requires exactly `store_.block_count()` values, and a table
                    # of the wrong length is rejected with "one score per selection block is
                    # required" - which silently left the selector with no ranking at all
                    # (measured: selected blocks 0,1,2,... i.e. tie-broken by id, mid block dropped,
                    # generation degenerated into "req req req").
                    block_tokens = int(self.selector.tokens_per_selection_block())
                    try:
                        total_blocks = int(self.selector.selection_blocks())
                    except Exception:
                        total_blocks = 0
                    if total_blocks <= 0:
                        total_blocks = max(1, (int(self.total_tokens) + block_tokens - 1)
                                           // max(1, block_tokens))

                    def _finite(v):
                        return v == v and v not in (float("inf"), float("-inf"))

                    finite = [v for v in raw if _finite(v)]
                    # Cosine lives in [-1, 1]; keep the floor in a sane range as well, so an outlier
                    # score can never push an extreme value into the core's LRFU decay.
                    pad = min(max(min(finite) - 1.0, -2.0), -1e-6) if finite else -2.0
                    full = [pad] * total_blocks
                    nonfinite = 0
                    for block, value in zip(blocks, raw):
                        index = int(block)
                        if not (0 <= index < total_blocks):
                            continue
                        if not _finite(value):
                            nonfinite += 1
                            continue
                        full[index] = value
                    # If NOTHING usable came back, do not feed the core at all: a table made only of
                    # the floor is not a ranking, and an extreme floor (1e9) can overflow the core's
                    # LRFU decay the same way the original `-inf` did. Leaving the selector with no
                    # scores is the honest state, and `stats` records why.
                    if finite:
                        self.selector.set_block_scores(full)
                    # Diagnostic: the highest-scoring selection blocks, so a caller can check
                    # whether the block that carries the needed record was even ranked.
                    ranked = sorted(range(len(full)), key=lambda i: full[i], reverse=True)[:8]
                    self.last_top_blocks = [(int(b), round(float(full[b]), 4)) for b in ranked]
                    stats = {"blocks": len(blocks), "total_blocks": total_blocks,
                             "top": max(finite) if finite else None,
                             "nonfinite": nonfinite,
                             "score_min": min(finite) if finite else None,
                             "score_max": max(finite) if finite else None,
                             "diag": getattr(self.retrieval_index, "last_diag", None)}
                    store = getattr(self, "store", None)
                    if store is not None and hasattr(store, "disk_stats"):
                        stats["disk"] = store.disk_stats()
                    self.last_score_stats = stats
                except Exception as exc:
                    self.last_score_stats = {"error": str(exc)[:120]}
        if prefill_pressure:
            raw = self.selector.select_prefill_pressure_pages(list(mandatory_pages))
        else:
            raw = self.selector.select_pages(list(mandatory_pages))
        selected = [p for p in raw if p * self.page_tokens < max(self.total_tokens, 1)]
        if forced_pages:
            # Hard requirement from the caller (the backend's prompt, which vLLM keeps immutable):
            # the selector/planner may rank, but they may not drop these pages. The result MUST stay
            # ascending (sink..recent) - the store rejects an unsorted selection, and a wrong order
            # also shifts every page's compacted position.
            merged = set(int(p) for p in selected)
            for page in forced_pages:
                page = int(page)
                if page * self.page_tokens < max(self.total_tokens, 1):
                    merged.add(page)
            selected = sorted(merged)
        if self.planner is not None and selected:
            # KVMem decides the compact order and the visible length; we only filter pages that
            # hold no written tokens yet (a partially filled tail block).
            plan = self.planner.plan(selected, self.total_tokens,
                                     session_id=getattr(self, "session_id", 1),
                                     mandatory_pages=list(mandatory_pages))
            selected = [p for p in plan.pages if p * self.page_tokens < max(self.total_tokens, 1)]
            self.last_visible_tokens = plan.visible_tokens
            # Diagnostic: what the planner actually returned (the compact refuses when the visible
            # set does not cover the prompt, so these numbers explain a refusal directly).
            self.last_plan = {"visible": int(plan.visible_tokens),
                              "kept_blocks": len(plan.kept_blocks),
                              "sel_pages": len(selected),
                              "mandatory": len(mandatory_pages)}
        else:
            self.last_visible_tokens = len(selected) * self.page_tokens
        self.steps_since_reselect = 0
        self._tokens_at_reselect = self.total_tokens
        evicted = sorted(self.resident - set(selected))
        for page in evicted:
            # Canonical already lives in the store; dropping the device copy frees the slot.
            self.resident.discard(page)
            self.slot_of.pop(page, None)
            self.baked_delta.pop(page, None)

        # Pages that were just written but not committed yet have no canonical bytes, and there is
        # nothing to relocate for them (they are already resident). Drop them instead of failing the
        # whole reselect: raising here killed the step and left the plan half-applied.
        missing = [p for p in selected if not self.store.has(p)]
        if missing:
            # A selected page can be missing from the store when its write step was never committed
            # (the engine had not published its page id yet). Its bytes ARE on the device at
            # `slot_of`, so capture them now instead of dropping the page: dropping shrank the view
            # below the prompt and the compact was then refused (measured: 475 -> 332 pages).
            recovered = []
            for page in list(missing):
                slot = self.slot_of.get(page)
                if slot is None:
                    continue
                try:
                    layers = [kv[int(slot)] for kv in self.kvs]
                    self.store.put(int(page), layers)
                    # `baked_delta` is the rotation this device copy already carries (semantic
                    # frame) and is authoritative; the id difference is only the fallback for a
                    # page the runtime never tracked.
                    delta = int(self.baked_delta.get(
                        int(page), (int(slot) - int(page)) * self.page_tokens))
                    self.store_delta[int(page)] = delta
                    self.baked_delta[int(page)] = delta
                    recovered.append(page)
                except Exception:
                    continue
            if recovered:
                self.last_recovered = len(recovered)
                missing = [p for p in missing if p not in set(recovered)]
            self.last_missing = len(missing)
            if missing:
                selected = [p for p in selected if self.store.has(p)]
            if not selected:
                raise RuntimeError(f"selected pages without canonical bytes: {missing[:4]}")

        # Rebuild every selected page from the store with exactly ONE rotation hop
        # (delta_needed - delta_already_baked). The store itself is never rewritten, so repeated
        # reselects cannot compound the fp8 rounding.
        if page_to_slot is not None:
            # Physical page per selected logical page, supplied by the caller (the engine's own
            # allocation). Slots must be NON-ZERO and UNIQUE: a null/duplicate engine page would
            # make two logical pages claim the same device page, so one of them silently loses its
            # bytes (that is exactly how a needle check fails only for the lowest pages).
            seen = set()
            kept_pages, kept_slots = [], []
            for p in selected:
                slot = int(page_to_slot.get(p, 0) or 0)
                if slot <= 0 or slot in seen:
                    continue
                seen.add(slot)
                kept_pages.append(p)
                kept_slots.append(slot)
            if len(kept_pages) != len(selected):
                self.dropped_pages = len(selected) - len(kept_pages)
            selected = kept_pages
            self.slots = kept_slots
        elif slots is not None:
            # `slots` may be a callable(needed) so the caller reserves exactly as many device
            # slots as this selection needs instead of the whole budget (the old behaviour
            # reserved budget_pages()+1 per request, which nearly drained the pool).
            if callable(slots):
                slots = slots(len(selected))
            if len(slots) < len(selected):
                raise ValueError("not enough device slots for the selection")
            self.slots = list(slots[:len(selected)])
        elif len(self.slots) < len(selected):
            self.slots = list(range(len(selected)))

        if selected:
            import torch

            needed = [delta for _, (_, delta) in zip(selected, rotation_deltas(
                selected, self.page_tokens))]
            self._needed_delta = dict(zip(selected, needed))
            moves, rebuild = self.plan_moves(selected, needed)
            ordered, cyclic = self.order_moves(moves)
            # Pages that only change slot are copied on the DEVICE (HBM), not re-read over PCIe.
            page_of_src = {self.slot_of[p]: p for p in selected if p in self.slot_of}
            rebuild_pages = [p for p, _ in rebuild] + [page_of_src[s] for s, _ in cyclic
                                                       if s in page_of_src]
            rebuild_dst = [d for _, d in rebuild] + [self._needed_delta and self.slots[
                selected.index(page_of_src[s])] for s, _ in cyclic if s in page_of_src]
            # DEVICE MOVES FIRST, rebuild writes second: a rebuild destination can be the source
            # of another page's move, and writing the rebuilt bytes first clobbered them (the
            # store copy is read into its own tensor below, so the move order is unaffected).
            for layer, kv in enumerate(self.kvs):
                for src, dst in ordered:
                    kv[dst].copy_(kv[src])
            if rebuild_pages:
                dst_t = torch.tensor(rebuild_dst, dtype=torch.long, device=self.kvs[0].device)
                deltas = [self._needed_delta[p] - self.store_delta.get(p, 0)
                          for p in rebuild_pages]
                for layer, kv in enumerate(self.kvs):
                    stacked = torch.stack([self.store.get(page)[layer] for page in rebuild_pages])
                    stacked = stacked.to(kv.device, non_blocking=True)
                    rotated = rotate_page_batch(
                        stacked, torch.tensor(deltas, dtype=torch.float64),
                        self.inv_freq, self.rotary_dim, self.k_slice)
                    same = [i for i, d in enumerate(deltas) if d == 0]
                    if same:
                        rotated[same] = stacked[same]
                    kv[dst_t] = rotated
            for rank, page in enumerate(selected):
                self.resident.add(page)
                self.slot_of[page] = self.slots[rank]
                self.baked_delta[page] = needed[rank]
            self.last_move_stats = {"device_moves": len(ordered), "host_rebuild": len(rebuild_pages),
                                    "cyclic": len(cyclic), "selected": len(selected)}
        return self._finish_reselect(selected, evicted, prefill_pressure)

    def plan_moves(self, selected, needed):
        """Split the rebuild into device-to-device moves and host re-reads.

        A page that is still resident (bytes valid on the device) with the SAME baked rotation
        only needs to change slots. Copying it on the device costs HBM bandwidth (~600 GB/s)
        instead of a PCIe round trip (~12 GB/s), and it does not add an fp8 rounding step.
        Everything else is rebuilt from the host page store (one hop, never chained).
        """
        moves, rebuild = [], []
        for rank, page in enumerate(selected):
            dst = self.slots[rank]
            want = needed[rank]
            src = self.slot_of.get(page)
            if (page in self.resident and src is not None
                    and self.baked_delta.get(page) == want):
                if src != dst:
                    moves.append((int(src), int(dst)))
                continue
            rebuild.append((page, dst))
        return moves, rebuild

    @staticmethod
    def order_moves(moves):
        """Order device moves so a copy never clobbers a page that still has to move.

        Returns (safe_moves, cyclic). Only moves whose destination is not itself a pending
        source are safe; whatever is left forms cycles (rare, small) and is re-read from the
        host store by the caller instead of being shuffled in place.
        """
        pending = dict(moves)
        ordered = []
        progress = True
        while pending and progress:
            progress = False
            for src in list(pending):
                dst = pending[src]
                if dst in pending:          # its bytes are still needed somewhere
                    continue
                ordered.append((src, dst))
                del pending[src]
                progress = True
        return ordered, list(pending.items())

    def _finish_reselect(self, selected, evicted, prefill_pressure):
        self.resident &= set(selected)
        self.last_view = build_sparse_view(selected, [], self.page_tokens, 0)
        return {"selected": len(selected), "evicted": len(evicted),
                "resident": len(self.resident), "prefill_pressure": bool(prefill_pressure)}

    def tick(self):
        self.steps_since_reselect += 1

    # --- helpers ----------------------------------------------------------------------
    def _slot_for(self, selected, page):
        return self.slots[selected.index(page)]

    def _rotate_page_bytes(self, canonical, delta):
        """Rotate one layer's page bytes (or a list of layers, returning a list)."""
        if isinstance(canonical, (list, tuple)):
            return [self._rotate_page_bytes(page, delta) for page in canonical]
        out = canonical.clone()
        k = _dequant(out[..., self.k_slice])
        out[..., self.k_slice] = _quantize(delta_rotate(k, delta, self.inv_freq, self.rotary_dim))
        return out

    # --- cross-request resume ----------------------------------------------------------
    def canonicalize_store(self):
        """Re-bake every stored page into the page's own LOGICAL frame (`store_delta == 0`).

        `store_delta[p]` is the rotation the stored bytes carry relative to page `p`'s logical
        start, so rotating by `-store_delta[p]` recovers the frame the engine wrote at true
        positions. That is now a well-defined operation (it was not, while the delta was recorded
        in the engine's page-id space), but it still costs one dequant/rotate/requant hop per page,
        so the resume path does NOT use it: `restore_page_rank` hops straight from the stored bytes
        to the new rank, exactly like a compaction does.
        """
        moved = 0
        for page in list(self.store.pages):
            delta = int(self.store_delta.get(int(page), 0))
            if delta == 0:
                continue
            layers = self.store.get(int(page))
            fixed = self._rotate_page_bytes(layers, -float(delta))
            self.store.put(int(page), fixed)
            self.store_delta[int(page)] = 0
            # The device copy now carries the old frame; drop residency so it can never be read.
            self.resident.discard(int(page))
            self.slot_of.pop(int(page), None)
            self.baked_delta.pop(int(page), None)
            moved += 1
        return moved

    def usable_pages(self, total_tokens=None):
        """Every logical page this session still holds canonical bytes for, ascending.

        A resume may only bring back pages that actually exist in the store: a selected page with no
        bytes would install a hole in the view and the model would read garbage for it.
        """
        limit = None if total_tokens is None else max(1, int(total_tokens))
        pages = sorted(int(p) for p in self.store.pages)
        if limit is None:
            return pages
        return [p for p in pages if p * self.page_tokens < limit]

    def restore_page_rank(self, page, dst_page, rank):
        """Load a stored page into tensor page `dst_page` at compacted position `rank`.

        This is the resume path's data plane. The logical page `p` is written back at a rank that
        differs from the rank it was parked at, and the engine page id it lands in is unrelated to
        either, so the rotation is derived from `(rank - p) * page_tokens` -- never from the id
        difference (see `_baked_delta`). One hop from the stored bytes, like a compaction.
        """
        return self.restore_page(page, slot=dst_page, rank=rank)

    def needle(self, page):
        """The device copy of `page` must equal its stored bytes rotated by the remaining delta."""
        slot = self.slot_of.get(int(page))
        if slot is None:
            return None
        step = self.baked_delta[int(page)] - self.store_delta.get(int(page), 0)
        stored = self.store.get(page)              # CPU bytes
        for layer, kv in enumerate(self.kvs):
            dev = stored[layer].to(kv.device, non_blocking=False)
            want = dev if step == 0 else self._rotate_page_bytes(dev, step)
            if not torch.equal(kv[slot], want):
                return False
        return True
