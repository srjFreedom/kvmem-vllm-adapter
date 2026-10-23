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

    def write_page(self, page, bytes_as_written, slot):
        """Record a page whose bytes are `bytes_as_written`, baked at `(slot - page)*tokens`.

        The store keeps those bytes verbatim; the device copy is one hop from them, so a page can
        be re-derived any number of times without compounding fp8 rounding.
        """
        delta = (int(slot) - int(page)) * self.page_tokens
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
        pass over `page_tokens x head_dim` per layer. `store_delta[page]` is exactly the rotation
        baked into them, so un-rotating by its negative yields the true pre-RoPE frame.
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

    def commit_written_page(self, page, slot, rank=None):
        """Record a page the ENGINE just wrote into physical `slot` at compacted `rank`.

        The rotation baked into those bytes is the POSITION shift, not the physical page id: the
        engine writes at the compacted positions the runtime installs, so a page placed at rank r
        carries `(r - page) * page_tokens` of RoPE. Using `(slot - page)` instead (the old form) was
        only correct while `slot == rank`; with the runtime's slot pool (ids allocated from the top
        of the pool) it recorded an arbitrary delta, `needle()` compared against that same arbitrary
        value so it always passed, and a later restore rotated the page to the wrong phase - the
        measured "checker green, answers rot". `rank=None` means identity (the engine wrote at true
        positions before any compaction), which is the pre-compaction path.
        """
        if rank is None:
            rank = int(page)
        delta = (int(rank) - int(page)) * self.page_tokens
        page_bytes = [kv[int(slot)].clone() for kv in self.kvs]
        self.store.put(page, page_bytes)
        self.store_delta[int(page)] = delta
        self._feed_retrieval(page, page_bytes)
        self.resident.add(int(page))
        self.slot_of[int(page)] = int(slot)
        self.baked_delta[int(page)] = delta

    def restore_page(self, page, slot=None, channel=None, request_id=None, rank=None):
        """Bring an evicted page back; without a slot, ask EngineCore for a free block."""
        if slot is None:
            if channel is None:
                raise RuntimeError("restore_page needs a slot or a block channel")
            slot = channel.request(request_id or "restore", 1)[0]
        """Bring an evicted page back from the host store into `slot`.

        The store keeps the canonical bytes forever, so a restore is exactly ONE rotation hop by
        `needed - store_delta` (never a rotation of a rotation). The control plane - asking the
        scheduler for a free block - is the caller's job; this is the data plane.

        `rank` is the compacted RANK the page is being restored for - that is what decides the RoPE
        phase; the physical slot id does not. Callers that pass a slot equal to the rank (the unit
        tests) can omit it.
        """
        page, slot = int(page), int(slot)
        if not self.store.has(page):
            raise RuntimeError(f"cannot restore page {page}: no canonical bytes")
        needed = (int(rank) - page) * self.page_tokens if rank is not None \
            else (slot - page) * self.page_tokens
        step = needed - self.store_delta.get(page, 0)
        for layer, kv in enumerate(self.kvs):
            dev = self.store.get(page)[layer].to(kv.device, non_blocking=True)
            kv[slot].copy_(dev if step == 0 else self._rotate_page_bytes(dev, step))
        self.resident.add(page)
        self.slot_of[page] = slot
        self.baked_delta[page] = needed
        return needed

    # --- reselect ---------------------------------------------------------------------
    def reselect(self, mandatory_pages=(), slots=None, page_to_slot=None,
                 prefill_pressure=False, forced_pages=()):
        # The selector works on the token frontier, which can include a partially-filled last
        # selection block; only pages that already hold written tokens may be selected.
        if self.retrieval_index is not None:
            blocks = self.retrieval_index.covered_blocks()
            if blocks:
                try:
                    scores = self.retrieval_index.scores(blocks)
                    # The selector requires exactly one score per selection block
                    # (`store_.block_count()`), while the index only covers blocks that actually
                    # carry statistics. Pad to the full table with -inf so unscored blocks are never
                    # chosen by relevance (sink/recent blocks still enter via the mandatory set).
                    block_tokens = int(getattr(self.retrieval_index, "block_tokens", 128) or 128)
                    total_blocks = max(1, (int(self.total_tokens) + block_tokens - 1) // block_tokens)
                    full = [float("-inf")] * total_blocks
                    for block, value in zip(blocks, scores):
                        index = int(block)
                        if 0 <= index < total_blocks:
                            full[index] = float(value)
                    self.selector.set_block_scores(full)
                    # Diagnostic: the highest-scoring selection blocks, so a caller can check
                    # whether the block that carries the needed record was even ranked.
                    ranked = sorted(range(len(full)), key=lambda i: full[i], reverse=True)[:8]
                    self.last_top_blocks = [(int(b), round(float(full[b]), 4)) for b in ranked]
                    stats = {"blocks": len(blocks), "total_blocks": total_blocks,
                             "top": max(scores) if scores else None}
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
                    delta = (int(slot) - int(page)) * self.page_tokens
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
