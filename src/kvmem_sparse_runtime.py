# The live wiring layer (KVMem "A mode" on vLLM): connects the sparse orchestrator to a model
# runner's three tensors and to the two hook points a step needs.
#
#   before_step(req, n_query): the engine is about to build attention metadata. Install the
#       request's sparse view (block-table row + seq_len) and the COMPACTED positions of its
#       query tokens, so the engine reads |sel| pages and writes new K at compacted positions.
#   after_step(req, n_new): execute_model returned. Commit the pages the engine just wrote back
#       into the canonical store, advance the clock, and reselect when due.
#
# Per-request state is explicit (row, seq_len, tokens, next_page) instead of being re-derived,
# because every one of those is installed into engine tensors and a mismatch silently corrupts
# attention.
import os
from kvmem_sparse_orchestrator import HostPageStore, SparseOrchestrator


class RunnerView:
    """What the wiring needs from a model runner (real or fake)."""

    def paged_kv(self):
        """[pages, heads, page_tokens, content] uint8 tensor for this rank."""
        raise NotImplementedError

    def block_table_rows(self):
        """List of per-group block-table tensors (rows x pages)."""
        raise NotImplementedError

    def seq_lens(self):
        raise NotImplementedError

    def positions(self):
        raise NotImplementedError

    def num_blocks(self):
        raise NotImplementedError


def pool_prefill_high_watermark(ceiling_tokens, page_tokens):
    """First pressure boundary: physical KV ceiling minus one chunk and allocator slack."""
    return max(64, max(0, int(ceiling_tokens) - 4096 - 2048)
               // max(1, int(page_tokens)))


def prefill_pressure_due(tokens, resident_pages, high_pages, last_reselect, reserve):
    """R is the minimum prefill advance between pressure compactions once one has succeeded."""
    return (int(resident_pages) > int(high_pages)
            and (int(last_reselect) <= 0 or
                 int(tokens) - int(last_reselect) >= int(reserve)))


class _SkipSeqFromRow(Exception):
    """Internal: this experiment is switched off."""


class SparseViewCapacityError(RuntimeError):
    """The compacted block-table row cannot address the scheduled attention tokens."""


class _ReqState:
    # row:      PHYSICAL slot ids written into the block table (what the kernel reads).
    # logical:  logical page id (true token page) occupying each row position, so a written
    #           token can be attributed to the history block it belongs to.
    __slots__ = ("row", "logical", "seq_len", "tokens", "next_page", "steps", "spare",
                 "row_index", "prefill_done", "query_reselected", "resume", "resumed",
                 "replay_snap", "replay_n", "replay_at", "replay_at_resident", "rebuilt",
                 "compacted_in_prefill", "applied_view_coord", "off_pages", "window_src")

    def __init__(self):
        self.row = ()
        self.logical = ()
        # logical page = row position + off_pages (set by the last APPLIED boundary compaction).
        self.off_pages = 0
        self.window_src = {}
        self.seq_len = 0
        self.tokens = 0
        self.next_page = 0
        self.steps = 0
        self.spare = None
        self.row_index = 0
        self.prefill_done = False
        # One retrieval-driven reselect right when the prompt is complete (the query is then in
        # the captured window); without it a prompt compacted during prefill never re-reads its
        # middle on a short generation.
        self.query_reselected = False
        # Cross-request resume: `resume` holds (parked entry, hit_tokens, visible_tokens) until the
        # first step installs the restored view; `resumed` marks that it happened.
        self.resume = None
        self.resumed = False
        # A prompt-end reconstruction is only meaningful when the prefill actually compacted the
        # working set. Triggering it from `prompt > B` alone rebuilt a 50K prompt that fit the
        # physical pool untouched, which showed up as a second ~36K prefill after the first one.
        self.compacted_in_prefill = False
        self.applied_view_coord = False
        # QueryReplay: recurrent state captured just before the LAST prefill chunk, plus that
        # chunk's token count. Replaying applies those tokens to the GDN state a second time unless
        # the state is restored to this boundary first (a hybrid model cannot "replay" for free).
        self.rebuilt = False
        self.replay_snap = None
        self.replay_n = 0
        self.replay_at = 0
        self.replay_at_resident = 0


class VllmSparseRuntime:
    def __init__(self, view: RunnerView, selector, page_tokens, head_dim, rotary_dim, base,
                 k_slice=None, reselect_interval=64, log=None, first_page=0,
                 select_block_tokens=128, budget_tokens=32768, gen_reserve=16384, recent_tokens=1024,
                 host_mib=16384, use_kvmem_plan=False, needle_trace=0, use_kvmem_host=True,
                 sessions=4, pages_per_block=1, ctx_tokens=0, no_prefill_compact=False,
                 sink_tokens=0):
        self.view = view
        self.selector = selector
        self.page_tokens = int(page_tokens)
        self.head_dim = int(head_dim)
        self.rotary_dim = int(rotary_dim)
        self.base = float(base)
        self.k_slice = k_slice
        self.reselect_interval = int(reselect_interval)
        # The kvmem contract parameters; the planner needs them verbatim.
        self.select_block_tokens = int(select_block_tokens)
        self.budget_tokens = int(budget_tokens)
        self.gen_reserve = int(gen_reserve)
        self.recent_tokens = int(recent_tokens)
        self.sink_tokens = int(sink_tokens or 0)
        self.host_mib = int(host_mib)
        self.use_kvmem_plan = bool(use_kvmem_plan)
        self.needle_trace = int(needle_trace)
        self.use_kvmem_host = bool(use_kvmem_host)
        self.sessions = int(sessions)
        # Engine allocation quantum in PAGES (attention block / kernel page). With a hybrid model
        # under mamba align this is e.g. 1616/16 = 101: the engine hands out 1616-token blocks and
        # the block table row is written in whole blocks, so the resident row must be padded to a
        # multiple of that quantum or the engine's next append lands in the middle of the row.
        self.pages_per_block = max(1, int(pages_per_block or 1))
        # kvmem's logical ceiling (entry CTX); only used for capacity diagnostics here.
        self.ctx_tokens = int(ctx_tokens or 0)
        # Bisect switch: skip compaction during prefill entirely (the resident book then grows to
        # the prompt length and only the R cadence compacts during decode). Isolates a prefill-time
        # compaction race from the selection policy.
        self.no_prefill_compact = bool(no_prefill_compact)
        self.full_needle = int(os.environ.get("KVMEM_FULL_NEEDLE", "0") or 0)
        self.budget_pool = None
        # Content-based retrieval (mean-K over pre-RoPE keys). Off by default: it costs a capture
        # per layer near each reselect.
        self.retrieval_enabled = False
        self._retrieval = {}
        # Bisect switches: when a part is False the runtime still tracks, selects, spills and
        # verifies, but never writes that piece into the engine (isolates hook damage).
        self.install = True
        self.install_row = True
        self.install_seq = True
        self.install_pos = True
        self.log = log or (lambda record: None)
        self._adv_logged = set()
        # `advance_tokens` (execute_model entry) is the single owner of token accounting; the
        # post-block must not add the same tokens again.
        self._advance_owns_accounting = True
        self._prev_n = {}
        self._decode_probe = {}
        self._last_report_tokens = {}
        self._install_row_now = {}
        self._slot_span = {}
        self._slot_span_prev = {}
        self._pending_frontier = {}
        self._pending_new = {}
        # Tokens scheduled for the step that is executing right now (0 outside a step). A
        # compaction that fires mid-prefill must commit their pages too, from the row.
        self._inflight = {}
        # Engine-reported num_computed_tokens per request: authoritative for ACCEPTED tokens.
        # Counting `num_scheduled_tokens` drifts under MTP (draft tokens that get rejected are
        # counted but never appended), which fires the R cadence early.
        self._engine_computed = {}
        # Engine-reported SERVED count (num_logical_tokens). Differs from `_engine_computed` once a
        # compaction rewound the resident book: `st.tokens` tracks this, `st.seq_len` the resident.
        self._engine_logical = {}
        # Engine-reported PROMPT length per request (immutable for the request's life). The
        # entry policy is supposed to keep it <= B; when it does not, compaction cannot shorten
        # the book below the prompt, so the runtime must not churn slots trying.
        self._prompt_len = {}
        self._over_budget_logged = set()
        # Cross-request prefix reuse (parked sessions keyed by native block hashes). Off by
        # default: the store hands the session's host share back only when it is evicted, so this
        # holds real RAM for every parked conversation.
        self.session_reuse = False
        self.session_store = None
        self.kv_block_size = 0
        self._prompt_ids = {}
        self._session_ids = {}
        self._session_checkpoints = {}  # req -> (frontier, exact tokens, StateImage, sealed statistics)
        self._preparked_requests = set()
        self.session_checkpoint_chunk_tokens = 0
        self.session_checkpoint_speculative_tokens = 0
        self.session_checkpoint_executed_frontier = None
        # req -> (hit_tokens, visible_tokens) decided by EngineCore for a resumed session.
        self._resume = {}
        # Optional hooks supplied by the plugin for the non-attention (GDN/Mamba) state, which is
        # NOT in the paged store: capture(req) -> snapshot, restore(req, snapshot) -> bool.
        self.capture_state = None
        self.restore_state = None
        self.reset_state = None          # zero the recurrent state (reconstruction)
        self.set_block_cursor = None     # reset the worker's block-table append cursor
        # Off by default: re-ranking at the prompt boundary replaces the working set the recurrent
        # state was built against, so it is only sound together with a reconstruction - and the
        # reconstruction currently stalls the request ("no sampled token").
        self.reselect_at_prompt_end = os.environ.get("KVMEM_PROMPT_END_RESELECT", "0") == "1"
        self.compact_rebuild = False     # rewrite the token book on compaction (reconstruction)
        # True when the model has non-attention (GDN/Mamba) KV groups. Their recurrent state is NOT
        # in the paged store, so a resume is only safe when a state snapshot travelled with the
        # parked session (otherwise the resumed prefill would read a stale recurrent state -- the
        # documented "draft acceptance collapses / generation repeats" failure).
        self.has_recurrent = False
        # callable(payload) that publishes a park to EngineCore (block channel), or None.
        self.session_park = None
        self._orch = {}
        self._state = {}
        # row_index -> the compacted length the ATTENTION kernel must read for that row.
        # The Mamba/GDN groups must keep the true seq_lens (their state indexing depends on it),
        # so this override is applied only to the full-attention metadata builder.
        self._seq_override = {}
        # Device slots are a shared pool; the runtime hands each request a disjoint set and
        # returns them when the request reselects.
        # logical page (token page index) -> the ENGINE's physical page id for it. The engine
        # owns allocation; we only reorder its pages inside the request's own row.
        self._page_map = {}         # req -> {logical: engine_page}
        # Rows the runtime has already absorbed: after a compaction the row is OUR packed layout,
        # so only entries the engine appended (index >= the cursor) are new.
        self._absorb_upto = {}
        # Logical pages written since the last reselect. Committing them per step forced a
        # synchronous D2H of 17 layers every step (~272 KiB) which dominated decode latency;
        # they stay on the device until the next reselect reads them in one batch.
        self._dirty_pages = {}      # req -> set(logical)
        self._owned_slots = {}      # req -> set of slot ids (physical pages we reordered)
        self._slot_owner = {}       # slot id -> req
        self._next_slot_base = 0
        # vLLM reserves a NULL block for padding. In `mamba_cache_mode=align` that block spans a
        # whole ALIGNED unit (block_size / page_tokens pages), so the first usable page id is
        # that offset — NOT 0. Allocating from 0 shifts every row by that amount and silently
        # makes attention read reserved pages (verified: ids start at 101 on this model).
        self.first_page = int(first_page)
        # Spill before the pool fills: resident pages per request during prefill.
        # The watermark is capped by the POOL's effective ceiling when the plugin publishes it:
        # B-driven thresholds left a dead zone where a large B never compacted before the pool ran
        # out (measured: stall at resident 86,016 with 0 compactions for B=90K/100K/110K).
        self.pool_ceiling_tokens = 0
        self.prefill_high_watermark = self.selector.budget_pages() + 64
        if int(getattr(self, "pool_ceiling_tokens", 0) or 0) > 0:
            import math as _m
            self.prefill_high_watermark = min(
                self.prefill_high_watermark,
                max(64, int(self.pool_ceiling_tokens) // max(1, self.page_tokens)))
        # --- mean-K statistics owned by kvmem's RawKvStore (v0.19 recall design) --------------
        # KVMEM_STATS=core fills one kvmem statistics store per request from the engine's
        # device-side 64-token bucket sums (kvmem_bridge.capture_k). The Python mean-key index
        # keeps running alongside it until the scoring path moves over (P1), so the two can be
        # compared block by block instead of trusting a rewrite.
        self.stats_core = os.environ.get("KVMEM_STATS", "").strip().lower() in ("core", "1", "on")
        self.stats_decode = os.environ.get("KVMEM_STATS_DECODE", "0").strip() == "1"
        self._statistics = {}       # req -> KvmemStatistics
        self._stats_next = {}       # req -> first TRUE position not yet written
        self._stats_written = {}    # req -> bucket writes
        self._stats_skipped = {}    # req -> {reason: count} so a refusal cannot hide behind another
        self._compare_done = {}     # req -> the core-vs-legacy score audit already ran
        # Set by the plugin from the attention module: per-rank KV heads and the number of
        # full-attention layers (= the statistics store's n_layer).
        self.kv_heads = 0
        self.statistics_layers = 0

    def is_rebuilt(self, req):
        return req in getattr(self, "_rebuilt_req", ())

    def accept_reconstruction(self, req):
        """Retire the old position space on EVERY TP rank before fresh worker admission.

        The original host archive remains owned until request cleanup. Recomputed KV uses the
        engine's native dense path; old logical-page statistics must never be overwritten by
        compacted position IDs.

        Idempotent: `_rebuild_select` marks the old request retired the moment Core queues the
        fresh admission, and the rebuilt request arrives at `execute_model` with
        `kvmem_rebuilt` and calls this again. When the rebuilt generation is ALREADY resident
        (row rebuilt by the fresh prefill), the second call must not wipe it.
        """
        st = self._state.get(req)
        if req in getattr(self, "_rebuilt_req", ()) and (st is None or not st.row):
            return
        self._rebuilt_req = getattr(self, "_rebuilt_req", set())
        self._rebuilt_req.add(req)
        if st is not None:
            self._seq_override.pop(st.row_index, None)
            self._release_slots(req, set())
            st.row = st.logical = ()
            st.tokens = st.seq_len = 0
            st.spare = None
            st.rebuilt = True
            st.query_reselected = True
        # stage-2 rebuilt generation: from here the request runs in VIEW/resident coordinates
        # (the rebuilt prompt is the whole resident stream; the old logical stream is retired
        # to the host store). `off_pages=0` is the transform the commit/store lookups use.
        if st is not None:
            st.off_pages = 0
            st.applied_view_coord = True
        for mapping in (self._prev_n, self._pending_new, self._pending_frontier,
                        self._inflight, self._engine_computed, self._engine_logical,
                        self._page_map, self._absorb_upto):
            mapping.pop(req, None)
        # The rebuilt generation's LOGICAL page ids restart at 0, but the engine writes them
        # into NEW physical pages. Seed the page-id map from the request's CURRENT live row so
        # the very first commit after `accept` already addresses the right pages (mirroring
        # only starts at the next `sync_row_from_engine`).
        try:
            row0 = [int(p) for p in (st.row or ())] if st is not None else []
            if not row0:
                ridx = int(getattr(st, "row_index", 0) or 0) if st is not None else 0
                tbl = self.view.block_table_rows()[0][ridx].tolist()
                row0 = [int(p) for p in tbl if int(p) > 0]
            pm = self._page_map.setdefault(req, {})
            for i, p in enumerate(row0):
                pm[i] = int(p)
            self._absorb_upto[req] = len(row0)
        except Exception:
            pass
        self.log({"kind": "reconstruction_started", "req": req,
                  "rank": int(getattr(self, "tp_rank", 0))})

    def needs_reconstruction(self, req):
        """True only when THIS prefill had to compact, so the real prompt no longer fits resident."""
        try:
            st = self.state(req)
        except Exception:
            return False
        return bool(self.compact_rebuild and st.compacted_in_prefill and not st.rebuilt
                    and not self.is_rebuilt(req))

    def finish_prompt_for_reconstruction(self, req, n_new, logical_end, resident_end):
        """Commit the final forward and queue its selected tokens before any sample escapes."""
        if not self.compact_rebuild or self.is_rebuilt(req):
            return False
        st = self.state(req)
        if st.rebuilt:
            return False
        runner = getattr(self.view, "runner", None)
        if runner is not None:
            batch = runner.execute_model_state.input_batch
            resident_end = int(batch.seq_lens[batch.req_ids.index(req)].item())
        self._engine_logical[req] = int(logical_end)
        self._engine_computed[req] = int(resident_end)
        self._pending_new[req] = int(n_new)
        self._finishing_reconstruction = True
        try:
            self.after_step(req, int(n_new), prefill=True)
        finally:
            self._finishing_reconstruction = False
        self._prev_n[req] = 0
        self._inflight[req] = 0
        st.query_reselected = True
        self._reselect(req, probe=True, rebuild=True)
        if not st.rebuilt:
            raise RuntimeError("prompt reconstruction did not queue a fresh admission")
        return True

    # --- state -------------------------------------------------------------------------
    def slot_pages_for(self, req):
        """Physical page ids the LAST forward wrote, per query token, from the attention
        kernels' own slot mapping (the ONLY address that is by construction where the bytes
        landed - the block-table tensor this plugin reads is a staging copy that the engine
        rewrites from its own bookkeeping each step). Returns {qpos: page} over the span the
        runtime recorded at the previous step, or None."""
        try:
            span = self._slot_span_prev.get(req)
            if not span:
                return None
            sm = self.view.runner.block_tables.slot_mappings
            g = self.view.attn_groups[0]
            row = sm[g]
            p0, p1 = int(span[0]), min(int(span[1]), int(row.shape[0]))
            if p1 <= p0:
                return None
            vals = row[p0:p1].tolist()          # ONE device sync for the whole span
            pt = self.page_tokens
            out = {p0 + k: v // pt for k, v in enumerate(vals) if v >= 0}
            return out or None
        except Exception:
            return None

    def _absorb_engine_row(self, req):
        """Record which engine page holds each logical page (index = logical page)."""
        st = self._state[req]
        try:
            table = self.view.block_table_rows()[0]
            row = table[st.row_index]
            values = row.tolist()
        except Exception:
            return
        page_map = self._page_map.setdefault(req, {})
        start = int(self._absorb_upto.get(req, 0))
        upto = start
        for logical in range(max(0, start), len(values)):
            page = int(values[logical])
            if page > 0:
                page_map[logical] = page
                upto = logical + 1
        # The row is a fixed-capacity tensor: the cursor follows the WRITTEN prefix, not its
        # capacity, so the engine's next appends are still picked up.
        self._absorb_upto[req] = max(start, upto)

    def _ensure(self, req):
        if req not in self._orch:
            adopted = self._adopt_session(req)
            if adopted is not None:
                # Continue a parked session: its orchestrator already owns the whole context in the
                # host tier, so nothing is re-prefilled and no new host share is opened.
                entry, hit_tokens, visible_tokens = adopted
                orch = entry.orchestrator
                # The parked device pages belong to a FINISHED request and have been returned to
                # the pool, so their residency bookkeeping is stale: every page is rebuilt from the
                # store on demand instead (the store is the only authority).
                orch.resident.clear()
                orch.slot_of.clear()
                orch.baked_delta.clear()
                orch.slots = []
                orch.steps_since_reselect = 0
                self._orch[req] = orch
                self._owned_slots[req] = set()
                st = _ReqState()
                st.resume = (entry, hit_tokens, visible_tokens)
                self._state[req] = st
                # A resumed request needs its OWN retrieval index. The parked orchestrator's index
                # was released with the previous request and carries no query for THIS turn, and
                # `self._retrieval[req]` was never populated on this early return - so
                # `runtime.retrieval_index(req)` came back None, `_active_index()` returned None,
                # `harvest_from_bridge` bailed out before reading the bridge (measured:
                # `layers_fed=0`, `capture.calls=0`), `scores()` raised "no query captured" and the
                # compacted selection degenerated to sink+recent - dropping the question itself,
                # which reads as repeated output.
                if self.retrieval_enabled:
                    index = getattr(orch, "retrieval_index", None)
                    if index is None:
                        # A cached KV prefix without its sealed mean-K identity cannot answer the
                        # next query safely. _adopt_session checks this before consuming the entry.
                        self.log({"kind": "resume_miss", "req": req,
                                  "reason": "sealed retrieval statistics missing"})
                        return
                    index.reset_query()
                    self._retrieval[req] = index
                    orch.retrieval_index = index
                    if orch.planner is not None:
                        orch.planner.pre_rope_statistics = True
                if self.stats_core:
                    saved_stats = getattr(orch, "statistics", None)
                    if saved_stats is None or int(saved_stats.frontier) != int(hit_tokens):
                        raise RuntimeError("resumed request lost exact-prefix mean-K statistics")
                    self._statistics[req] = saved_stats
                    self._stats_next[req] = int(hit_tokens)
                    self._stats_written[req] = 0
                try:
                    self.attach_statistics(req)
                except Exception:
                    pass
                return
            kv_tensors = self.view.paged_kv()
            session = None
            store = None
            # KVMem's host tier owns the spilled payload and enforces H. Falls back to the plain
            # host store when the native extension is unavailable.
            if getattr(self, "use_kvmem_host", True):
                try:
                    from kvmem_sparse_orchestrator import KvmemHostStore
                    # H is shared across live and parked sessions. Reusing len(_orch)+1 after a
                    # park aliased an idle grant and silently handed two stores the full H budget.
                    session, share = self._open_host_share()
                    # Disk tier (D budget): pages evicted from the H share stage out here instead of
                    # being dropped, matching what the llama/ninfer backends do with nvme_kv_tier.
                    store = KvmemHostStore(capacity_bytes=share,
                                           page=self.page_tokens, layers=len(kv_tensors),
                                           disk_bytes=int(getattr(self, "disk_mib", 0) or 0) * 1024 * 1024,
                                           disk_dir=getattr(self, "disk_path", None) or None)
                    # Make the host budget observable: with the resident model a request can spill
                    # (CTX - B) tokens, so the share must be sized for that (it is per SESSION).
                    try:
                        per_page = int(sum(int(t[0].numel()) for t in kv_tensors))
                        share_tokens = (int(share) // max(1, per_page)) * self.page_tokens
                        self.log({"kind": "host_share", "req": req,
                                  "session": session,
                                  "share_mib": int(share) >> 20,
                                  "share_tokens": share_tokens,
                                  "spill_tokens_at_ctx": int(getattr(self, "ctx_tokens", 0) or 0)
                                  - int(self.budget_tokens)})
                    except Exception:
                        pass
                except Exception as exc:
                    from kvmem_session_budget import BudgetError
                    if isinstance(exc, BudgetError):
                        raise
                    if session is not None and getattr(self, "budget_pool", None) is not None:
                        self.budget_pool.close(session)
                    store = None
            orch = SparseOrchestrator(
                self.selector, kv_tensors, self.page_tokens, self.head_dim,
                self.rotary_dim, self.base, k_slice=self.k_slice,
                store=store or HostPageStore(), reselect_interval=self.reselect_interval)
            # In-request compaction is opt-in for this backend: vLLM tracks sequence length
            # monotonically, so truncation must be requested explicitly.
            orch.inrequest_compaction = bool(getattr(self, "inrequest_compaction", False))
            # Cadence: KVMem's R is the authority - the pool (B+R) fills exactly when the output
            # since the last reselect reaches R. Drop the adapter's own 4096-token timer in that
            # mode, and expose R here so the trigger also holds when the native planner is absent.
            orch.gen_reserve = int(self.gen_reserve)
            if orch.inrequest_compaction:
                orch.reselect_token_budget = 0
            # KVMem is the authority for the attention view. The planner is optional so the same
            # runtime still works when the native extension is unavailable.
            if getattr(self, "use_kvmem_plan", False):
                try:
                    from kvmem_plan_view import KvmemPlanner
                    layers = len(self.view.paged_kv())
                    orch.planner = KvmemPlanner(
                        page_tokens=self.page_tokens, block_tokens=self.select_block_tokens,
                        budget_tokens=self.budget_tokens, gen_reserve=self.gen_reserve,
                        host_mib=self.host_mib, record_bytes=None, layers=layers,
                        device_tokens=(self.view.num_blocks() * self.page_tokens
                                       if hasattr(self.view, "num_blocks") else None))
                except Exception:
                    orch.planner = None
                    try:
                        import traceback as _tb
                        self.log({"kind": "planner_error", "req": req,
                                  "error": _tb.format_exc()[-600:]})
                    except Exception:
                        pass
            if self.retrieval_enabled:
                from kvmem_retrieval_index import KvmemRetrievalIndex
                index = KvmemRetrievalIndex(self.select_block_tokens, self.page_tokens,
                                            self.head_dim, len(kv_tensors))
                # KVMem query policy: the tail of the last-user span (512 max), falling back to the
                # last 64 query tokens. Averaging a whole prefill chunk diluted the question.
                index.query_tail = max(1, int(os.environ.get("KVMEM_QUERY_TAIL", "512") or 512))
                self._retrieval[req] = index
                orch.retrieval_index = index
                if orch.planner is not None:
                    orch.planner.pre_rope_statistics = True
            try:
                self.attach_statistics(req)
            except Exception:
                pass
            orch.session_id = session      # identity used by the shared H pool on release
            self._orch[req] = orch
            self._owned_slots[req] = set()
            self._state[req] = _ReqState()

    # --- cross-request prefix reuse (parked sessions) -----------------------------------
    def enable_session_reuse(self, store=None, kv_block_size=0):
        """Turn on park/adopt for finished requests, keyed by native block hashes."""
        from kvmem_session_store import get_session_store
        if store is None:
            store = get_session_store("worker")
        self.session_store = store
        self.kv_block_size = int(kv_block_size or self.kv_block_size or 0)
        self.session_reuse = True
        # A parked session holds a slice of H until it is evicted; hand it back then.
        store._on_evict = self._release_parked
        store._max_sessions = max(1, int(self.sessions))
        return store

    def _pure_copy(self):
        """True = compaction moves bytes without re-baking RoPE (the fork's true-position split)."""
        return os.environ.get("KVMEM_REBAKE", "").strip().lower() not in ("1", "true", "yes", "on")

    def _open_host_share(self):
        """Use a fresh H identity; evict idle checkpoints before spending another share."""
        from kvmem_session_budget import BudgetError, SessionBudgetPool

        if self.budget_pool is None:
            self.budget_pool = SessionBudgetPool(
                host_bytes=int(self.host_mib) * 1024 * 1024,
                sessions=max(1, int(self.sessions)))
        while self.budget_pool.active_sessions() >= self.budget_pool.sessions:
            if self.session_store is None or not self.session_store.evict_oldest():
                raise BudgetError("no idle H share: all configured sessions are active")
        session = int(getattr(self, "_next_host_session_id", 0)) + 1
        self._next_host_session_id = session
        return session, self.budget_pool.open(session).capacity_bytes

    def _release_parked(self, entry):
        pool = getattr(self, "budget_pool", None)
        session = getattr(entry, "session_id", None)
        if pool is not None and session is not None:
            try:
                pool.close(session)
            except Exception:
                pass

    def note_prompt_ids(self, req, ids):
        """Remember the prompt tokens: they are the resume key and the logical book origin."""
        try:
            self._prompt_ids[req] = tuple(int(t) for t in (ids or ()))
        except Exception:
            pass

    def note_session_id(self, req, session_id, cache_salt=None):
        """Carry vLLM's native (cache_salt, session_id) namespace into the worker store."""
        if (isinstance(session_id, str) and session_id and len(session_id) <= 1024
                and (cache_salt is None or isinstance(cache_salt, str))
                and len(cache_salt or "") <= 1024):
            self._session_ids[req] = (cache_salt or "", session_id)

    def _seal_session_checkpoint_kv(self, req, frontier):
        """Fill missing pre-pressure H pages from this request's own completed GPU row."""
        orch = self._orch.get(req)
        if orch is None:
            return False
        self._absorb_engine_row(req)
        mapping = self._page_map.get(req, {})
        pages = int(frontier) // self.page_tokens
        st = self._state.get(req)
        if st is None or st.compacted_in_prefill or st.row:
            return False
        try:
            row = self.view.block_table_rows()[0][st.row_index]
            for page in range(pages):
                if orch.store.has(page) or getattr(orch.store, "has_disk", lambda p: False)(page):
                    continue
                slot = mapping.get(page)
                if slot is None:
                    # The row is still the original dense prefix: the engine owns these pages,
                    # even when the lagged `_absorb_upto` map has not recorded every column.
                    # Never use this fallback after a compaction changed logical page positions.
                    if page >= len(row):
                        return False
                    slot = int(row[page].item())
                if int(slot) <= 0:
                    return False
                orch.commit_written_page(page, int(slot), baked=0)
            return all(orch.store.has(page) or
                       getattr(orch.store, "has_disk", lambda p: False)(page)
                       for page in range(pages))
        except Exception as exc:
            self.log({"kind": "session_checkpoint_seal_error", "req": req,
                      "error": f"{type(exc).__name__}: {exc}"})
            return False

    def _maybe_capture_session_checkpoint(self, req, st):
        """Capture a completed, pre-pressure prompt prefix before the next forward."""
        if (not self.session_reuse or req in self._session_checkpoints
                or self.capture_state is None or not self.kv_block_size
                or int(self.session_checkpoint_speculative_tokens) > 0
                or self.session_checkpoint_executed_frontier is None):
            return False
        owner = self._session_ids.get(req)
        ids = self._prompt_ids.get(req)
        prompt = int(self._prompt_len.get(req, 0) or 0)
        from kvmem_session_store import session_checkpoint_frontier
        target = session_checkpoint_frontier(prompt, self.kv_block_size,
                                             self.budget_tokens, self.select_block_tokens)
        if not owner or not ids or len(ids) < prompt or getattr(st, "compacted_in_prefill", False):
            return False
        try:
            frontier = int(self.session_checkpoint_executed_frontier(req))
        except Exception:
            return False
        if frontier != target:
            return False
        if int(getattr(st, "tokens", 0) or 0) > frontier:
            self.log({"kind": "session_checkpoint_skipped", "req": req,
                      "reason": "runtime ahead of GPU frontier", "frontier": frontier,
                      "runtime_tokens": int(st.tokens)})
            return False
        if not self._seal_session_checkpoint_kv(req, frontier):
            self.log({"kind": "session_checkpoint_skipped", "req": req,
                      "reason": "prefix KV could not be sealed", "frontier": frontier})
            return False
        try:
            state = self.capture_state(req, frontier) if self.has_recurrent else None
        except Exception:
            state = None
        if self.has_recurrent and state is None:
            self.log({"kind": "session_checkpoint_skipped", "req": req,
                      "reason": "recurrent StateImage unavailable", "frontier": frontier})
            return False
        exact_tokens = tuple(int(t) for t in ids[:frontier])
        index = self._retrieval.get(req)
        try:
            index_checkpoint = index.checkpoint_prefix(frontier) if index is not None else None
        except Exception as exc:
            self.log({"kind": "session_checkpoint_skipped", "req": req,
                      "reason": f"retrieval checkpoint failed: {type(exc).__name__}"})
            return False
        if self.retrieval_enabled and (index_checkpoint is None or
                index_checkpoint.covered_blocks() !=
                list(range(frontier // index_checkpoint.block_tokens))):
            return False
        stats = self._statistics.get(req) if self.stats_core else None
        if self.stats_core and (stats is None or stats.frontier != frontier):
            self.log({"kind": "session_checkpoint_skipped", "req": req,
                      "reason": "core statistics frontier incomplete", "frontier": frontier,
                      "stats_frontier": getattr(stats, "frontier", -1)})
            return False
        try:
            mean_tail = stats.mean_checkpoint(frontier) if stats is not None else None
        except Exception:
            return False
        self._session_checkpoints[req] = (
            frontier, exact_tokens, state, index_checkpoint, mean_tail)
        self.log({"kind": "session_checkpoint_captured", "req": req,
                  "frontier": frontier, "suffix": prompt - frontier,
                  "state": state is not None})
        return True

    def note_resume(self, req, hit_tokens, visible_tokens):
        """EngineCore decided `req` continues a parked session after `hit_tokens` tokens."""
        try:
            self._resume[req] = (int(hit_tokens), int(visible_tokens))
        except Exception:
            pass

    def _session_hashes(self, req, ids=None):
        ids = self._prompt_ids.get(req) if ids is None else ids
        if not ids or not self.kv_block_size:
            return []
        from kvmem_session_store import block_hash_sequence
        return block_hash_sequence(ids, self.kv_block_size)

    def _park_session(self, req):
        """Prepare a sealed prefix for EngineCore's all-TP terminal publication barrier."""
        if not self.session_reuse or self.session_store is None:
            return False
        owner_session_id = self._session_ids.get(req)
        if not owner_session_id:
            self.log({"kind": "session_park_skipped", "req": req,
                      "reason": "native session_id missing"})
            return False
        checkpoint = self._session_checkpoints.pop(req, None)
        if checkpoint is None:
            self.log({"kind": "session_park_skipped", "req": req,
                      "reason": "no sealed executed-frontier checkpoint"})
            return False
        covered, ids, state, index_checkpoint, mean_tail = checkpoint
        if (covered <= 0 or covered % max(1, self.kv_block_size)
                or len(ids) != covered or (self.has_recurrent and state is None)):
            self.log({"kind": "session_park_skipped", "req": req,
                      "reason": "invalid checkpoint frontier", "frontier": int(covered),
                      "state": state is not None})
            return False
        orch = self._orch.get(req)
        if orch is None:
            return False
        hashes = self._session_hashes(req, ids)
        pages = orch.usable_pages(covered)
        if self.retrieval_enabled and index_checkpoint is None:
            self.log({"kind": "session_park_skipped", "req": req,
                      "reason": "sealed retrieval statistics checkpoint unavailable"})
            return False
        stats = self._statistics.get(req) if self.stats_core else None
        if self.stats_core and (stats is None or stats.frontier < covered):
            self.log({"kind": "session_park_skipped", "req": req,
                      "reason": "core statistics checkpoint unavailable"})
            return False
        expected_pages = covered // max(1, self.page_tokens)
        if not hashes or len(pages) < expected_pages:
            self.log({"kind": "session_park_skipped", "req": req,
                      "reason": "checkpoint KV coverage incomplete",
                      "blocks": len(hashes), "pages": len(pages),
                      "expected_pages": expected_pages, "checkpoint_tokens": int(covered)})
            return False
        ppb = max(1, int(self.pages_per_block))
        budget = int(self.selector.budget_pages())
        capacity = (min(len(pages), budget) // ppb) * ppb
        if capacity <= 0 or capacity * self.page_tokens != covered:
            # A compressed checkpoint would seed V2 Mamba's running slot from resident length
            # while its StateImage belongs to the larger logical frontier. Only restore the full
            # early prefix until that model-state slot contract has been separately qualified.
            self.log({"kind": "session_park_skipped", "req": req,
                      "reason": "checkpoint requires full resident prefix",
                      "frontier": covered, "visible": capacity * self.page_tokens})
            return False
        import hashlib as _hl
        _digest = _hl.sha256(b"".join(
            int(t).to_bytes(4, "little", signed=False) for t in ids)).hexdigest()
        payload = {"hashes": [h.hex() for h in hashes], "covered": int(covered),
                   "prefix_digest": _digest, "kv_block_size": int(self.kv_block_size),
                   "page_tokens": int(self.page_tokens), "pages_per_block": ppb,
                   "capacity_pages": int(capacity), "has_state": state is not None,
                   "session_id": owner_session_id, "request": str(req),
                   "rank": int(getattr(self, "tp_rank", 0)),
                   "expected_ranks": int(getattr(self, "tp_size", 1))}
        if index_checkpoint is not None:
            orch.retrieval_index = index_checkpoint
        if stats is not None:
            try:
                stats.truncate_to(covered)
                stats.restore_mean_checkpoint(covered, mean_tail)
                if stats.frontier != covered or any(
                        not stats.has_block(b) for b in range(covered // stats.block_tokens)):
                    raise ValueError("checkpoint statistics coverage is incomplete")
                orch.statistics = stats
            except Exception as exc:
                self.log({"kind": "session_park_prepare_failed", "req": req,
                          "reason": f"statistics restore: {type(exc).__name__}"})
                return None
        try:
            self.session_store.park(
                hashes, orch, covered, self.page_tokens, mamba_state=state,
                session_id=getattr(orch, "session_id", None),
                owner_session_id=owner_session_id, prefix_digest=_digest,
                source_request=str(req), capacity_pages=capacity)
        except Exception as exc:
            self.log({"kind": "session_park_prepare_failed", "req": req,
                      "error": f"{type(exc).__name__}: {exc}"})
            return None
        self._preparked_requests.add(req)
        self.log({"kind": "session_park_prepared", "req": req, "blocks": len(hashes),
                  "covered": covered, "pages": len(pages), "capacity": capacity,
                  "state": state is not None, "rank": int(getattr(self, "tp_rank", 0))})
        return payload

    def prepare_session_checkpoint(self, req):
        """Called by EngineCore only after this request has emitted a terminal output."""
        try:
            pending = int(self._prev_n.get(req, 0) or 0)
            if pending:
                self.after_step(req, pending, prefill=pending >= self.page_tokens)
                self._prev_n[req] = 0
            return self._park_session(req)
        except Exception as exc:
            self.log({"kind": "session_park_prepare_error", "req": req,
                      "error": f"{type(exc).__name__}: {exc}"})
            return None

    def abort_session_checkpoint(self, req):
        if self.session_store is not None:
            self.session_store.discard_requests((str(req),))
        self._preparked_requests.discard(req)
        checkpoint = self._session_checkpoints.pop(req, None)
        return checkpoint is not None

    def check_session_checkpoint(self, owner, hashes, frontier, digest, capacity_pages):
        """Prove this TP rank still owns the exact StateImage/KV before scheduler admission."""
        if not self.session_reuse or self.session_store is None or not owner:
            return False
        found = self.session_store.lookup(tuple(hashes), owner_session_id=tuple(owner))
        if found is None:
            return False
        entry, blocks = found
        if (blocks != len(hashes) or int(entry.covered_tokens) != int(frontier)
                or int(capacity_pages) * self.page_tokens != int(frontier)
                or entry.prefix_digest != digest
                or int(entry.capacity_pages) != int(capacity_pages)
                or (self.has_recurrent and entry.mamba_state is None)):
            return False
        orch = entry.orchestrator
        if (len(orch.usable_pages(frontier)) < frontier // self.page_tokens
                or (self.retrieval_enabled and getattr(orch, "retrieval_index", None) is None)):
            return False
        if self.stats_core:
            stats = getattr(orch, "statistics", None)
            if stats is None or int(stats.frontier) != int(frontier):
                return False
        return True

    def _resume_selection(self, orch, count, total_tokens):
        """Exactly `count` pages to bring back: sink head, then the newest tail, then inward.

        Without a query there are no retrieval scores yet, so the KVMem policy degenerates to
        "sink + recent" and fills inward from the newest tokens. The first decode reselect then
        re-scores the middle with the real query, exactly like a normal compaction.
        """
        avail = orch.usable_pages(total_tokens)
        count = max(0, min(int(count), len(avail)))
        if count <= 0:
            return []
        sink = max(1, int(getattr(self, "sink_tokens", 0) or 0) // self.page_tokens)
        recent = max(1, int(getattr(self, "recent_tokens", 0) or 0) // self.page_tokens)
        chosen, seen = [], set()

        def take(seq):
            for p in seq:
                if len(chosen) >= count:
                    return
                if p in seen:
                    continue
                seen.add(p)
                chosen.append(p)

        take(avail[:sink])
        take(reversed(avail[-recent:]))
        take(reversed(avail))
        return sorted(chosen)

    def _complete_resume(self, req, st, row_index):
        """Install a resumed session's working set into the engine's already-allocated row.

        EngineCore has set the resident book to `visible_tokens` (= |sel| * page_tokens) and the
        served book to the hit length, so the engine has allocated exactly |sel| pages for this
        request. The KV for those logical pages lives in the parked store and is restored here at
        the compacted rank each page occupies -- one rotation hop from the stored bytes.
        """
        entry, hit_tokens, visible_tokens = st.resume
        if int(visible_tokens) != int(hit_tokens):
            raise RuntimeError("compressed session checkpoint lacks a valid Mamba state cursor")
        orch = self._orch.get(req)
        if orch is None:
            return False
        count = int(visible_tokens) // self.page_tokens
        pages = self._resume_selection(orch, count, int(hit_tokens))
        self._absorb_engine_row(req)
        # `st.row` may already mirror the engine's appended pages for THIS step (the plugin
        # syncs the block-table row before prepare_inputs). Those entries beyond the restored
        # prefix are the scheduled chunk's own pages and must survive the resume; dropping them
        # left the view one page short of the attention length the scheduler had just admitted.
        # Only the synced prefix is trusted: block-table entries past the engine's write cursor
        # can be stale ids from an earlier request, so they are never adopted here.
        synced = tuple(int(v) for v in (getattr(st, "row", ()) or ()))
        try:
            values = [int(v) for v in self.view.block_table_rows()[0][row_index].tolist()]
        except Exception:
            values = []
        # Take the row PREFIX of exactly |sel| entries: the engine allocated |sel| pages for the
        # resident book it was given. Page id 0 is a legitimate page (this geometry numbers pages
        # from 0), so it must NOT be filtered out as "empty" -- dropping it shifted the resident
        # book by one page (16 tokens) against the engine's, and the resumed turn answered with
        # garbage.
        row_values = values[:len(pages)]
        if len(pages) != count or len(row_values) != count or any(v <= 0 for v in row_values):
            self.log({"kind": "resume_no_room", "req": req, "pages": len(pages),
                      "row": len(row_values), "visible": int(visible_tokens)})
            return False
        for rank, page in enumerate(pages):
            try:
                orch.restore_page_rank(page, row_values[rank], rank)
            except Exception:
                self.log({"kind": "resume_restore_failed", "req": req, "page": int(page)})
                return False
        restored = tuple(row_values[:len(pages)])
        # Preserve the engine's already-synced tail (this step's chunk pages). A resumed
        # request schedules the rest of its prompt in the same admission, so the row must cover
        # `resident prefix + scheduled tokens` before attention is built.
        st.row = restored + (synced[len(restored):] if len(synced) > len(restored) else ())
        if self.set_block_cursor is not None:
            try:
                self.set_block_cursor(req, len(st.row))
            except Exception:
                pass
        if self.has_recurrent:
            if (self.restore_state is None or entry.mamba_state is None
                    or not self.restore_state(req, entry.mamba_state)):
                raise RuntimeError("resumed GDN StateImage could not be fully restored")
        st.resume = None
        st.seq_len = len(pages) * self.page_tokens
        st.tokens = int(hit_tokens)
        st.prefill_done = False
        st.resumed = True
        # KVMem's R cadence counts OUTPUT tokens from the resume point, not from the parked
        # session's last compaction.
        orch._tokens_at_reselect = st.seq_len
        orch.prompt_tokens = len(self._prompt_ids.get(req, ()) or ())
        orch.note_tokens(st.tokens)
        self.log({"kind": "session_resumed", "req": req, "hit": int(hit_tokens),
                  "resident": st.seq_len, "pages": len(pages),
                  "state": entry.mamba_state is not None})
        return True

    def _adopt_session(self, req):
        """Take the parked session for `req` out of the store, if EngineCore asked for a resume."""
        pending = self._resume.pop(req, None)
        if pending is None or not self.session_reuse or self.session_store is None:
            return None
        hit_tokens, visible_tokens = pending
        hashes = self._session_hashes(req)
        if not hashes:
            return None
        owner_session_id = self._session_ids.get(req)
        if not owner_session_id:
            self.log({"kind": "resume_skip", "req": req, "reason": "native session_id missing"})
            return None
        ids = tuple(self._prompt_ids.get(req, ()) or ())

        def _digest(frontier):
            import hashlib
            return hashlib.sha256(b"".join(
                int(t).to_bytes(4, "little", signed=False)
                for t in ids[:int(frontier)])).hexdigest()

        found = self.session_store.lookup(
            hashes, owner_session_id=owner_session_id, prefix_digest=_digest)
        if found is None:
            self.log({"kind": "resume_miss", "req": req, "hit": hit_tokens})
            return None
        entry, _blocks = found
        if int(entry.covered_tokens) != int(hit_tokens):
            self.log({"kind": "resume_frontier_mismatch", "req": req,
                      "covered": entry.covered_tokens, "hit": hit_tokens})
            return None
        if self.retrieval_enabled and getattr(entry.orchestrator, "retrieval_index", None) is None:
            self.log({"kind": "resume_miss", "req": req,
                      "reason": "sealed retrieval statistics missing"})
            return None
        if self.stats_core and (getattr(entry.orchestrator, "statistics", None) is None
                or int(entry.orchestrator.statistics.frontier) != int(hit_tokens)):
            self.log({"kind": "resume_miss", "req": req,
                      "reason": "sealed core statistics missing"})
            return None
        self.session_store.take(entry)
        return entry, int(hit_tokens), int(visible_tokens)

    def retrieval_index(self, req):
        return self._retrieval.get(req)

    # --- mean-K statistics (kvmem RawKvStore, v0.19 design) --------------------------------
    def attach_statistics(self, req):
        """Give this request the kvmem statistics store the recall design owns.

        Upstream keeps one `RawKvStore` per sequence holding only mean-K sums
        (`n_layer = full-attention layers`, `n_embd_k = kv_heads*head_dim`, `block_tokens = 64`)
        and fills it from device-side bucket sums. The store is per REQUEST because a request is
        the unit the engine rewinds/truncates.
        """
        if not self.stats_core:
            return None
        layers = int(self.statistics_layers or 0)
        heads = int(self.kv_heads or 0)
        if layers <= 0 or heads <= 0:
            self._skip("(global)", "no geometry")
            return None
        stats = self._statistics.get(req)
        if stats is None:
            try:
                from kvmem_statistics import KvmemStatistics
                stats = KvmemStatistics(layers, heads, self.head_dim)
            except Exception as exc:                   # binding missing / build stale
                self._skip("(global)", f"store init: {exc}")
                return None
            self._statistics[req] = stats
        self._stats_next.setdefault(req, 0)
        # The link is re-made on every call: `_ensure` attaches the store before it registers the
        # orchestrator, so the first attempt found no orchestrator and the request kept a store the
        # orchestrator never knew about - which silently disabled the core-vs-legacy audit
        # (measured: orch=True index=True statistics=False on every probe).
        orch = self._orch.get(req)
        if orch is not None:
            orch.statistics = stats
        return stats

    def _skip(self, req, reason):
        """Count a statistics refusal by reason (a single `setdefault` slot hid the real cause)."""
        bucket = self._stats_skipped.setdefault(req, {})
        bucket[reason] = int(bucket.get(reason, 0)) + 1

    def publish_position_offset(self, req):
        """Tell the capture op this step's compacted->true offset (before the model runs)."""
        if not self.stats_core or req not in self._statistics:
            return
        try:
            st = self.state(req)
            offset = max(0, int(getattr(st, "tokens", 0)) - int(getattr(st, "seq_len", 0)))
            from vllm import kvmem_bridge as bridge
            bridge.set_position_offset(offset)
        except Exception:
            pass

    def harvest_decode_rows(self, req, take):
        """Fold the ACCEPTED decode rows of the PREVIOUS step at their true positions.

        Called from the plugin's pre-step hook: the bridge still holds the previous step's rows
        (nothing has run yet), and the accepted count is known by then - the post-step hook sees
        pre-postprocess counters, which under async scheduling lags by one step (measured: 3 rows
        written for 7 accepted tokens).
        """
        if not self.stats_core:
            return 0
        stats = self._statistics.get(req)
        take = int(take or 0)
        if stats is None or take <= 0:
            return 0
        try:
            from vllm import kvmem_bridge as bridge
            import numpy as _np
            # The STORE's frontier is authoritative here: `_stats_next` is advanced by the prefill
            # path too, and using a stale copy wrote rows past the prompt (measured: first=55448).
            first = int(getattr(stats, "frontier", 0) or 0)
            written = 0
            for layer in range(int(stats.layers)):
                rows = bridge.read_key_rows(layer, take)
                if rows is None:
                    continue
                arr = _np.ascontiguousarray(rows.numpy() if hasattr(rows, "numpy") else rows,
                                            dtype=_np.float32)
                for i in range(min(take, int(arr.shape[0]))):
                    stats._store.write_layer_mean_sum(first + i, 1, layer, arr[i])
                    written += 1
            if written:
                stats.bucket_writes += written
                stats.steps += 1
                stats.frontier = first + take
                self._stats_next[req] = first + take
                self._stats_written[req] = int(self._stats_written.get(req, 0)) + written
            return written
        except Exception:
            if self.log is not None:
                import traceback as _tb
                self.log({"kind": "stats_rows_error", "req": req,
                          "error": _tb.format_exc()[-400:]})
            return 0

    def harvest_key_statistics(self, req, n_tokens):
        """Drain this step's device bucket sums into the request's kvmem RawKvStore.

        Called right AFTER the model ran for the step, so the bucket rows describe exactly the
        tokens the model pushed through. Prompt steps are exact; a step that carries speculative
        drafts is refused, because the model writes 1+draft rows while acceptance is decided
        later, and committing them would inject rejected KV into the statistics.
        """
        if not self.stats_core:
            return 0
        stats = self._statistics.get(req)
        if stats is None:
            # `attach_statistics` runs when the request is created, which can be BEFORE the plugin
            # has published the model geometry (the bridge is installed on the first step of the
            # first request), so retry here rather than leaving that request without a store.
            stats = self.attach_statistics(req)
            if stats is None:
                self._skip(req, "no store")
                return 0
        n = int(n_tokens or 0)
        if n <= 0:
            return 0
        # Which tokens of this step may enter the statistics:
        #   * a prompt step contributes every scheduled token (the bridge accumulated them all);
        #   * a draft-carrying decode step contributes ONLY the target row - the plugin publishes
        #     KEY_ROWS_LIMIT=1 before such a step, so the bucket sums hold exactly that token and the
        #     accepted frontier advances by 1, not by 1+K;
        #   * a draft-free decode step contributes its single token.
        drafts = req in (getattr(self, "spec_reqs", None) or ())
        decode = False
        try:
            decode = bool(getattr(self.state(req), "prefill_done", False))
        except Exception:
            decode = False
        if drafts:
            import vllm.kvmem_bridge as _kb
            if int(getattr(_kb, "KEY_ROWS_LIMIT", 0) or 0) != 1:
                self._skip(req, "drafts")
                return 0
            n = 1
        elif decode and not getattr(self, "stats_decode", False):
            # Decode: fold exactly the ACCEPTED rows of this step. The bridge kept them as rows
            # rather than bucket sums precisely because acceptance is decided after the step, and
            # each accepted token still has to land on the true frontier.
            n = min(int(n), int(getattr(self, "_accepted_rows", 0) or 0))
        if int(getattr(self, "batch_size", 1) or 1) > 1:
            # One position offset is published per step; a batched step would attribute every
            # request's buckets to the same offset.
            self._skip(req, "batch")
            return 0
        try:
            from vllm import kvmem_bridge as bridge
            from kvmem_statistics import stats_from_bridge
            first = int(self._stats_next.get(req, 0))
            sums = stats_from_bridge(bridge, range(stats.layers), first, n)
            if not sums:
                self._skip(req, "no capture")
                return 0
            written = stats.write_step(first, n, sums)
            if written == 0:
                self._skip(req, f"wrote 0 (frontier {stats.frontier} first {first} n {n})")
            self._stats_next[req] = first + n
            try:
                import json as _json, os as _os
                if int(self._stats_written.get("_dbg", 0)) < 40:
                    self._stats_written["_dbg"] = int(self._stats_written.get("_dbg", 0)) + 1
                    with open("/tmp/kvmem_stats_debug.jsonl", "a") as _fh:
                        _fh.write(_json.dumps({"pid": _os.getpid(), "req": str(req)[:12],
                                               "pid_f": "harvest", "first": first, "n": n,
                                               "layers": len(sums), "written": written,
                                               "frontier": stats.frontier,
                                               "sums_shape": [int(x) for x in
                                                              list(sums.values())[0].shape]}) + "\n")
            except Exception:
                pass
            self._stats_written[req] = int(self._stats_written.get(req, 0)) + int(written)
            # P1 audit hook: once the whole prompt has statistics, score the SAME blocks with
            # kvmem's mean-K scorer and with the legacy Python index. This fires exactly when the
            # query window holds the question, not at release time after generation.
            if written and not self._compare_done.get(req):
                plen = int(self._prompt_len.get(req, 0) or 0)
                if int(stats.frontier) >= min(plen or 4096, 4096):
                    self.attach_statistics(req)     # idempotent; re-links orch.statistics
                    orch = self._orch.get(req)
                    index = self._retrieval.get(req)
                    # Unconditional probe: the audit silently never fired for four engine cycles
                    # because one link of this chain was None and the request was marked "done"
                    # anyway. Print the whole chain once per link state, and only mark it done once
                    # the audit actually ran.
                    try:
                        from vllm import kvmem_bridge as _bridge
                        armed = bool(getattr(_bridge, "ARMED", False))
                    except Exception:
                        armed = None
                    try:
                        covered = len(list(index.covered_blocks())) if index is not None else -1
                    except Exception:
                        covered = -2
                    if int(self._stats_written.get("_audit_probe", 0)) < 12:
                        self._stats_written["_audit_probe"] = int(
                            self._stats_written.get("_audit_probe", 0)) + 1
                        self.log({"kind": "stats_audit_probe", "req": str(req)[:12],
                                  "frontier": int(stats.frontier), "plen": plen,
                                  "orch": orch is not None, "index": index is not None,
                                  "covered": covered, "armed": armed,
                                  "orch_statistics": bool(orch is not None
                                                          and getattr(orch, "statistics", None)),
                                  "write_step": True})
                    if orch is not None and index is not None and armed and covered > 0:
                        blocks = list(index.covered_blocks())
                        legacy = []
                        # The legacy index only has a QUERY after a reselect-time harvest, so reading
                        # it here raised "no query captured" and the audit silently never ran
                        # (measured: scorer_compare = 0 records while every other link was healthy).
                        try:
                            from kvmem_engine_capture import harvest_from_bridge
                            if self.capture is not None:
                                harvest_from_bridge(self.capture)
                            legacy = [float(v) for v in index.scores(blocks)]
                        except Exception as exc:
                            self.log({"kind": "stats_audit_legacy_error",
                                      "req": str(req)[:12],
                                      "error": f"{type(exc).__name__}: {exc}"})
                        if legacy:
                            orch._compare_core_scores(blocks, legacy)
                            self._compare_done[req] = True
                        else:
                            self._stats_written["_audit_no_legacy"] = int(
                                self._stats_written.get("_audit_no_legacy", 0)) + 1
            return written
        except Exception:
            try:
                import traceback as _tb
                self.log({"kind": "stats_error", "req": req, "error": _tb.format_exc()[-800:]})
            except Exception:
                pass
            return 0

    def statistics_stats(self, req):
        stats = self._statistics.get(req)
        if stats is None:
            return {"enabled": bool(self.stats_core), "present": False,
                    "skipped": self._stats_skipped.get(req)}
        out = stats.stats()
        out["written"] = int(self._stats_written.get(req, 0))
        out["next"] = int(self._stats_next.get(req, 0))
        out["skipped"] = self._stats_skipped.get(req)
        return out

    def live_requests(self):
        """Requests the runtime still holds state for (for the worker-side sweep)."""
        return list(self._orch.keys())

    def release(self, req):
        """Retire a request; only a terminal-output handshake may authorize a session park."""
        st = self._state.get(req)
        row = getattr(st, "row_index", None) if st is not None else None
        # One unconditional record per request: whether the kvmem statistics store actually
        # received this prompt. Without it the only evidence was a sampled step log, and a silent
        # harvest refusal looked identical to "statistics are off".
        if self.stats_core and self.log is not None:
            try:
                self.log({"kind": "statistics_final", "req": str(req)[:12],
                          **self.statistics_stats(req)})
            except Exception:
                pass
        # Only an already acknowledged EngineCore terminal checkpoint retains this H share.
        # The next prepare_inputs() sweep must never create a park after admission has happened.
        parked = req in self._preparked_requests
        self._preparked_requests.discard(req)
        orch = self._orch.pop(req, None)
        self._retrieval.pop(req, None)
        self._state.pop(req, None)
        # The attention seq_lens override is indexed by BATCH ROW, not by request: if it outlives
        # the request, the next request reusing that row inherits the old compacted length while
        # it is still prefilling (row empty -> before_step returns early and never rewrites it).
        # Measured: the second request on a warmed engine answered with noise from its first token.
        if row is not None:
            self._seq_override.pop(int(row), None)
        # Free the device slots as well: dropping only `_owned_slots` left `_slot_owner` entries
        # pointing at dead requests, so a later request could never reuse them (the slot pool
        # silently shrinks to exhaustion) and the stale owner table outlived the request.
        for slot in list(self._owned_slots.pop(req, set()) or ()):
            if self._slot_owner.get(slot) == req:
                self._slot_owner.pop(slot, None)
        self._page_map.pop(req, None)
        self._absorb_upto.pop(req, None)
        self._engine_logical.pop(req, None)
        self._dirty_pages.pop(req, None)
        self._engine_computed.pop(req, None)
        self._prompt_len.pop(req, None)
        self._over_budget_logged.discard(req)
        self._pending_frontier.pop(req, None)
        self._pending_new.pop(req, None)
        self._inflight.pop(req, None)
        self._prev_n.pop(req, None)
        self._prompt_ids.pop(req, None)
        self._session_ids.pop(req, None)
        self._session_checkpoints.pop(req, None)
        self._statistics.pop(req, None)
        self._stats_next.pop(req, None)
        self._stats_written.pop(req, None)
        self._stats_skipped.pop(req, None)
        self._resume.pop(req, None)
        getattr(self, "_rebuilt_req", set()).discard(req)
        if (orch is not None and not parked
                and getattr(self, "budget_pool", None) is not None):
            session = getattr(orch, "session_id", None)
            if session is not None:
                self.budget_pool.close(session)
        return orch is not None

    def state(self, req):
        self._ensure(req)
        return self._state[req]

    def orchestrator(self, req):
        self._ensure(req)
        orch = self._orch[req]
        if hasattr(orch, "inrequest_compaction"):
            orch.inrequest_compaction = bool(getattr(self, "inrequest_compaction", False))
        return orch

    # --- hook 1 ------------------------------------------------------------------------
    def before_step(self, req, n_query_tokens=1, row_index=None, position_slots=None,
                    rows_only=False):
        """Install the request's sparse view. `row_index` is the batch row of the request and
        `position_slots` the positions[] indices of its query tokens (live runner: derived from
        idx_mapping and query_start_loc)."""
        if self.is_rebuilt(req):
            return None
        self._ensure(req)
        st = self._state[req]
        if row_index is not None:
            # Bind the ROW even before a view exists: the first reselect must know which block
            # table row is the request's own (it compacts inside the request's own pages).
            st.row_index = int(row_index)
        # Slot-mapping history: remember THIS step's query-token range so the LAGGED commit
        # (which runs at the start of the NEXT step, while the device slot mapping still holds
        # this step's values) can ask the kernel where it wrote each page.
        try:
            if position_slots:
                self._slot_span[req] = (int(min(position_slots)), int(max(position_slots)) + 1)
        except Exception:
            pass
        # Without in-request compaction the sparse view must NOT be installed: the block table is
        # vLLM's own, and overriding seq_lens/positions on top of an uncompacted row is exactly the
        # half-initialised state that produced gibberish answers (measured with
        # --kvmem-inrequest-compaction off). Retrieval statistics can still be collected.
        if not bool(getattr(self, "inrequest_compaction", False)):
            return None
        if st.resume is not None and row_index is not None:
            # First step of a request that continues a parked session: fill the engine's freshly
            # allocated resident pages from the store and adopt the parked token book. This must run
            # BEFORE the `not st.row` early-return below (a resumed request has no row yet).
            if not self._complete_resume(req, st, int(row_index)):
                raise RuntimeError("KVMem session checkpoint restore could not complete")
        if not st.row:
            return None
        row_index = req if row_index is None else int(row_index)
        if position_slots is None:
            position_slots = [req * n_query_tokens + j for j in range(n_query_tokens)]
        st.row_index = row_index
        step_end = int(st.seq_len) + len(position_slots)
        capacity = len(st.row) * int(self.page_tokens)
        if st.seq_len < 0 or step_end > capacity:
            raise SparseViewCapacityError(
                f"request {req}: sparse attention needs {step_end} tokens "
                f"but {len(st.row)} published pages cover only {capacity}")
        if rows_only:
            # Pre-gather phase: only the block table row must be in place (the kernel's gathered
            # copy is taken from it). seq_lens/positions MUST be written after the engine's own
            # `prepare_pos_seq_lens`, because that runs with the optimistic device counter, which a
            # compaction does not rewind - writing ours first would be overwritten with a stale
            # length and the attention would read pages beyond the row.
            if self.install_row and st.row and not bool(getattr(self, "sched_compact", False)):
                import torch
                for table in self.view.block_table_rows():
                    table[row_index, :len(st.row)] = torch.tensor(
                        st.row, dtype=table.dtype, device=table.device)
            return st
        self._seq_override[row_index] = step_end
        # Positions must stay TRUE once a compaction makes logical > resident: the fork adds
        # `kvmem_rope_offset` to the engine's positions for RoPE and `prepare_attn` subtracts
        # `kvmem_slot_offset` for the write address. Overwriting them with compacted values here
        # made the slot index negative after the boundary compaction (measured: CUDA illegal access
        # at 163,840 tokens; the same run completed with positions left alone).
        self._positions_must_stay_true = (
            int(st.tokens) > int(st.seq_len)
            and not (os.environ.get("KVMEM_NO_ROPE_OFFSET", "").strip() in ("1", "true", "yes", "on")))
        if not self.install:
            return st
        # seq_lens must COVER the tokens of this step: vLLM expects the number of computed
        # tokens INCLUDING the current query. Installing the previous end (what we did before)
        # made every chunk's queries blind to their own chunk — the KV was written but never
        # attended, which silently produced gibberish instead of an error.
        _sched_c = bool(getattr(self, "sched_compact", False))
        _prefix = bool(getattr(self, "_prefix_active", False))
        if _prefix and getattr(self, "compact_mode", None) == "prefix":
            # The window IS the installed row: only the first n_keep entries describe the
            # attention window; the row's tail entries are the engine's own pages (read never
            # reaches them because seq_len stops inside the window).
            row_to_install = st.row if st.row else None
        else:
            row_to_install = ((st.row if (_sched_c and bool(
                                  getattr(self, "install_row_in_view", False)))
                              else (None if _sched_c else st.row)))
        self._install(row_index, row_to_install, step_end, list(position_slots))
        if self.install_row:
            self._absorb_upto[req] = max(int(self._absorb_upto.get(req, 0)), len(st.row))
        return st

    def _install(self, row_index, row, seq_len, position_slots):
        import torch

        if self.install_row and row is not None:
            for table in self.view.block_table_rows():
                # Only overwrite the slots we own: vLLM reads `seq_len` tokens, so entries past it
                # are ignored, and clobbering the whole row would disturb the engine's own
                # prefix-cache bookkeeping.
                table[row_index, :len(row)] = torch.tensor(row, dtype=table.dtype,
                                                           device=table.device)
        if self.install_seq:
            self.view.seq_lens()[row_index] = seq_len
        positions = self.view.positions()
        # The query tokens sit right after the keys, in compacted space: their absolute positions
        # are [seq_len - n, seq_len), NOT [seq_len, seq_len + n). The previous form used the
        # INCLUDING-query length as the base, so every step shifted positions by one (decode) or
        # by a whole chunk (prefill) - the KV was then rotated to the wrong RoPE and the output
        # degenerated into gibberish (measured live).
        if self.install_pos and not getattr(self, "_positions_must_stay_true", False):
            base = seq_len - len(position_slots)
            for j, slot in enumerate(position_slots):
                positions[slot] = base + j
        return {"row": row, "seq_len": seq_len}

    # --- hook 2 ------------------------------------------------------------------------
    def note_frontier(self, req, frontier):
        """Set the request's absolute token frontier from the scheduler (authoritative)."""
        st = self.state(req)
        self._pending_frontier[req] = int(frontier)

    def note_engine_logical(self, req, tokens):
        """The served-token count (logical frontier) as the ENGINE counts it."""
        try:
            self._engine_logical[req] = int(tokens)
        except Exception:
            pass

    def note_prompt_len(self, req, tokens):
        """The request's prompt length as the ENGINE counts it (post tokenizer)."""
        try:
            self._prompt_len[req] = int(tokens)
        except Exception:
            pass

    def note_engine_computed(self, req, computed):
        """The engine's own num_computed_tokens for this request (accepted tokens only)."""
        try:
            self._engine_computed[req] = int(computed)
        except Exception:
            pass

    def note_pending(self, req, n_new):
        """Record the scheduler's token count for this step (seen before the model runs)."""
        st = self.state(req)
        self._pending_new[req] = int(n_new)

    def sync_row_from_engine(self, req, written, n_tokens=0):
        """Copy the pages the ENGINE has actually written into the request's row.

        `num_blocks.np[group, req]` is the engine's own count of written page slots: it is advanced
        by `append_block_ids` for every new block and was reset to the compacted length by the last
        compaction. Mirroring exactly that prefix keeps the row faithful - mirroring further (which
        is what the earlier attempt did) copied slots the engine had not written yet, and the
        attention then read zeros/stale ids (measured: known-good 88K answer turned into garbage).

        Returns the number of entries added.
        """
        st = self._state.get(req)
        w = int(written or 0)
        if st is None or w <= 0:
            return 0
        try:
            if w <= len(st.row) and not bool(getattr(self, "sched_compact", False)):
                return 0
            if w <= len(st.row) and bool(getattr(self, "sched_compact", False)):
                # In boundary mode the engine's table is authoritative EVERYWHERE, not only past
                # the row length: the adopted prefix plus its appends is what the kernels read.
                # Mirror it (positions/slot ids for the commit come from here).
                pass
            table = self.view.block_table_rows()[0]
            values = table[st.row_index].tolist()[:w]
            if (bool(getattr(self, "sched_compact", False))
                    and bool(getattr(st, "applied_view_coord", False))
                    and w > len(st.row)):
                # A boundary-compacted request: the ENGINE table runs in logical coordinates
                # (the feed appends blocks for the un-trimmed book), so its entries past the
                # adopted view are NOT the view's pages. Keep our view prefix and append the
                # new engine entries the scheduler published for this step's growth.
                grown = list(st.row) + values[len(st.row):w]
                values = [int(p) for p in grown]
            st.row = tuple(values)
            if bool(getattr(self, "sched_compact", False)):
                # Mirror `logical` alongside the row in boundary mode: entries past the old
                # length are the request's APPENDED pages, whose logical page is the row
                # position plus the offset the last applied compaction left (`st.off_pages`).
                # Without this, commit/store lookups drift exactly when the row grows past the
                # adopted view (measured class: post-adopt decode never committed its pages).
                off = int(getattr(st, "off_pages", 0) or 0)
                old_log = list(st.logical or ())
                if len(old_log) < len(st.row):
                    for rank in range(len(old_log), len(st.row)):
                        old_log.append(rank + off)
                    st.logical = tuple(old_log[:len(st.row)])
            # `seq_len` is the resident length written BEFORE this step; `before_step` adds the
            # current query exactly once. Deriving it here as `tokens + n_tokens` made the two
            # additions double-count (live: prefill step scheduled 4,096 tokens against a 4,704-token
            # row and installed seq_lens=8,192, then every step was clamped by one chunk).
            cap = len(st.row) * max(1, self.page_tokens)
            if int(getattr(st, "seq_len", 0) or 0) <= 0:
                st.seq_len = 0
            elif int(st.seq_len) > cap:
                st.seq_len = cap
            if self.log is not None:
                self.log({"kind": "row_synced", "req": req, "row": len(st.row),
                          "seq_len": int(st.seq_len), "tokens": int(getattr(st, "tokens", 0)),
                          "n": int(n_tokens or 0)})
            return len(st.row)
        except Exception:
            if self.log is not None:
                import traceback as _tb
                self.log({"kind": "row_sync_error", "req": req, "error": _tb.format_exc()[-300:]})
            return 0

    def ensure_row_for_chunk(self, req, n_tokens):
        """Mirror the engine's appended blocks so the row covers the attention length THIS step.

        The row used to be extended only in `after_step`, so every prefill chunk ran with a row one
        chunk short of the `seq_len` the kernel was told to attend to (measured: seq_len=49152 needs
        3072 entries, row had 2828 -> deficit 244). This mirrors entries the ENGINE really appended
        (never synthesises from the page map, which mixed logical pages with resident ranks and
        produced duplicate ids).
        """
        if not bool(getattr(self, "row_grow_pre_step", False)):
            return 0
        st = self._state.get(req)
        n = int(n_tokens or 0)
        if st is None or n <= 0:
            return 0
        try:
            need = (int(getattr(st, "seq_len", 0)) + n + self.page_tokens - 1) // max(1, self.page_tokens)
            if len(st.row) >= need:
                return 0
            table = self.view.block_table_rows()[0]
            values = table[st.row_index].tolist()
            grown = list(st.row)
            for i in range(len(grown), min(need, len(values))):
                page = int(values[i])
                if page <= 0:
                    break
                grown.append(page)
            added = len(grown) - len(st.row)
            if added:
                st.row = tuple(grown)
                if self.log is not None:
                    self.log({"kind": "row_mirrored_pre_step", "req": req, "added": int(added),
                              "row": len(st.row), "need": int(need)})
            if len(st.row) < need and self.log is not None:
                self.log({"kind": "row_short_pre_step", "req": req, "row": len(st.row),
                          "need": int(need), "seq_len": int(getattr(st, "seq_len", 0)), "n": n})
            return added
        except Exception:
            if self.log is not None:
                import traceback as _tb
                self.log({"kind": "row_grow_error", "req": req, "error": _tb.format_exc()[-300:]})
            return 0

    def tail_replay(self, req, window=None):
        """Recompute the last `window` resident tokens against the CURRENT row.

        Upstream's QueryReplay: the whole-history state keeps whatever approximation the compactions
        left, but the tail - the tokens that shape sampling - is recomputed against the selection we
        actually installed. It only rewinds the two token books (an operation the compaction path
        already does every chunk), so the sampler is unaffected; rewriting the token book is what
        stalled the request.
        """
        st = self._state.get(req)
        if st is None:
            return None
        try:
            visible = int(getattr(st, "seq_len", 0))
            n = min(int(window or getattr(self, "replay_window", 2048) or 2048), visible)
            if n <= 0 or self.compact_channel is None:
                return None
            res = self.compact_channel.replay(req, visible, n)
            if self.log is not None:
                self.log({"kind": "tail_replay", "req": req, "visible": visible,
                          "n": int(n), "reply": res})
            return res
        except Exception:
            if self.log is not None:
                import traceback as _tb
                self.log({"kind": "tail_replay_error", "req": req,
                          "error": _tb.format_exc()[-300:]})
            return None

    def advance_tokens(self, req, n_new):
        self._inflight[req] = int(n_new or 0)
        # Prompt-complete reselect: once the WHOLE prompt is in, the question is inside the captured
        # query window, so rank the working set with the real query. This used to hang off the first
        # DECODE step, which a long prompt never reaches in that state (the compaction paths set
        # `prefill_done` earlier), so a 147K prompt finished prefill, never re-ranked, and answered
        # with the tail instead of the middle ("7").
        try:
            _stq = self.state(req)
            _plq = int(self._prompt_len.get(req, 0) or 0)
            _orchq = self._orch.get(req)
            # Detect "the prompt just finished" without trusting `_prompt_len`: the step AFTER a
            # prefill chunk is the first decode step. (The prompt-length check never fired for a
            # long prompt - 147K prefilled, no re-rank, tail answer.)
            _prev_n = int(getattr(self, "_prev_chunk", {}).get(req, 0) or 0)
            try:
                self._prev_chunk = getattr(self, "_prev_chunk", {})
                self._prev_chunk[req] = int(n_new or 0)
            except Exception:
                pass
            # Re-ranking the working set right at the end of prefill REPLACES the row the recurrent
            # state was built against. Measured on a 147K prompt: the ranking found the needle block
            # at the top (score 5.32 vs 1.77) but the answer went from correct to "7", because the
            # state no longer matches the installed view. Off unless explicitly enabled; the
            # state-consistent re-rank needs a reconstruction pass.
            _finished = (not self.compact_rebuild and self.reselect_at_prompt_end
                         and _prev_n > 1 and int(n_new or 0) <= 1)
            if (not self.compact_rebuild
                    # In Core-boundary mode the final, query-driven reselect runs on the EngineCore
                    # step boundary (after the last chunk's output); doing it here would use the
                    # socket path and leave the next chunk one page short (measured: 38,945 needed
                    # vs 38,944 covered).
                    and not bool(getattr(self, "sched_compact", False))
                    and ((_plq and int(getattr(_stq, "tokens", 0)) >= _plq) or _finished)
                    and not getattr(_stq, "query_reselected", False)
                    and getattr(_stq, "row", ())
                    and _orchq is not None
                    and getattr(_orchq, "retrieval_index", None) is not None):
                _stq.query_reselected = True
                if self.log is not None:
                    self.log({"kind": "query_reselect", "req": req, "tokens": int(_stq.tokens),
                              "prompt": _plq, "n_new": int(n_new or 0), "why": "prompt_complete"})
                # `rebuild` (reconstruction) is opt-in: it rewrites the engine's token book so the
                # selected sequence can be re-prefilled from scratch. `tail_replay` is the cheap
                # alternative (QueryReplay) and runs whenever the checkpoint-replay switch is on.
                self._reselect(req, probe=True,
                               rebuild=bool(getattr(self, "compact_rebuild", False)))
                if bool(getattr(self, "checkpoint_replay", False)):
                    self.tail_replay(req)
        except Exception:
            if self.log is not None:
                import traceback as _tb
                self.log({"kind": "query_reselect_error", "req": req,
                          "error": _tb.format_exc()[-400:]})
        # QueryReplay boundary: this step prefills the FINAL chunk of the prompt, so snapshot the
        # recurrent state BEFORE it runs. Replaying that chunk afterwards must start from this state
        # or the tokens are applied to the GDN state twice.
        try:
            _st0 = self.state(req)
            # Capture ONCE, at the boundary that the replay will return to: the last <=2 chunks of
            # the prompt. The snapshot must be taken at the START of the step (before the model
            # runs), and the replay length is what is still missing at that moment (capped at the
            # replay window). Capturing on every step instead let a final 1-token step overwrite it
            # and reduced the replay to a single token - which reprocesses nothing.
            _plen = int(self._prompt_len.get(req, 0) or 0)
            _left = _plen - int(getattr(_st0, "tokens", 0))
            if (_plen and 0 < _left <= 8192 and getattr(_st0, "replay_snap", None) is None
                    and self.capture_state is not None):
                _snap = self.capture_state(req)
                if _snap:
                    _st0.replay_snap = _snap
                    # Remember WHERE the snapshot was taken. The replay length is derived from it and
                    # the engine's visible length, so both TP ranks agree on n: taking n from "what
                    # was left" instead let the two ranks disagree (2042 vs 2048) and the first
                    # message won, which rewound to the wrong position.
                    _st0.replay_at = int(getattr(_st0, "tokens", 0))
                    # The rewind happens in RESIDENT coordinates (num_computed_tokens), so the
                    # boundary must be recorded in the same units. Mixing logical and resident made
                    # n negative and the replay a no-op that still cost a message.
                    _st0.replay_at_resident = int(getattr(_st0, "seq_len", 0))
                    if self.log is not None:
                        self.log({"kind": "replay_snapshot", "req": req, "left": int(_left),
                                  "at": int(_st0.replay_at)})
        except Exception:
            pass
        """One step of accounting, driven from the entry of the runner's execute_model wrapper.

        Committing is deliberately LAGGED by one step: at the entry of step k the model has not yet
        written step k's KV, but step k-1's pages are complete. Committing eagerly produced
        "selected pages without canonical bytes" because the selection named pages whose bytes did
        not exist yet. The post-block only runs for some steps, so it cannot be relied on.
        """
        n_new = int(n_new or 0)
        if n_new <= 0:
            return None
        try:
            self._slot_span_prev[req] = self._slot_span.get(req)
        except Exception:
            pass
        st = self.state(req)
        orch = self._orch[req]
        prev = self._prev_n.get(req, 0)
        if prev:
            try:
                self.after_step(req, prev, prefill=prev >= self.page_tokens)
            except Exception:
                if self.log is not None:
                    import traceback as _tb
                    self.log({"kind": "after_step_error", "req": req, "n": prev,
                              "error": _tb.format_exc()})
            # This call boundary is after the previous model forward and its KV commit, but before
            # the current chunk starts. A prompt checkpoint saved here names an actually executed
            # frontier; session hits never combine release-time state with a shorter prompt hash.
            try:
                self._maybe_capture_session_checkpoint(req, self.state(req))
            except Exception:
                if self.log is not None:
                    import traceback as _tb
                    self.log({"kind": "session_checkpoint_capture_error", "req": req,
                              "error": _tb.format_exc()[-400:]})
        self._prev_n[req] = n_new
        st.tokens = max(st.tokens, getattr(st, "tokens", 0))
        orch.note_tokens(st.tokens)
        try:
            # Route A compaction runs during DECODE: the pool (B+R) fills as the output grows, and
            # the reselect rewrites the block table + compacted positions. That rewrite is safe
            # because the three-book channel (BlockChannel.compact) makes EngineCore truncate
            # `_all_token_ids` and rewind `num_computed_tokens` synchronously before the next
            # schedule, so the worker and engine stay reconciled (the old mid-decode failure,
            # `assert num_scheduled_tokens >= num_logits`, came from the engine keeping a monotone
            # sequence the worker had already compacted). Prefill steps are deferred: the prompt is
            # already entry-selected to <= B, and an oversized prefill is bounded by the separate
            # prefill-pressure path below. A step that carries ANY draft tokens is also deferred
            # (D5): speculative logits are addressed relative to the query end, so the block table
            # must not move underneath them; the page move drains at the next draft-free step
            # (<= 1 wasted spec round). Sync scheduling (forced when inrequest_compaction is on)
            # keeps every decode step self-contained, so concurrent decode batches are fine.
            prefill_step = int(n_new) >= self.page_tokens
            has_drafts = req in (getattr(self, "spec_reqs", None) or ())
            deferred = has_drafts or prefill_step
            if deferred:
                self._deferred = int(getattr(self, "_deferred", 0)) + 1
            if (not deferred and not bool(getattr(self, "sched_compact", False))
                    and st.tokens > 0 and orch.should_reselect()):
                # Core-boundary mode owns every compaction: the step wrapper triggers
                # `quiescent_compact` (pool pressure + final reselect). A socket compaction here
                # would move the row without the EngineCore books, which is exactly the double
                # path measured on the 169,998-token run (36 socket compactions while
                # sched_compact was on).
                self._reselect(req)
        except Exception:
            if self.log is not None:
                import traceback as _tb
                self.log({"kind": "reselect_error", "req": req, "tokens": st.tokens,
                          "error": _tb.format_exc()})
        logged = self._adv_logged
        if self.log is not None and req not in logged \
                and st.tokens >= max(1, int(getattr(orch, "gen_reserve", 16384) or 16384)):
            logged.add(req)
            self.log({"kind": "advance_seen", "req": req, "tokens": st.tokens,
                      "row": len(st.row), "R": getattr(orch, "gen_reserve", None),
                      "budget": orch.reselect_token_budget,
                      "planner": orch.planner is not None,
                  "plan": getattr(orch, "last_plan", None),
                      "due": bool(orch.should_reselect())})
        return st

    def _own_pages(self, req):
        """Every device page the request's OWN engine blocks already hold, in row order.

        The engine allocates 1616-token blocks (101 pages of 16 tokens) out of a ~117-block pool; a
        runtime that takes fresh pages from the top of that pool is stealing pages the allocator
        still considers free. Under N=2 long requests the stolen set and the engine's own
        allocations overlap: the engine hands the same pages to the second request (needle
        failures / gibberish) and the pool is accounted twice (stall, then a full re-prefill).
        Compacting INSIDE the request's own pages keeps the two books identical and adds no memory.
        """
        st = self._state.get(req)
        if st is None or st.row_index is None:
            return []
        try:
            row = self.view.block_table_rows()[0][int(st.row_index)]
            values = [int(v) for v in row.tolist()]
        except Exception:
            return []
        pages = [v for v in values if v > 0]
        # The quantum trim below is about the ENGINE's allocation: it appends a whole block
        # (pages_per_block pages) at a time, so a partially filled block must not be treated as
        # owned. A COMPACTED row is our own packed layout, where the tail page may sit at any
        # rank - trimming it there throws away the newest page (the one the next token needs).
        if st.row:
            return pages
        quantum = int(self.pages_per_block)
        if quantum > 1 and pages:
            rem = len(pages) % quantum
            if rem:
                pages = pages[: len(pages) - rem]
        return pages

    def _grow_from_engine(self, req, grown, logical):
        """Publish the slot the ENGINE holds at the row position being built.

        The block table is the only authority for which physical page the next token writes
        into: below the installed prefix it carries OUR packed ids, beyond it the engine's own
        appends. The stale `_page_map` (dense ids for positions the cursor already passed) must
        never answer for a post-compaction rank - measured on the 164,918-byte cold run, that
        is how released (zero) pages were committed into the store and the compacted view read
        back all-zero K/V. Returns the page id, or None when the engine has not published it.
        """
        try:
            rows = self.view.block_table_rows()[0]
            st = self._state[req]
            row = rows[st.row_index] if st.row_index < len(rows) else None
            if row is None:
                return None
            pos = len(grown)
            if pos >= len(row):
                return None
            slot = int(row[pos])
            if slot <= 0:
                return None
            self._page_map.setdefault(req, {})[logical] = slot
            return slot
        except Exception:
            return None

    def _logical_for_rank(self, st, rank):
        """The logical page held at compact `rank` for the tokens being committed now.

        A token at compact position c carries logical position c + D, where D is the executed
        frontier minus the resident length; whole pages cancel the residue (visible keeps the
        prompt's residue, the offset is page-aligned), so `logical = rank + D // page_tokens`
        is exact. Before the first compaction D = 0 and rank == logical.
        """
        d = int(getattr(st, "tokens", 0)) - int(getattr(st, "seq_len", 0))
        return int(rank) + max(0, d) // max(1, self.page_tokens)

    def _sweep_missing_pages(self, req, orch, st):
        """Re-commit prefill pages the lagged per-step commit could not map yet.

        Only valid while the request is still DENSE (`st.tokens == st.seq_len`): then the page
        map names the engine's true-position pages and their bytes have not moved. Once ANY
        compaction has packed the row (tokens > seq_len), the map's ids point at PRE-PACK
        physical positions - pages the boundary protocol may already have released and
        recycled - and sweeping from them writes all-zero pages into the canonical store
        (measured: needle pages captured as sha256-of-zeros on the 164,918-byte run). After a
        compaction the only safe commit source is the LIVE engine table at the row position,
        which is what the per-step commit branch does. Returns the number of pages committed.
        """
        if int(getattr(st, "tokens", 0)) > int(getattr(st, "seq_len", 0)):
            return 0
        committed = 0
        try:
            map_now = self._page_map.setdefault(req, {})
            known = orch.store.pages
            upto = -(-int(st.tokens) // self.page_tokens)
            for lp in range(upto):
                if lp in known:
                    continue
                slot = map_now.get(lp)
                if slot is None:
                    continue
                try:
                    orch.commit_written_page(lp, int(slot), baked=0)
                    committed += 1
                except Exception:
                    continue
        except Exception:
            pass
        return committed

    def after_step(self, req, n_new_tokens, prefill=False):
        self._ensure(req)
        if (bool(getattr(self, "sched_compact", False))
                and bool(getattr(self._state[req], "applied_view_coord", False))
                and int(self._decode_probe.get(req, 0) or 0) < 2):
            # GPU truth probe (measured once per request, first three decode steps after the
            # engine entered view coordinates): does the engine's OWN device book stay rewound,
            # and does the kernel really write new tokens at the row tail? If the feed pushes
            # `num_computed` back to the logical frontier, positions run in logical coordinates
            # while our K pages were rotated to VIEW ranks - a ~52K phase gap that explains the
            # exact observed signature (even an unrotated sink needle unreadable).
            self._decode_probe[req] = int(self._decode_probe.get(req, 0) or 0) + 1
            try:
                runner = getattr(self.view, "runner", None)
                rs = getattr(runner, "req_states", None)
                idx = None
                try:
                    idx = int(rs.req_id_to_index[req])
                except Exception:
                    idx = None
                book_np = book_gpu = None
                if rs is not None and idx is not None:
                    try:
                        book_np = int(rs.num_computed_tokens_np[idx])
                        book_gpu = int(rs.num_computed_tokens.gpu[idx].item())
                    except Exception:
                        pass
                wslot = None
                wcol = None
                ncols = None
                try:
                    bt = runner.block_tables
                    grp = self.view.attn_groups[0]
                    smap = bt.slot_mappings[grp]
                    q = int(self._slot_span_prev.get(req, (0, 0))[1]) or int(self._slot_span.get(req, (0, 0))[1])
                    if q > 0:
                        vals = smap[max(0, q - 4):q].tolist()
                        wslot = [int(v) // int(self.page_tokens) for v in vals if int(v) >= 0]
                        # THE COLUMN each write used: slot -> which entry of the table row the
                        # kernel expanded it from (`_compute_slot_mappings_kernel` reads
                        # row[pos // kernel_block]). A write COLUMN past the adopted row length
                        # proves the engine table still runs on the pre-adopt geometry.
                        idx = None
                        try:
                            idx = int(bt.num_blocks.np[grp, rs.req_id_to_index[req]])
                        except Exception:
                            idx = None
                        ncols = idx
                        row_t = bt.block_tables[grp].gpu[rs.req_id_to_index[req]]
                        # col = (slot // page) % 49 within its block, block = slot//(page*49):
                        wcol = [(int(v) // (int(self.page_tokens) * int(getattr(self, "block_pages", 49) or 49)))
                                for v in vals if int(v) >= 0]
                except Exception:
                    pass
                if self.log is not None:
                    self.log({"kind": "decode_book_probe", "req": req,
                              "book_np": book_np, "book_gpu": book_gpu,
                              "seq_len": int(self._state[req].seq_len),
                              "tokens": int(self._state[req].tokens),
                              "row": len(self._state[req].row or ()),
                              "num_blocks": ncols, "write_blocks": wcol,
                              "write_pages": wslot})
            except Exception:
                pass
        # A step that schedules a whole page worth of tokens is a prefill chunk, not a decode
        # step (a decode step is 1 token, or a few with speculative decoding).
        frontier = int(self._pending_frontier.get(req, 0) or 0)
        if frontier:
            # Trust the scheduler's absolute frontier; derive this step's delta from it.
            _st = self._state[req]
            _st.tokens = max(_st.tokens, frontier - int(n_new_tokens or 0))
        logical_c = self._engine_logical.get(req)
        if logical_c is not None and logical_c > 0:
            # The SERVED count published by the PREVIOUS schedule describes the frontier BEFORE
            # the chunk committed here. Treating it as if it already included this chunk and
            # subtracting `n_new` again made the executed book lag one whole chunk (measured on
            # the 169,998-token TP2 run: worker 145,088 vs Core resident 153,280 at the first
            # pressure compaction). The commit below appends this chunk's tokens.
            self._state[req].tokens = max(int(self._state[req].tokens), int(logical_c))
        engine_c = self._engine_computed.get(req)
        if engine_c is not None and engine_c > 0 and not self._state[req].row:
            # RESIDENT book for a still-dense prefix. V2's CPU `num_computed_tokens_np` is the
            # sampler's LOGICAL mirror, not the resident KV length, so it must never overwrite a
            # compacted `seq_len`: a compaction keeps `row`/`seq_len` and the commit step advances
            # both together.
            self._state[req].seq_len = max(int(self._state[req].seq_len), int(engine_c))
        pending = int(self._pending_new.get(req, 0) or 0)
        if pending > int(n_new_tokens or 0):
            n_new_tokens = pending            # prefill chunks are visible only via query_start_loc
        prefill = prefill or n_new_tokens >= self.page_tokens
        st = self._state[req]
        orch = self._orch[req]

        if not st.row and n_new_tokens:
            # Before the first reselect the engine writes at TRUE positions: the page holding
            # token p is logical page p // page_tokens and its slot is the same id (delta 0).
            # Absorb the engine's row ONCE per step, not per token: `row.tolist()` inside the loop
            # made a long prefill quadratic (53K tokens x a 3.4K-entry row).
            self._absorb_engine_row(req)
            refused = []
            for i in range(n_new_tokens):
                logical = (st.tokens + i) // self.page_tokens
                if logical in orch.store.pages:
                    continue
                # The engine allocated this page itself; use ITS id (never a self-built pool,
                # which exhausted on long runs and does not own page ids).
                slot = self._page_map[req].get(logical)
                if slot is None:
                    continue
                # No view is installed yet, so the engine wrote these tokens at their TRUE
                # positions: the page's bytes carry no offset from its logical start.
                if not orch.commit_written_page(logical, slot, baked=0):
                    refused.append((int(logical), int(slot)))
            if refused and self.log is not None:
                self.log({"kind": "commit_refused_zero", "req": req, "dense": True,
                          "count": len(refused), "sample": refused[:8],
                          "tokens": int(st.tokens), "seq_len": int(st.seq_len)})
            st.tokens += n_new_tokens
            orch.note_tokens(st.tokens)
        elif st.row and n_new_tokens:
            self._absorb_engine_row(req)
            # Streaming tokens (decode, or a prefill continuing past a pressure reselect) land at
            # compacted positions starting at st.seq_len. A newly crossed page needs a fresh slot
            # appended to the row; the spare slot taken at reselect is the first of those.
            start = st.seq_len
            refused = []
            # A newly crossed page needs a slot appended to the row. This is correct for BOTH
            # prefill and decode (a decode step normally consumes the spare slot, so it is a
            # no-op there); gating it on the prefill flag broke streaming after a pressure
            # reselect made the row non-empty.
            last_rank = (start + n_new_tokens - 1) // self.page_tokens
            if len(st.row) <= last_rank:
                self._absorb_engine_row(req)
                grown = list(st.row)
                while len(grown) <= last_rank:
                    # The row may be PACKED (rank != logical after a compaction): the slot must
                    # come from the ENGINE's table at the position being built - never from the
                    # stale dense `_page_map`, whose ids for passed positions name released pages.
                    page = self._grow_from_engine(req, grown,
                                                  self._logical_for_rank(st, len(grown)))
                    if page is None:
                        break
                    grown.append(page)
                st.row = tuple(grown)
            touched = {}
            _sm_span = getattr(self, "_slot_span_prev", {}).get(req)
            _smap = self.slot_pages_for(req) if _sm_span else None
            for i in range(n_new_tokens):
                logical = (st.tokens + i) // self.page_tokens
                rank = (start + i) // self.page_tokens
                if rank >= len(st.row):
                    # The spare slot handed out by the last reselect is used up: the engine has
                    # appended the next page in ITS table at exactly this row position (below the
                    # prefix the table carries our installed packed ids, so both agree). Only if
                    # the engine has not published it is the row genuinely out of space.
                    grown = list(st.row)
                    while len(grown) <= rank:
                        page = self._grow_from_engine(req, grown,
                                                      self._logical_for_rank(st, len(grown)))
                        if page is None:
                            raise RuntimeError("sparse row has no slot for the next token page")
                        grown.append(int(page))
                    st.row = tuple(grown)
                # The ENGINE's table is the authority for where the previous forward wrote:
                # `adopt_attention_blocks` rewrote the manager list and the engine gathers it
                # over the GPU tensor at every input prep. Our own `st.row` tail (the reselect
                # "rest" pages) is a mirror that can be stale exactly where the engine appended
                # (measured: bytes at physical 1176 while st.row[1176]=1372 was all zero).
                tb_val = None
                slot_src = "row"
                page = None
                # Priority: the KERNEL's slot mapping for the step under commit
                # (authoritative - it IS the address the kernels used to write the bytes we
                # are capturing), then the staging table mirror at rank, then the row.
                if _smap and _sm_span:
                    _q = int(_sm_span[0]) + max(0, logical * self.page_tokens - int(st.tokens))
                    if _q in _smap:
                        page = _smap[_q]
                        slot_src = "slotmap"
                if page is None:
                    try:
                        _tbl = self.view.block_table_rows()[0]
                        tb_val = int(_tbl[st.row_index][rank])
                        page = tb_val if tb_val > 0 else int(st.row[rank])
                        slot_src = "stage1" if (tb_val and tb_val > 0) else "row"
                    except Exception:
                        page = int(st.row[rank])
                slot = int(page)
                touched[logical] = (slot, rank, tb_val, slot_src)
            _tp = getattr(self, "trace_pages", None)
            if _tp and self.log is not None and any(lp in _tp for lp in touched):
                # Decisive commit-time probe: does the device page we are about to capture carry
                # bytes? Compare the chosen slot against the engine table at RANK and at LOGICAL
                # position, with each candidate page's live byte-sum.
                try:
                    tbl = self.view.block_table_rows()[0][st.row_index]
                    tb_list = tbl.tolist()
                    dev = list(self._orch[req].kvs)
                    rows = []
                    for lp in _tp:
                        if lp not in touched:
                            continue
                        s, r = touched[lp][0], touched[lp][1]
                        def dsum(page):
                            if page is None or int(page) <= 0 or int(page) >= len(dev[0]):
                                return None
                            return int(dev[0][int(page)].view(
                                __import__("torch").uint8).to("cpu").numpy().astype("int64").sum())
                        rows.append({"logical": int(lp), "rank": int(r), "slot": int(s),
                                     "slot_src": (str(touched[lp][3])
                                                  if len(touched[lp]) > 3 else "row"),
                                     "start": int(start), "tokens": int(st.tokens),
                                     "seq_len": int(st.seq_len),
                                     "d_pages": (int(st.tokens) - int(st.seq_len)) // self.page_tokens,
                                     "eng_at_rank": tb_list[r] if r < len(tb_list) else None,
                                     "eng_at_logical": tb_list[lp] if lp < len(tb_list) else None,
                                     "row_len": len(st.row), "dev_sum_slot": dsum(s),
                                     "dev_sum_eng_logical": dsum(tb_list[lp]) if lp < len(tb_list) else None})
                    # Where did the engine ACTUALLY write? One reduction over the whole layer-0
                    # cache. If the non-zero pages cluster around a TRUE-position mapping instead
                    # of the packed row, the kernel's write address ignored the compaction.
                    try:
                        import torch as _t
                        sums = dev[0].view(_t.uint8).to(_t.int16).sum(dim=(1, 2, 3))
                        sums_cpu = sums.to("cpu").tolist()
                        nz_pages = sum(1 for v in sums_cpu if v > 1000)
                        first_nz = [i for i, v in enumerate(sums_cpu) if v > 1000][:8]
                        probe_page = next((x for x in _tp if x in touched), None)
                        s0 = int(touched[probe_page][0]) if probe_page is not None else 0
                        want_at = {}
                        for name, pos in (("logical_plus_base", probe_page + 196),
                                          ("slot", s0), ("rank", int(touched[probe_page][1])
                                                          if probe_page else 0)):
                            if 0 <= pos < len(sums_cpu):
                                want_at[name] = sums_cpu[pos]
                        # Does the DEVICE page holding real bytes sit where the DENSE mapping
                        # would put the chunk? Search a window around the written token span.
                        lo = max(0, int(st.seq_len) // self.page_tokens - 8)
                        window = sums_cpu[lo:lo + 200]
                        self.log({"kind": "commit_map_probe", "req": req,
                                  "nonzero_pages": int(nz_pages),
                                  "first_nonzero": first_nz, "row_len": len(st.row),
                                  "seq_len": int(st.seq_len), "tokens": int(st.tokens),
                                  "want_at": want_at,
                                  "tail_row_pages_sum": [sums_cpu[p] for p in
                                                         list(st.row[lo:lo + 8])]})
                    except Exception:
                        self.log({"kind": "commit_map_probe_error", "req": req,
                                  "error": __import__("traceback").format_exc()[-200:]})
                    self.log({"kind": "commit_probe", "req": req, "rows": rows})
                except Exception:
                    import traceback as _tb
                    self.log({"kind": "commit_probe_error", "req": req,
                              "error": _tb.format_exc()[-300:]})
            _view_basis = bool(getattr(st, "applied_view_coord", False))
            for logical, tup in touched.items():
                slot, rank = int(tup[0]), int(tup[1])
                # PHASE BASIS. After a FINAL apply (book rewind landed) the engine bakes at
                # VIEW coordinates, so the page carries `(rank - logical) * page_tokens` of
                # rotation. After a PRESSURE apply - or with no apply at all - the engine's book
                # still runs on the logical frontier (the feed never rewinds it for a running
                # request, 891a2cf) and the engine's bake is ALREADY the logical phase: delta 0.
                # Passing the rank-based delta during a pressure epoch stored 124K tokens of
                # wrongly-rotated pages - the streamed commits survive a later repack, so the
                # poison reached the final view (measured: 891a2cf run, ArchiveArchive).
                _vb = bool(getattr(st, "applied_view_coord", False))
                if not orch.commit_written_page(
                        logical, slot,
                        baked=0 if (self._pure_copy() or not _vb)
                        else (rank - logical) * self.page_tokens):
                    refused.append((int(logical), int(slot), int(rank)))
                if st.spare == slot:
                    st.spare = None
            if refused and self.log is not None:
                # The zero guard converted store poisoning into MISSING pages: the page is absent,
                # the selection cannot offer it, and Core (rightly) refuses a view without the
                # tail. This is the live signature of "commit reads a page the engine wrote
                # elsewhere" - the sample gives logical/slot/rank to compare with the map probe.
                entry = {"kind": "commit_refused_zero", "req": req, "dense": False,
                         "count": len(refused), "sample": refused[:8],
                         "tokens": int(st.tokens), "seq_len": int(st.seq_len),
                         "tb_at": [touched[lg][2] for lg, _, _ in refused[:4]
                                   if lg in touched and len(touched[lg]) > 2],
                         "row_at": [int(st.row[rr]) if rr < len(st.row) else None
                                    for _, rr, _ in [tuple(x) for x in refused[:4]]]}
                try:
                    import torch as _t
                    dev0 = list(self._orch[req].kvs)[0]
                    sums = dev0.view(_t.uint8).to(_t.int32).sum(dim=(1, 2, 3)).to("cpu").tolist()
                    entry["nonzero_pages"] = int(sum(1 for v in sums if v > 1000))
                    want = {}
                    for lg_, sl_, rk_ in refused[:4]:
                        want[f"slot_{lg_}"] = sums[sl_] if sl_ < len(sums) else None
                        want[f"rank_{lg_}"] = sums[rk_] if rk_ < len(sums) else None
                    entry["sums_at_refused"] = want
                except Exception:
                    pass
                self.log(entry)
            st.tokens += n_new_tokens
            st.seq_len += n_new_tokens
            orch.note_tokens(st.tokens)

        st.steps += 1
        orch.tick()
        if getattr(self, "_finishing_reconstruction", False):
            return st
        if not prefill and not st.prefill_done and st.tokens:
            # First decode step: anchor the R cadence at the prompt length. Without this the first
            # R check is already true (the logical total covers the prompt) and every request would
            # reselect before generating a single token; with it, reselection fires after R OUTPUT
            # tokens, exactly when the pool (B history + R output) is full.
            st.prefill_done = True
            orch._tokens_at_reselect = max(0, st.tokens - n_new_tokens)
            if (not self.compact_rebuild and not bool(getattr(self, "sched_compact", False))
                    and st.row and not st.query_reselected
                    and getattr(orch, "retrieval_index", None) is not None):
                # The prompt was compacted during prefill (no query existed then, so only
                # sink+recent survived). Now the capture bridge holds the real query: reselect
                # once with retrieval so the middle can come back before generation starts.
                st.query_reselected = True
                self.log({"kind": "query_reselect", "req": req, "tokens": int(st.tokens),
                          "prompt": int(self._prompt_len.get(req, 0) or 0)})
                self._reselect(req, probe=True)
                # QueryReplay: the tail was prefilled while the middle was evicted, so re-schedule
                # it now that the working set holds the real middle.
                self.send_replay(req, int(getattr(self, "replay_tokens", 2048) or 2048))
            # The prompt is IMMUTABLE for this backend: vLLM derives the output length from
            # `num_tokens - prompt_len` and requires `_all_token_ids` to cover the prompt, so a
            # compaction may only ever drop GENERATED tokens. Remember the prefill length here so
            # every reselect can declare the prompt pages mandatory.
            orch.prompt_tokens = max(0, st.tokens - n_new_tokens)
        # Freeze the query once the prompt is complete: the last prefill chunk captured the
        # question, and subsequent decode steps must not replace it (KVMem's frozen attachment).
        try:
            _idx = self._retrieval.get(req)
            _pl = int(self._prompt_len.get(req, 0) or 0)
            if (_idx is not None and _pl and not prefill and int(st.tokens) >= _pl
                    and not getattr(_idx, "_query_frozen", False)):
                _idx.freeze_query()
        except Exception:
            pass
        # Resident model: a prompt LONGER than B is the normal case now - it is prefilled in
        # chunks and the working set is compacted as it grows (the logical book keeps every token,
        # the resident book stays <= B+R). Log the shape once during diagnosis.
        prompt_len = int(self._prompt_len.get(req, 0) or 0)
        if (prompt_len > int(self.budget_tokens) and req not in self._over_budget_logged):
            self._over_budget_logged.add(req)
            self.log({"kind": "long_prompt_chunked", "req": req, "prompt_tokens": prompt_len,
                      "budget_tokens": int(self.budget_tokens),
                      "note": "prefill runs in chunks over the resident window"})
        # During a long prefill the resident set must stay bounded even though relevance is not
        # meaningful yet: keep the sink prefix + newest tail and spill the middle (kvmem's
        # prefill-pressure selection). Without this a 200K prefill would consume the whole pool.
        if (prefill and st.tokens and orch.selector is not None
                and bool(getattr(self, "inrequest_compaction", False))):
            if self.no_prefill_compact:
                # Resident prefill: the prompt is written into the WHOLE physical pool with no
                # mid-prefill compaction (that path is not reliable). The borrowed pool is given
                # back when the LAST chunk completes the prompt - and it must happen HERE, inside
                # a PREFILL step, never on the first decode step: that step carries speculative
                # drafts, and moving the block table underneath the verifier desynchronises the
                # two books. Measured on the first decode step: the request stalled with
                #   kvmem: prompt served but no sampled token  logical=43800 computed=36864
                # and the scheduler then re-classified the next step as a prefill chunk.
                #
                # A prefill step is safe: it writes at true positions (no drafts in flight), and by
                # the time `after_step` reaches this branch `st.tokens` already covers the chunk and
                # its pages are committed to the host store, so `_reselect` selects only pages that
                # really carry bytes.
                _plen = int(self._prompt_len.get(req, 0) or 0)
                # Byte probe at prompt completion, independent of any shrink decision: this is the
                # one moment both the baseline run and the compaction run have the FULL prompt
                # committed, so the same logical page can be fingerprinted and compared.
                if (_plen and int(st.tokens) >= _plen
                        and list(getattr(self, "trace_pages", ()) or ())
                        and req not in getattr(self, "_prompt_end_traced", set())):
                    self._prompt_end_traced = getattr(self, "_prompt_end_traced", set())
                    self._prompt_end_traced.add(req)
                    self._trace_pages(req, list(self.trace_pages))
                if (_plen and int(st.tokens) >= _plen and not st.query_reselected
                        and not bool(getattr(self, "sched_compact", False))):
                    # Socket shrink is disabled in Core-boundary mode: the step wrapper's final
                    # trigger performs the same query-driven reselect with cross-rank consensus,
                    # under the EngineCore books. Leaving this ungated produced a second,
                    # book-free compaction path (measured: 36 applied / 12 query_reselect while
                    # sched_compact was on).
                    try:
                        _rp = len(getattr(orch, "resident", ()) or ())
                        _bp = int(self.budget_tokens) // max(1, int(self.page_tokens))
                    except Exception:
                        _rp, _bp = 0, 0
                    if _rp > _bp:
                        st.query_reselected = True
                        self.log({"kind": "post_prefill_shrink", "req": req,
                                  "tokens": int(st.tokens), "prompt": _plen,
                                  "resident_pages": int(_rp), "budget_pages": int(_bp)})
                        try:
                            self._reselect(req, probe=True)
                            self.send_replay(req, int(getattr(self, "replay_tokens", 2048) or 2048))
                        except Exception:
                            if self.log is not None:
                                import traceback as _tb
                                self.log({"kind": "post_prefill_shrink_error", "req": req,
                                          "error": _tb.format_exc()[-500:]})
                return st
            high = self.prefill_high_watermark
            # Hard cap at the comparison itself: the pool ceiling minus one chunk. The watermark is
            # (re)assigned in several places (init, after each compaction), so capping only at the
            # assignment left the trigger above the ceiling and the prefill stopped with the pool dry
            # (measured: row 5,656 pages = 90,496 tokens, free_blocks=9, 0 compactions).
            if int(getattr(self, "pool_ceiling_tokens", 0) or 0) > 0:
                high = min(high, pool_prefill_high_watermark(
                    self.pool_ceiling_tokens, self.page_tokens))
            finishing = False  # query reselect happens on the first decode step (see below)
            if (finishing and st.row and not st.query_reselected
                    and getattr(orch, "retrieval_index", None) is not None):
                # The prompt is about to be complete, so the question is inside the captured
                # query window: reselect ONCE with retrieval now. Without this a prompt that was
                # compacted during prefill keeps only sink + recent (no query existed yet), the
                # middle never comes back on a short generation, and the answer is invented from
                # the tail (measured: KEY-MID recited with the END key's value).
                st.query_reselected = True
                self.log({"kind": "query_reselect", "req": req, "tokens": int(st.tokens),
                          "prompt": prompt_len, "n_new": int(n_new_tokens or 0)})
                self._reselect(req, probe=True)
            elif (not bool(getattr(self, "sched_compact", False))
                  and prefill_pressure_due(
                      st.tokens, max(len(orch.resident), len(st.row)), high,
                      getattr(orch, "_tokens_at_reselect", 0), self.gen_reserve)):
                # The ROW is the true residency: pages the engine appended since the last
                # compaction are in the row immediately (the cursor sync copies them), while
                # `orch.resident` only knows the set the last compaction installed. Comparing
                # against the stale set let residency climb unchecked until the physical pool ran
                # out (measured: row 11,008 pages = 176,128 tokens on a 179,182-token pool, then
                # the next allocation faulted).
                try:
                    self._reselect(req, prefill_pressure=True)
                    if int(st.tokens) > int(st.seq_len):
                        st.compacted_in_prefill = True
                except Exception:
                    if self.log is not None:
                        import traceback as _tb
                        self.log({"kind": "prefill_reselect_error", "req": req,
                                  "error": _tb.format_exc()[-500:]})
                # Do not re-fire on the very next page: the reselect just rebuilt the working set,
                # so the next useful one is a reserve later (R tokens = 2K pages at B=53K). Without
                # this every prefill chunk compacted - 24 reselects on a 200K prompt, each moving
                # ~3.3K pages (measured: prefill fell to ~300 tok/s, 9x slower).
                self.prefill_high_watermark = max(len(orch.resident), len(st.row)) + max(
                    64, int(self.gen_reserve) // max(1, self.page_tokens))
                if int(getattr(self, "needle_trace", 0)) > 0:
                    resident = sorted(orch.resident)
                    self._trace_pages(req, resident[:2] + resident[-2:])
            return st
        due = orch.should_reselect()
        if self.log is not None and req not in self._adv_logged and st.tokens > 10000:
            self._adv_logged.add(req)
            self.log({"kind": "reselect_check", "req": req, "tokens": st.tokens,
                      "last": orch._tokens_at_reselect, "R": getattr(orch, "gen_reserve", None),
                      "budget": orch.reselect_token_budget,
                      "planner": orch.planner is not None,
                  "plan": getattr(orch, "last_plan", None), "due": bool(due)})
        # Compaction is DECODE-only and TRUNCATION-driven: reselect when the R cadence is due, which
        # is exactly when the pool (B history + R output) is full and the visible set must shrink.
        # Building an "initial view" on the first decode step is NOT done: it truncates nothing
        # (visible == the whole book) and installing a compacted row anyway desynchronised the two
        # books - measured as an 8K prompt stalling right after prefill while the same prompt
        # completed with compaction off.
        # A step that carries speculative drafts must NOT be compacted underneath the verifier:
        # vLLM places the logits rows at [query_end - (drafts + bonus), query_end) and the scheduler
        # counts the drafts in `num_computed_tokens`, so rewinding the resident book in the middle of
        # a spec round leaves the two books disagreeing by the draft count. Measured with R=2048
        # (compaction fires early): the next step was classified as a PREFILL chunk
        # (`num_computed < num_tokens`) and the engine died on
        #   FULL CUDA graph metadata topology mismatch ... num_prefills: runtime=1, captured=0,
        # after writing KV at positions that overlapped the previous step. The reselect drains on
        # the next draft-free step instead.
        has_drafts = req in (getattr(self, "spec_reqs", None) or ())
        if (bool(getattr(self, "inrequest_compaction", False)) and not prefill and due
                and not bool(getattr(self, "sched_compact", False))):
            if has_drafts:
                if req not in getattr(self, "_deferred_drafts", set()):
                    self._deferred_drafts = getattr(self, "_deferred_drafts", set())
                    self._deferred_drafts.add(req)
                    if self.log is not None:
                        self.log({"kind": "reselect_deferred", "req": req,
                                  "tokens": int(st.tokens), "seq_len": int(st.seq_len),
                                  "reason": "spec drafts in flight"})
                return st
            self._deferred_drafts = getattr(self, "_deferred_drafts", set())
            self._deferred_drafts.discard(req)
            try:
                result = self._reselect(req)
            except Exception:
                if self.log is not None:
                    import traceback as _tb
                    self.log({"kind": "reselect_error_soft", "req": req,
                              "error": _tb.format_exc()[-500:]})
                return st
            if int(getattr(self, "needle_trace", 0)) > 0:
                resident = sorted(orch.resident)
                self._trace_pages(req, resident[:2] + resident[-2:])
            return result
        return st

    def _page_fingerprint(self, tensors):
        """Digest of a page's raw bytes (sha256 over all layers). Cross-run byte equality is the
        only way to validate the CANONICAL store bytes: the device-vs-store check in `needle()` is
        self-consistent (both sides come from the same wrong slot if the capture read one), so it
        cannot prove the capture was right. A baseline run (no compaction, correct answer) and the
        compaction run must fingerprint the same logical pages identically."""
        try:
            import hashlib
            import torch
            h = hashlib.sha256()
            for t in tensors:
                h.update(t.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
            return h.hexdigest()[:16]
        except Exception:
            return None

    def _trace_pages(self, req, pages):
        """Diagnostic: for a few pages, show every number the needle depends on."""
        orch = self._orch.get(req)
        st = self._state.get(req)
        if orch is None or st is None or self.log is None:
            return
        rows = []
        for page in pages[:8]:
            slot = orch.slot_of.get(page)
            dig_store = dig_dev = None
            try:
                if orch.store.has(page):
                    dig_store = self._page_fingerprint(orch.store.get(page))
                if slot is not None:
                    dig_dev = self._page_fingerprint([kv[int(slot)] for kv in orch.kvs])
            except Exception:
                pass
            rows.append({"page": int(page),
                         "engine_page": int(self._page_map.get(req, {}).get(page, 0)),
                         "compacted_slot": (int(st.row.index(page)) if page in st.row else None),
                         "slot_of": int(orch.slot_of.get(page, -1)),
                         "store_delta": int(orch.store_delta.get(page, 0)),
                         "baked_delta": int(orch.baked_delta.get(page, 0)),
                         "store_sha": dig_store, "device_sha": dig_dev,
                         "needle": orch.needle(page)})
        self.log({"kind": "needle_trace", "req": req, "tokens": st.tokens, "rows": rows})

    def send_replay(self, req, n_tokens=2048):
        """QueryReplay: ask EngineCore to re-schedule the last N tokens against the new view.

        Only meaningful right after a query-driven reselect: the working set now holds the real
        middle, and those N tokens were computed while it did not. Returns the engine's reply or
        None (disabled / nothing to do / error), never raises.
        """
        ch = getattr(self, "compact_channel", None)
        if ch is None or not hasattr(ch, "replay"):
            return None
        try:
            st = self.state(req)
            visible = int(getattr(st, "seq_len", 0) or 0)
            snap = getattr(st, "replay_snap", None)
            at = int(getattr(st, "replay_at_resident", 0) or 0)
            # Resident coordinates, and only within one bounded window: a snapshot taken many
            # compactions ago cannot be restored into the current view.
            n = (visible - at) if (snap and at > 0) else 0
            if n <= 0 or n > int(getattr(self, "replay_max_tokens", 4096) or 4096):
                if self.log is not None:
                    self.log({"kind": "query_replay_skipped", "req": req,
                              "visible": visible, "at_resident": at, "n": int(n),
                              "reason": "no valid resident boundary"})
                return None
            n = min(int(n), visible)
            if n <= 0:
                return None
            if snap is not None and self.restore_state is not None:
                # Put the recurrent state back at the replay boundary first: the tokens about to be
                # recomputed are then applied exactly once.
                try:
                    self.restore_state(req, snap)
                except Exception:
                    pass
                st.replay_snap = None
                if self.log is not None:
                    self.log({"kind": "replay_state_restored", "req": req, "n": int(n)})
            res = ch.replay(req, visible, n)
            if self.log is not None:
                self.log({"kind": "query_replay", "req": req, "n": int(n),
                          "visible": visible, "reply": res})
            return res
        except Exception:
            if self.log is not None:
                import traceback as _tb
                self.log({"kind": "query_replay_error", "req": req,
                          "error": _tb.format_exc()[-400:]})
            return None

    def quiescent_compact(self, req, frontier=None, computed=None, final=None):
        """EngineCore-callable compaction on the scheduler thread, between steps.

        The worker has completed the previous forward but not yet committed it (the commit runs at
        the next `advance`). This flushes that chunk, checks that the executed frontier still equals
        the one EngineCore scheduled, then moves pages and installs the compacted row WITHOUT the
        BlockChannel call: EngineCore adopts the blocks and rewrites its own books from the returned
        proposal, so no worker-originated scheduler mutation happens off the main thread.

        REFUSAL MUST NOT DIVERGE: the worker moves its view before Core can validate, so every
        pre-move book check runs HERE first (frontier AND resident: `computed` is Core's
        `num_computed_tokens`), and the pre-move state is snapshotted so Core's
        `kvmem_revert_quiescent` RPC can restore it after a refusal. The revert is safe because the
        canonical store holds the real bytes (verified this session: dense commits captured them),
        so re-installing the old view is a pure store->device copy.
        """
        if not req or req not in self._state:
            return None
        pending = int(self._prev_n.get(req, 0) or 0)
        if pending:
            self.after_step(req, pending, prefill=pending >= self.page_tokens)
            self._prev_n[req] = 0
        st = self._state[req]
        if frontier is not None and int(frontier) != int(st.tokens):
            self.log({"kind": "quiescent_compact_stale", "req": req,
                      "want": int(frontier), "have": int(st.tokens)})
            return None
        if computed is not None and int(computed) != int(st.seq_len):
            # The two resident books must agree BEFORE the move; refusing here (instead of after
            # the move) is what keeps a disagreement recoverable (measured: one refused final
            # proposal left the worker at 20,161 while Core kept 54,326 and every later proposal
            # was refused with "frontier differs" for the rest of the request).
            self.log({"kind": "quiescent_compact_stale", "req": req,
                      "want_logical": int(frontier or 0), "have_logical": int(st.tokens),
                      "want_resident": int(computed), "have_resident": int(st.seq_len),
                      "reason": "resident"})
            return None
        self._quiescent_snap = {
            "req": req, "row": st.row, "logical": st.logical, "seq_len": st.seq_len,
            "spare": st.spare, "owned": tuple(self._owned_slots.get(req, ())),
            "resident": set(self._orch[req].resident),
            "slot_of": dict(self._orch[req].slot_of),
            "baked_delta": dict(self._orch[req].baked_delta)}
        self._last_proposal = None
        # Two different core selections: prefill PRESSURE compaction has no usable query yet and
        # the core's pressure picker deliberately ranks by sink/recency; the FINAL reselect (the
        # prompt is complete, the question is the frozen query) must go through `select_pages` ->
        # `pick_topk_blocks`, the only path that reads retrieval scores. Measured: the pressure
        # picker ignored a #1-ranked needle block and filled the middle by recency.
        _pl = int(self._prompt_len.get(req, 0) or 0)
        if final is None:
            pressure = not (_pl and int(st.tokens) >= _pl)
        else:
            # EngineCore owns the final-vs-pressure decision (it already computed `request in
            # final` from `num_logical_tokens >= num_prompt_tokens`). The worker's `_prompt_len`
            # can be missing (the plen row-index hook did not fire), which silently classified
            # every prompt-end reselect as PRESSURE - the recency picker then dropped the actual
            # question (measured: the final selection stopped at the pre-prompt tail page and Core
            # refused it for lacking the last page).
            pressure = not bool(final)
        # Both TP ranks are inside this collective_rpc together: the exact boundary where the
        # score consensus may run. Without it each rank ranked the middle by its OWN shard's
        # scores and proposed different working sets (measured: "TP ranks proposed different
        # working sets" refused the first prompt-end compaction, and the already-compacted local
        # books then made every retry non-shrinking).
        self._in_quiescent = True
        try:
            _pl0 = int(self._prompt_len.get(req, 0) or 0)
            _st0 = self._state.get(req)
            if (int(getattr(self, "compact_rebuild", False)) and _pl0
                    and _st0 is not None and int(_st0.tokens) >= _pl0
                    and req in getattr(self, "_rebuild_queue", {})):
                # Stage 2 at the QUiescent (scheduler-thread) boundary: the prompt is fully
                # resident and committed by the flush above; select over the frozen question
                # and hand Core the rebuilt token list - right here, right thread, so the
                # requeue lands BEFORE the next schedule (pressure would otherwise starve the
                # request at the pool ceiling with the GPU idle, measured at 165K/161K).
                _q = self._rebuild_queue
                _q.pop(req, None)
                _ok, _det = self._rebuild_select(req)
                if not _ok and self.log is not None:
                    _tr = getattr(self, "_rebuild_tries", None)
                    if _tr is None:
                        _tr = self._rebuild_tries = {}
                    _tr[req] = int(_tr.get(req, 0)) + 1
                    if _tr[req] < 6:
                        _q[req] = int(_st0.tokens)   # re-arm next boundary
                    self.log({"kind": "rebuild_select_retry", "req": req,
                              "detail": str(_det)[:200], "tries": _tr[req]})
            mode = getattr(self, "compact_mode", None) or \
                (__import__("os").environ.get("KVMEM_COMPACT_MODE", "") or None)
            if mode == "prefix":
                # Prefix (retirement-report) mode: the worker re-checks its books and the store
                # coverage, but proposes NOTHING to Core - the engine keeps its dense resident
                # window and retires heads through its own R-SWA pass. A report proposal would
                # only loop refusals (`non-shrinking`) and revert traffic every step.
                self._reselect_prefix(req, pressure=pressure)
                self._prefix_active = True
                self._last_proposal = None
                if int(getattr(self, "compact_rebuild", False)):
                    # Stage 2 fires at BOTH boundaries: the prompt-complete final (the only
                    # moment the frozen question exists for retrieval) AND pool pressure
                    # during a long prefill (measured: a 165K prompt stalls the scheduler at
                    # ~91% of the pool, so the final boundary NEVER arrives and the request
                    # starves with the GPU idle - pressure must be able to trigger it too).
                    q = getattr(self, "_rebuild_queue", None)
                    if q is None:
                        q = self._rebuild_queue = {}
                    q.setdefault(req, int(st.tokens))
            elif bool(getattr(self, "blockwise_view", False)):
                self._reselect_blockwise(req, pressure=pressure)
            else:
                _rebuild_mode = bool(mode == "page"
                                     and int(getattr(self, "compact_rebuild", False)))
                self._in_page_rebuild = _rebuild_mode
                try:
                    self._reselect(req, prefill_pressure=pressure, send=False)
                finally:
                    self._in_page_rebuild = False
                if _rebuild_mode:
                    # page mode + rebuild: the worker's own reselect only plans; fire the
                    # stage-2 pack (picker+move+channel) the moment the prompt is complete.
                    _plq = int(self._prompt_len.get(req, 0) or 0)
                    if _plq and int(st.tokens) >= _plq and not bool(
                            getattr(st, "rebuilt", False)):
                        _ok, _det = self._rebuild_select(req)
                        if self.log is not None:
                            self.log({"kind": "page_rebuild_try", "req": req,
                                      "ok": bool(_ok), "detail": str(_det)[:160]})
        finally:
            self._in_quiescent = False
        proposal = getattr(self, "_last_proposal", None)
        if proposal is not None:
            # Byte probe (KVMEM_TRACE_PAGES): fingerprint the canonical store bytes and the device
            # bytes at the slot the proposal just installed. The store-vs-device `needle` check is
            # self-consistent, so a WRONG-SLOT capture would pass it; the sha lets the same page be
            # compared between the baseline (correct) run and the compaction run.
            _tp = list(getattr(self, "trace_pages", ()) or ())
            if _tp and self.log is not None:
                self._trace_pages(req, _tp)
        if proposal is None:
            # Diagnostic: Core saw "incomplete TP compaction proposal" with no field saying why
            # (alignment deferral, empty selection and a refused planner all looked alike).
            # Once per (req, shape) - the pressure trigger re-enters this path every 64 epochs
            # and an unconditional line per entry is a log storm (measured: 366K rows).
            _sig = ("empty", req, int(st.tokens), int(st.seq_len))
            _seen = getattr(self, "_empty_logged", None)
            if _seen is None:
                _seen = self._empty_logged = set()
            if _sig in _seen:
                return None
            _seen.add(_sig)
            self.log({"kind": "quiescent_compact_empty", "req": req,
                      "tokens": int(getattr(st, "tokens", 0) or 0),
                      "seq_len": int(getattr(st, "seq_len", 0) or 0),
                      "row": len(getattr(st, "row", ()) or ()),
                      "aligned": int(st.tokens) % int(self.page_tokens) == 0,
                      "want": None if frontier is None else int(frontier)})
        else:
            self.log({"kind": "quiescent_compact_proposal", "req": req,
                      "visible": proposal.get("visible_tokens"),
                      "pages": len(proposal.get("physical_pages") or ()),
                      "logical": proposal.get("logical_frontier"),
                      "resident": proposal.get("resident_frontier"),
                      "store": proposal.get("store_pages"),
                      "store_max": proposal.get("store_max"),
                      "sel_max": max(proposal.get("selected_pages") or (0,))
                      if proposal.get("selected_pages") else -1})
        return proposal

    def _rebuild_select(self, req):
        """Stage 2 (v6): official selection+pack, then hand Core the rebuilt token list.

        The picker scores the store over the frozen question (the only moment the real query
        exists) and, given REAL destination slots, moves the selection with the official data
        plane: device copies for pages that keep their rotation, one-hop host rebuilds from
        canonical bytes for the rest (rebake ON - this fork has no rope-offset consumer, so
        the view frame must be baked into the bytes). `forced_pages` guarantees the question
        tail is in the set; the whole selection must fit inside the pool the detached request
        just freed, so the rebuilt prompt never outruns the engine.

        Core's rebuild branch turns the logical pages into the rebuilt prompt via
        `select_tokens` (original tokens, ascending = compact order: the engine re-prefills
        exactly the retrieved stream) and re-queues the SAME request id fresh - dense
        coordinates again, every fork law satisfied by construction, the answer already
        generated carried over in the rebuilt stream's decode.

        Returns (ok, detail).
        """
        st = self._state[req]
        orch = self._orch[req]
        pt = int(self.page_tokens)
        bp = max(1, int(getattr(self, "block_pages", 1) or 1))
        ch = getattr(self, "compact_channel", None)
        if ch is None:
            return False, "no block channel"
        ids = self._prompt_ids.get(req)
        if not ids:
            return False, "no prompt ids recorded"
        prompt_tokens = len(ids)
        n_pages = -(-prompt_tokens // pt)
        tail_tokens = max(pt, int(getattr(self, "rebuild_tail_tokens", 0) or 4096))
        tail_pages_n = max(1, tail_tokens // pt)
        tail_first = max(0, n_pages - tail_pages_n)
        forced_tail = list(range(tail_first, n_pages))
        # destination slots: every pool page the engine can hand the rebuilt admission after
        # its blocks are freed, page-ascending; the picker packs 0..k-1 (complete blocks first
        # in `selected` ascending order, exactly what adopt_attention_blocks validates).
        try:
            n_blocks = int(self.view.num_blocks())
        except Exception:
            n_blocks = 0
        if n_blocks <= 0:
            return False, "pool size unavailable"
        # BLOCK-aligned window: vLLM's pool starts with the null block at pages
        # [0..bp-1], so the first usable page is (first_page//bp + 1)*bp and the last is
        # whole-block truncated. A non-aligned destination window makes the packed physical
        # row span partial blocks, which `adopt_attention_blocks` can never validate.
        # vLLM addresses a page by ITS BLOCK INDEX times the pages-per-engine-block; the
        # paged-KV tensor rows are indexed the same way, so the pool window must be built as
        # ascending block ids scaled by `bp` (a bare `range()` here would address sub-pages
        # of the first two blocks and every page above them would alias).
        first = max(1, (-(-int(self.first_page) // bp)))
        last = int(self.view.num_blocks())
        pool_pages = [b * bp + o for b in range(first, last) for o in range(bp)]
        if len(pool_pages) < tail_pages_n + bp:
            return False, f"pool window too small ({len(pool_pages)})"
        # 1) score-ranked selection in the picker's SELECT-BLOCK grid, quantised to whole
        #    ENGINE blocks (a block is kept when any of its pages scored in - a needle block
        #    can never be half-dropped), question tail forced, capped to the aligned window.
        self._in_rebuild_select = True
        try:
            pool_blocks = int(self.view.num_blocks())
            cap_engine_blocks = max(2, min(
                pool_blocks, int(self.budget_tokens) // (bp * pt)))
            # Rank by OUR OWN core scores, not by `select_pages` (which returns the working set
            # ASCENDING by design - the header says so - so slicing it yielded the sink blocks).
            # score grid = selection blocks (64 tokens); an ENGINE block (49 pages) is top-ranked
            # by the max score over its selection blocks.
            core = getattr(orch, "_last_core", None)
            if core is None:
                try:
                    core = orch.core_selection_scores()
                except Exception:
                    core = None
            if core is None:
                return False, "no retrieval scores (query never frozen?)"
            _rc = getattr(orch, "_recall", None)
            bt = max(1, int(getattr(_rc, "block_tokens", 0) or 64))
            pages_per_sel_block = max(1, bt // pt)
            eng_scores = {}
            for gi, v in enumerate(core):
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv != fv:
                    continue
                first_pg = gi * pages_per_sel_block
                eng = first_pg // bp
                if eng > cap_engine_blocks * 4 and eng * bp >= prompt_tokens:
                    continue
                if fv > eng_scores.get(eng, 0.0):
                    eng_scores[eng] = fv
            rank_blocks = sorted(eng_scores, key=lambda b: -eng_scores[b])
            tail_blocks = {p // bp for p in forced_tail}
            # a block enters the pack ONLY when every page inside it is complete AND has
            # canonical store bytes (or is the engine-resident tail): a hole would shift
            # later pages against their assigned ranks. Filter BEFORE assigning destinations.
            def block_ok(b):
                for p in range(b * bp, (b + 1) * bp):
                    if p >= prompt_tokens:
                        continue
                    if not (orch.store.has(p) or p >= tail_first):
                        return False
                return True
            room = max(0, cap_engine_blocks - len(tail_blocks))
            middle = [b for b in rank_blocks
                      if b not in tail_blocks and block_ok(b)][:room]
            chosen = sorted(set(middle) | tail_blocks)
            logical_pages = sorted({p for b in chosen for p in
                                    range(b * bp, (b + 1) * bp) if p < n_pages})
            missing = 0
            # 2) self-contained device pack: page -> pool window slot, each written as the
            #    canonical store bytes rotated ONCE to its new rank (`restore_page_rank`:
            #    `needed - store_delta`, never a rotation of a rotation).
            dest = {lp: pool_pages[i] for i, lp in enumerate(logical_pages)}
            done = 0
            for rank_, lp in enumerate(logical_pages):
                if orch.store.has(lp):
                    try:
                        orch.restore_page_rank(int(lp), int(dest[lp]), int(rank_))
                        done += 1
                        continue
                    except Exception:
                        return False, f"pack failed at page {lp} rank {rank_}"
                # resident question page: still at its engine slot; copy the raw bytes (no
                # rotation hop needed - it keeps its baked phase and rebakes to `rank_`)
                src = orch.slot_of.get(int(lp))
                if src is None:
                    return False, f"tail page {lp} lost its slot"
                for kv in orch.kvs:
                    kv[int(dest[lp])].copy_(kv[int(src)])
                baked = (rank_ - int(lp)) * pt
                orch.baked_delta[int(lp)] = int(baked)
                orch.slot_of[int(lp)] = int(dest[lp])
                done += 1
            phys = [int(dest[lp]) for lp in logical_pages]
            selected = [int(lp) for lp in logical_pages]
            if self.log is not None:
                self.log({"kind": "rebuild_select_diag", "req": req,
                          "engine_rank_head": rank_blocks[:24],
                          "rank_blocks": len(rank_blocks),
                          "rank_blocks_tail": rank_blocks[-40:],
                          "tail_blocks": sorted(tail_blocks),
                          "chosen": len(chosen), "chosen_head": chosen[:12],
                          "chosen_tail": chosen[-24:],
                          "want": len(logical_pages),
                          "packed": int(done), "store_missing": int(missing)})
        except Exception as exc:
            self._in_rebuild_select = False
            return False, f"pack error: {type(exc).__name__}: {exc}"
        self._in_rebuild_select = False
        if not selected:
            return False, "selection packed nothing"
        if max(selected) < n_pages - 1:
            return False, "selection lost the question tail"
        # 3) validate the packed physical row: ascending pool pages, unique, block-aligned
        if self.log is not None:
            self.log({"kind": "rebuild_phys_diag", "req": req,
                      "n": len(phys), "sel_head": selected[:4], "sel_tail": selected[-4:],
                      "phys_head": phys[:4], "phys_tail": phys[-4:],
                      "block_first": [p // bp for p in phys[:4]],
                      "block_tail": [p // bp for p in phys[-4:]]})
        if len(phys) != len(selected) or min(phys) <= 0 or len(set(phys)) != len(phys):
            return False, f"packed row invalid (pages={len(selected)}, uniq={len(set(phys))})"
        # The packed destination runs over CONSECUTIVE pool pages starting at a block base
        # (`first*bp`), so in the page-number space the row must equal its own first page
        # expanded: [base, base+1, ..., base+n-1]. (Block ids would be the same arithmetic
        # in the block namespace; the device paged-KV tensor is indexed by pages.)
        base = phys[0]
        if base % bp:
            return False, f"pack start {base} is not a block base"
        if phys != list(range(base, base + len(phys))):
            if self.log is not None:
                self.log({"kind": "rebuild_contig_diag", "req": req,
                          "first_bad": next((i for i in range(len(phys))
                                             if phys[i] != base + i), -1),
                          "phys_head": phys[:4], "phys_tail": phys[-6:],
                          "pool_head": pool_pages[:4]})
            return False, "packed row is not contiguous from its block base"
            # the picker kept whole blocks in ascending order normally; a partial tail block
            # is legal only for the LAST page (the prompt's final page) - anything else means
            # the pack landed on scattered pool pages Core would reject.
            if pages[:-1] != expanded[:len(pages) - 1]:
                return False, "packed row is not complete-block contiguous"
        # 3) books: the packed view IS the old generation's resident window now (it retires at
        #    the rebuild admission, but the decode steps until then read these pages)
        st.row = tuple(phys)
        st.logical = tuple(selected)
        st.seq_len = int(len(phys) * pt)
        st.tokens = int(st.tokens)          # logical frontier untouched (prefix mode = resident)
        st.off_pages = 0
        st.applied_view_coord = True
        st.compacted_in_prefill = True
        st.window_src = {rank: int(lp) for rank, lp in enumerate(selected)}
        self._owned_slots[req] = list(st.row)
        # 4) hand Core the reconstruction over the channel (rebuild branch -> pending -> fresh
        #    dense admission of `select_tokens(logical)` on the scheduler thread)
        try:
            res = ch.compact(req, selected, pt, visible_tokens=0, rebuild=True,
                             block_pages=bp, physical_pages=phys,
                             tail_tokens=int(tail_tokens))
        except Exception as exc:
            return False, f"channel error: {type(exc).__name__}: {exc}"
        ok = isinstance(res, dict) and bool(res.get("ok"))
        if ok:
            self.accept_reconstruction(req)
        if self.log is not None:
            self.log({"kind": "rebuild_select", "req": req, "ok": bool(ok),
                      "pages": len(selected), "tail_forced": len(forced_tail),
                      "row": len(phys), "prompt_tokens": prompt_tokens,
                      "detail": str(res)[:160]})
        return ok, res

    def note_applied_epoch(self, req, epoch, visible, logical):
        """Core applied a boundary compaction at this epoch. A PRESSURE apply (visible <
        logical) leaves the engine's book on the logical frontier until the final rewind -
        the request therefore still runs in TRUE coordinates; a final apply with the book
        rewind switches the whole pipeline to VIEW coordinates (book_rewound)."""
        st = self.state(req)
        hist = getattr(self, "_applied_epochs", None)
        if hist is None:
            hist = self._applied_epochs = {}
        hist[req] = int(epoch)
        if visible < logical and not bool(getattr(st, "applied_view_coord", False)):
            st.applied_view_coord = False
        return True

    def book_rewound(self, req, visible):
        """EngineCore applied a boundary compaction and the engine's resident book now reads
        `visible`: positions, slot mapping and attention length all run in VIEW coordinates from
        here on. Recorded so the lagged commit bakes the rotation the engine actually applied
        (required by KVMEM_REBAKE: without it the page phases keep the pre-compaction basis and
        the model sees a view whose keys disagree with every position it generates)."""
        st = self.state(req)
        st.applied_view_coord = True
        # The engine no longer addresses positions >= visible. If the proposal-apply path did
        # not already record the frame (ordering, or the apply went through the socket path),
        # adopt it now so the lagged commit and the row mirror keep the right offset.
        try:
            if int(st.tokens) > int(visible):
                st.off_pages = (int(st.tokens) - int(visible)) // max(1, int(self.page_tokens))
            if len(st.logical or ()) < len(st.row or ()):
                pad = [(rank + int(st.off_pages)) for rank in range(len(st.logical), len(st.row))]
                st.logical = tuple(st.logical or ()) + tuple(pad)
        except Exception:
            pass
        if self.log is not None:
            self.log({"kind": "book_rewound", "req": req, "visible": int(visible),
                      "tokens": int(getattr(st, "tokens", 0)),
                      "seq_len": int(getattr(st, "seq_len", 0))})
        return True

    def _reselect_remobilize(self, req, pressure=False):
        """THE FINAL PROTOCOL: a view expressed as PHYSICAL BLOCKS the engine really holds.

        Three measured truths forced this shape:
          1. the kernel write address is `positions // page -> row[pos]` (logical positions, the
             engine book is never rewound by anything we control - the probe showed the GPU book
             pushed back to `tokens + step` one step after every rewind);
          2. so a view row can never be shorter/longer than the engine's own allocation: any
             `visible < logical` compaction makes the NEXT write index exceed the row - the
             kernel then reads a zero block id and writes into physical page 0 (the c10/c11/c88K/
             165K failures: pages 1298 written while the row had 1274, needle pages captured as
             zeros or stranded outside the adopted blocks);
          3. the engine's block table is rebuilt from ITS list every step - installed rows are
             cosmetic, the only rows that survive are the request's real allocation.

        Therefore compaction here means: the engine keeps its dense allocation (writes stay
        correct, book untouched), and we RE-RANK the K pages in place - the view becomes the
        ascending physical block list the engine holds, while the SEMANTIC order (what RoPE says)
        is rewritten so rank == position: each kept logical page's bytes are re-baked from the
        canonical store to the row position the engine will read at that position. Selection is
        whole-block quantised (the kernel table granularity), scores consensus TP-averaged.

        No proposals, no adoption, no rewind: Core's books are left alone. The view must be
        installed at every step (KVMEM_INSTALL_ROW_VIEW) because it is a re-RANKING, not a
        re-allocation - and the mirror in `sync_row_from_engine` must not clobber it.
        """
        st = self._state[req]
        orch = self._orch[req]
        pt = int(self.page_tokens)
        q = max(1, int(getattr(self, "block_pages", None)
                       or getattr(self, "pages_per_block", 1) or 1))
        block_tokens = q * pt
        tokens = int(st.tokens)
        n_pages = -(-tokens // pt)
        n_blocks = -(-n_pages // q)
        budget_blocks = max(2, int(self.budget_tokens) // block_tokens)
        sink_blocks = max(1, int(getattr(self, "sink_tokens", 0) or 0) // block_tokens)
        recent_blocks = max(1, -(-int(getattr(self, "recent_tokens", 0) or 0) // block_tokens))
        if sink_blocks + recent_blocks >= budget_blocks:
            recent_blocks = max(1, budget_blocks - sink_blocks - 1)
        kept = set(range(min(sink_blocks, n_blocks))) \
            | set(range(max(0, n_blocks - recent_blocks), n_blocks))
        core = None
        try:
            core = orch.core_selection_scores()
        except Exception:
            core = None
        if core is None:
            core = getattr(orch, "_last_core", None)
        if (core is not None and len(core) and bool(getattr(self, "_in_quiescent", False))
                and int(getattr(self, "tp_size", 1) or 1) > 1):
            try:
                from kvmem_tp_scores import average_reconstruction_scores
                _sel = getattr(orch, "selector", None)
                _sb = getattr(_sel, "selection_blocks", None)
                expected = int(_sb() if callable(_sb) else len(core))
                device = self.view.paged_kv()[0].device
                core = average_reconstruction_scores(list(core), expected, req,
                                                     int(st.tokens), device)
            except Exception as exc:
                if self.log is not None:
                    self.log({"kind": "tp_consensus_degraded", "req": req,
                              "error": f"{type(exc).__name__}: {exc}"})
                core = None
        if core is not None and len(core):
            grid_pages = max(1, int(getattr(orch, "select_block_tokens", 2 * pt)
                                    or 2 * pt) // pt)
            block_scores: dict[int, float] = {}
            for gi, v in enumerate(core):
                b = (gi * grid_pages) // q
                try:
                    block_scores[b] = block_scores.get(b, 0.0) + float(v)
                except (TypeError, ValueError):
                    continue
            for b in sorted(block_scores, key=lambda x: -block_scores[x]):
                if len(kept) >= budget_blocks:
                    break
                if 0 <= b < n_blocks:
                    kept.add(int(b))
        kept = sorted(kept)
        # the tail page of the LOGICAL stream must exist in the view (decode continues there)
        tail_block = (n_pages - 1) // q
        if tail_block not in kept and len(kept) < budget_blocks:
            kept.append(tail_block)
            kept = sorted(kept)
        # physical page at row position p = the engine's own allocation: st.row (dense mirror).
        old_row = list(st.row or ())
        if len(old_row) < q * n_blocks:
            pad = self._page_map.get(req, {})
            old_row = old_row + [int(pad.get(p, 0)) for p in range(len(old_row), q * n_blocks)]
        phys_pages, logical_pages = [], []
        for b in kept:
            for o in range(q):
                p = b * q + o
                if p >= n_pages:
                    break
                if p < len(old_row) and int(old_row[p]) > 0:
                    phys_pages.append(int(old_row[p]))
                    logical_pages.append(int(p))
        if not phys_pages:
            self._last_proposal = None
            return st
        if len(phys_pages) * pt >= tokens + pt:
            # nothing to compress yet (selection covers the stream); keep the pure dense prefix
            # which the engine already reads correctly.
            self._last_proposal = None
            if self.log is not None:
                self.log({"kind": "remobilize_noop", "req": req, "tokens": tokens,
                          "pages": len(phys_pages)})
            return st
        missing = 0
        for rank, (lp, phys) in enumerate(zip(logical_pages, phys_pages)):
            try:
                if orch.store.has(int(lp)):
                    orch.restore_page_rank(int(lp), int(phys), int(rank))
                else:
                    missing += 1
            except Exception:
                missing += 1
        st.row = tuple(phys_pages)
        st.logical = tuple(logical_pages)
        st.seq_len = int(len(phys_pages) * pt)
        st.spare = None
        st.off_pages = 0
        st.applied_view_coord = True
        orch.slot_of = dict(zip(logical_pages, phys_pages))
        orch.resident = set(logical_pages)
        self._install_row_now[req] = tuple(phys_pages)
        if self.log is not None:
            self.log({"kind": "remobilize", "req": req, "blocks": len(kept),
                      "pages": len(phys_pages), "seq_len": int(st.seq_len),
                      "tokens": tokens, "missing": missing,
                      "scores": bool(core is not None and len(core or ()))})
        return st

    def _reselect_prefix(self, req, pressure=False):
        """THE protocol for this fork: the engine retires its OWN prefix; we do not move KV.

        Measured laws (2026-10-30 SM75, probe-verified):
          1. the worker block table is append-only in LOGICAL order (`append_block_ids
             (overwrite=False)` from the feed) - no adoption or install can reorder it;
          2. the device book is pushed back to the logical frontier by the feed one step after
             every rewind, and positions/slot-mappings derive from it - the token stream is
             permanently LOGICAL-COORDINATED in the engine;
          3. the fork ALREADY runs `remove_skipped_blocks(processed_computed_tokens)` every
             step, replacing out-of-window blocks with null_block placeholders - its native
             sliding-window retirement.

        So the only self-consistent compaction here is the engine's own: the resident window
        stays at the logical frontier (K keeps its true-position phase, writes hit their real
        pages, attention reads them unchanged), history beyond the window is retired, and the
        CANONICAL STORE keeps every retired page in the logical frame (sha-verified) for
        retrieval and session replay. Our job: never corrupt that machinery - proposals are
        made ONLY to Core when Core's own books need the retirement record (pressure), and the
        store is fed at every step from the real write slots. `visible_tokens` is the resident
        window as the ENGINE sees it; `selected_pages` the logical pages still resident.

        This is deliberately NOT a shrinking of the GPU allocation at an arbitrary mid-stream
        boundary: on this fork the GPU window can only be moved by the engine itself (null
        placeholders + the R-SWA pass) or by a genuine prefill-rebuild admission (stage 2:
        KVMEM_COMPACT_REBUILD restores a selection and re-prefills the rest). A proposal that
        shrinks below what the engine's next feed re-derives is refused - correctly - and the
        revert keeps books intact; the 40K pairing run measured that path clean (refusal,
        dense continuation, CORRECT needle answer).
        """
        st = self._state[req]
        orch = self._orch[req]
        pt = int(self.page_tokens)
        q = max(1, int(getattr(self, "block_pages", None)
                       or getattr(self, "pages_per_block", 1) or 1))
        tokens = int(st.tokens)
        resident = int(st.seq_len)
        budget = int(self.budget_tokens)
        row = [int(p) for p in (st.row or ())]
        logical = [int(v) if v is not None else -1 for v in (st.logical or ())]
        if len(logical) != len(row):
            logical = list(range(len(row)))
        # Reports are once per request; pressure/final both re-enter this path and a per-step
        # audit line is log spam (measured: 1.4M rows during one stalled run).
        if self.log is not None and int(getattr(st, "tokens", 0)) != getattr(
                self, "_last_report_tokens", {}).get(req, -1):
            self._last_report_tokens = getattr(self, "_last_report_tokens", {})
            self._last_report_tokens[req] = int(tokens)
            self.log({"kind": "prefix_report", "req": req, "resident": int(resident),
                      "tokens": tokens, "budget": budget, "pages": len(row)})
        return st

    def _reselect_blockwise(self, req, pressure=False):
        """The NO-MOVE block view - the only view this engine's kernel table can express.

        The scheduler feeds the attention kernel a list of FULL blocks and the gather expands
        every block id into its 49 pages, so a page-granular selection becomes fiction at
        adopt: `kept = unique(page // 49)` drags whole neighbouring blocks in and drops the
        pages whose block was only partly chosen. Measured (the 86,686-token pairing): the
        SAME prompt+question answers `ZX-4417-QJ7` when no compaction runs and degenerates to
        a token loop after ONE page-granular compaction - a needle sitting at view rank 0 (no
        rotation at all) was unreadable, because the row we installed and the row the kernel
        expands were different sequences.

        Protocol (official KVMem build_sparse_view semantics under block quantisation):
          * selection stays in the LOGICAL frame the page picker and the store keys use, but
            picks whole LOGICAL blocks (sink band + score-ranked middle + recent/tail band);
          * every kept page stays EXACTLY where it physically is - the physical page of a
            logical page is looked up through the PREVIOUS view (row position of that logical
            id), so no device move happens and the row is by construction the ascending
            expansion of whole blocks (the induction: round one sees a dense row);
          * each kept page is rebuilt IN PLACE from the canonical store rotated to its new
            view rank (`restore_page_rank`, one hop) - the fork's engine bakes/reads rope at
            the resident book position, which the apply rewinds to the view frame, so the
            rotation is what makes the view's phases agree with the positions the model
            will generate (this fork has NO `kvmem_rope_offset` consumer - pure-copy views
            cannot be corrected at read time and every one of them looped).
        """
        st = self._state[req]
        orch = self._orch[req]
        pt = int(self.page_tokens)
        q = max(1, int(getattr(self, "block_pages", None)
                       or getattr(self, "pages_per_block", 1) or 1))
        block_tokens = q * pt
        resident_before = int(st.seq_len)
        tokens = int(st.tokens)
        old_row = list(st.row or ())
        old_logical = list(st.logical or ())
        if not old_row:
            self._last_proposal = None
            if self.log is not None:
                self.log({"kind": "blockwise_no_row", "req": req})
            return st
        if len(old_logical) != len(old_row):
            if bool(getattr(st, "applied_view_coord", False)) or int(getattr(st, "off_pages", 0)):
                self._last_proposal = None
                if self.log is not None:
                    self.log({"kind": "blockwise_no_logical", "req": req,
                              "row": len(old_row), "logical": len(old_logical)})
                return st
            # DENSE frame: the engine wrote at true positions, so row position IS the logical
            # page (the page ids themselves are physical and irrelevant here).
            old_logical = list(range(len(old_row)))
        pos_of_logical = {}
        for i, v in enumerate(old_logical):
            if v is not None:
                pos_of_logical.setdefault(int(v), i)
        n_pages = -(-tokens // pt)                       # logical pages that exist
        n_blocks = -(-n_pages // q)
        budget_blocks = max(2, int(self.budget_tokens) // block_tokens)
        sink_blocks = max(1, int(getattr(self, "sink_tokens", 0) or 0) // block_tokens)
        recent_blocks = max(1, -(-int(getattr(self, "recent_tokens", 0) or 0) // block_tokens))
        if sink_blocks + recent_blocks >= budget_blocks:
            recent_blocks = max(1, budget_blocks - sink_blocks - 1)
        kept = set(range(min(sink_blocks, n_blocks))) \
            | set(range(max(0, n_blocks - recent_blocks), n_blocks))
        kept.add((n_pages - 1) // q)                     # Core validates the tail page
        core = None
        try:
            core = orch.core_selection_scores()
        except Exception:
            core = None
        if core is None:
            core = getattr(orch, "_last_core", None)
        if (core is not None and len(core) and bool(self._in_quiescent)
                and int(getattr(self, "tp_size", 1) or 1) > 1):
            # TP2 consensus: without it each rank ranks the middle by its own shard's scores,
            # the proposals differ and Core refuses ("TP ranks proposed different working
            # sets"). The collective raises SYMMETRICALLY, so degrading to `no ranking` keeps
            # every rank on the same deterministic sink/recent/tail selection.
            try:
                from kvmem_tp_scores import average_reconstruction_scores
                _sel = getattr(orch, "selector", None)
                _sb = getattr(_sel, "selection_blocks", None)
                expected = int(_sb() if callable(_sb) else len(core))
                device = self.view.paged_kv()[0].device
                core = average_reconstruction_scores(list(core), expected, req,
                                                     int(st.tokens), device)
            except Exception as exc:
                if self.log is not None:
                    self.log({"kind": "tp_consensus_degraded", "req": req,
                              "error": f"{type(exc).__name__}: {exc}"})
                core = None
        if core is not None and len(core):
            try:
                grid_pages = max(1, int(getattr(orch, "select_block_tokens", 2 * pt)
                                        or 2 * pt) // pt)
                block_scores = {}
                for gi, v in enumerate(core):
                    b = (gi * grid_pages) // q
                    try:
                        block_scores[b] = block_scores.get(b, 0.0) + float(v)
                    except (TypeError, ValueError):
                        continue
                order = sorted(block_scores, key=lambda b: -block_scores[b])
                room = max(0, budget_blocks - len(kept))
                for b in order:
                    if room <= 0:
                        break
                    if 0 <= b < n_blocks and b not in kept:
                        kept.add(int(b))
                        room -= 1
            except Exception:
                pass
        kept = sorted(kept)
        while len(kept) < budget_blocks:                 # nearest-missing middle blocks
            nxt = max(kept) + 1
            if nxt >= n_blocks:
                break
            kept.append(nxt)
        logical_pages, phys_pages = [], []
        for b in kept:
            members = []
            complete = True
            for o in range(q):
                lp = b * q + o
                if lp >= n_pages:
                    break                              # the partial tail block is allowed
                i = pos_of_logical.get(lp)
                if i is None or int(old_row[i]) <= 0:
                    complete = False
                    break
                members.append((lp, int(old_row[i])))
            if not complete and b != (n_pages - 1) // q:
                continue                               # a block with lost pages cannot be
                                                       # adopted as a whole block - drop it
            for lp, phys in members:
                logical_pages.append(lp)
                phys_pages.append(phys)
        if not logical_pages:
            self._last_proposal = None
            return st
        visible = sum(min(pt, max(0, tokens - p * pt)) for p in logical_pages)
        if visible <= 0 or visible >= resident_before:
            self._last_proposal = None
            if self.log is not None:
                self.log({"kind": "blockwise_no_shrink", "req": req, "visible": int(visible),
                          "resident": int(resident_before), "pages": len(logical_pages)})
            return st
        moved = failed = 0
        for rank, (lp, phys) in enumerate(zip(logical_pages, phys_pages)):
            try:
                orch.restore_page_rank(int(lp), int(phys), int(rank))
                moved += 1
            except Exception:
                failed += 1
        st.row = tuple(phys_pages)
        st.logical = tuple(logical_pages)
        st.seq_len = int(visible)
        st.spare = None
        # The engine book rewinds to `visible` at apply; positions the engine will generate next
        # are VIEW coordinates while the LOGICAL stream keeps running (off = frontier-visible).
        st.off_pages = max(0, (tokens - int(visible)) // pt)
        orch.slot_of = dict(zip(logical_pages, phys_pages))
        orch.resident = set(logical_pages)
        try:
            orch.rebuild_index(tuple(logical_pages))
        except Exception:
            pass
        self._owned_slots[req] = list(st.row)
        self._last_proposal = {
            "request": str(req), "epoch": int(getattr(self, "epoch", 0) or 0),
            "logical_frontier": tokens,
            "resident_frontier": resident_before,
            "selected_pages": [int(p) for p in logical_pages],
            "physical_pages": [int(p) for p in phys_pages],
            "visible_tokens": int(visible), "page_tokens": int(pt),
            "block_pages": int(q),
            "engine_logical": self._engine_logical.get(req),
            "engine_computed": self._engine_computed.get(req),
            "store_pages": len(orch.store.pages),
            "store_max": max(orch.store.pages, default=-1),
            "blockwise": True, "moved": moved, "failed": failed,
        }
        if self.log is not None:
            self.log({"kind": "blockwise_reselect", "req": req,
                      "blocks": len(kept), "kept": [int(b) for b in kept],
                      "pages": len(logical_pages),
                      "visible": int(visible), "resident_before": int(resident_before),
                      "moved": moved, "failed": failed,
                      "scores": bool(core is not None and len(core or ()))})
        return st

    def revert_quiescent(self, req):
        """EngineCore-callable rollback after it REFUSED the last quiescent proposal.

        The worker moves its view before Core can validate; without this, one refusal (measured:
        the final reselect whose store lacked the tail) leaves the worker permanently at the
        proposed books while Core keeps the old ones, and every later proposal then refuses with
        "frontier differs" until the request ends with a torn view. The canonical store has the
        real bytes, so restoring the snapshotted row is a pure store->device copy plus a book
        rewind. Consumes the snapshot (idempotent afterwards)."""
        snap = getattr(self, "_quiescent_snap", None)
        if not snap or snap.get("req") != req or req not in self._state:
            return False
        self._quiescent_snap = None
        st = self._state[req]
        orch = self._orch[req]
        try:
            # rebuild the slide-undo: old ranks restored one hop from the store
            for rank, (page, slot) in enumerate(zip(snap["logical"] or (), snap["row"])):
                if page is None:
                    continue
                try:
                    if self._pure_copy():
                        orch.restore_page(int(page), int(slot))
                    else:
                        orch.restore_page_rank(int(page), int(slot), int(rank))
                except Exception:
                    continue
        except Exception:
            pass
        st.row, st.logical = snap["row"], snap["logical"]
        st.seq_len, st.spare = snap["seq_len"], snap["spare"]
        self._owned_slots[req] = list(snap["owned"])
        orch.resident = snap["resident"]
        orch.slot_of = snap["slot_of"]
        orch.baked_delta = snap["baked_delta"]
        if self.log is not None:
            self.log({"kind": "quiescent_reverted", "req": req,
                      "seq_len": int(st.seq_len), "tokens": int(st.tokens),
                      "row": len(st.row or ())})
        return True

    def _reselect(self, req, prefill_pressure=False, probe=False, rebuild=False, send=True):
        st = self._state[req]
        resident_before = int(st.seq_len)
        # ALIGNMENT INVARIANT: a compaction may only run on a page boundary of the LOGICAL stream.
        # The engine writes the next tokens at compacted positions starting at `visible`. If
        # `logical - visible` is not a whole page, the tail logical page would be split across two
        # ranks (part of it written under the old layout, part under the new one) and no single
        # RoPE phase can describe it. Measured on 0.2.2: 207 of 238 compactions ran unaligned, the
        # needle check compared against the same wrong delta so it passed, and the answers rotted.
        # Defer instead: decode crosses a page boundary within <= page_tokens steps.
        # A REBUILD is exempt: it discards the old KV and pre-fills the working set from token 0,
        # so no write continues from the compacted position that invariant protects. Blocking it
        # here made the prompt-end hook raise ("did not queue a fresh admission") and killed the
        # engine on requests whose last chunk ended mid-page.
        # The invariant protects the PARTIAL tail page from being split across layouts. When the
        # recent window is non-empty it always keeps that page, and it stays last in the ascending
        # selection, so the next token continues filling it under the new layout - alignment of the
        # logical frontier itself is not required. Requiring it delayed the prompt-end reselect by
        # up to 31 decode steps (measured: prompt 169,998 -> fired only at 170,016, after the
        # garbage tokens were already conditioned on the missing needle).
        _pt_ = int(self.page_tokens)
        tail_partial = bool(int(st.tokens) % _pt_) and \
            (int(st.tokens) // _pt_) == ((int(st.tokens) - 1) // _pt_)
        if (not rebuild and int(st.tokens) % _pt_
                and not (tail_partial and int(getattr(self, "recent_tokens", 0) or 0) > 0)):
            if self.log is not None:
                self.log({"kind": "compact_deferred_align", "req": req,
                          "tokens": int(st.tokens), "seq_len": int(st.seq_len),
                          "page": int(self.page_tokens)})
            return st
        if (bool(getattr(self, "blockwise_view", False)) is False
                and (getattr(self, "compact_mode", None) or "") == "prefix"
                and int(st.tokens) > int(st.seq_len) > 0):
            # prefix mode is retirement REPORTING only; while a request is decoding, every
            # step would re-enter this blocking collective_rpc (the trigger re-fires because
            # pressure never clears). Report once and leave - the next prompt-complete or a
            # new epoch re-enters through the final path. (Measured: decode crawled on
            # per-step 120s-timeout RPC + mirror traffic.)
            if self.log is not None and int(getattr(st, "tokens", 0)) != getattr(
                    self, "_last_report_tokens", {}).get(req, -1):
                self._last_report_tokens = getattr(self, "_last_report_tokens", {})
                self._last_report_tokens[req] = int(st.tokens)
                self.log({"kind": "prefix_report", "req": req,
                          "resident": int(st.seq_len), "tokens": int(st.tokens),
                          "budget": int(self.budget_tokens),
                          "pages": len(st.row or ()), "decoding": True})
            self._last_proposal = None
            return st
        cap = getattr(self, "capture", None)
        if cap is not None and getattr(self, "retrieval_enabled", False):
            try:
                from kvmem_engine_capture import harvest_from_bridge
                fed = harvest_from_bridge(cap)
                if self.log is not None and not getattr(self, "_bridge_log", None) == fed:
                    self._bridge_log = fed
                    import vllm.kvmem_bridge as _b
                    self.log({"kind": "bridge_harvest", "req": req, "layers_fed": int(fed),
                              "bridge": _b.stats()})
            except Exception:
                if self.log is not None:
                    import traceback as _tb
                    self.log({"kind": "bridge_harvest_error", "error": _tb.format_exc()})
        # Diagnostic: what page ids does the ENGINE actually use for this row? If they do not
        # start at the ids our slot pool hands out, every read is shifted and the output turns
        # to noise.
        try:
            row_now = self.view.block_table_rows()[0][st.row_index]
            self.log({"kind": "engine_row", "req": req,
                      "head": [int(v) for v in row_now[:8].tolist()],
                      "nonzero": int((row_now != 0).sum())})
        except Exception:
            pass
        orch = self._orch[req]
        # Do NOT release the request's slots here: `orch.slot_of` still points at them and they are
        # the SOURCES of this reselect's device moves. Releasing first let a concurrent request take
        # them (`_slot_owner` cleared), so the moves copied the other request's bytes into this
        # view - measured live as sink pages failing the needle check and gibberish output.
        old_owned = set(self._owned_slots.get(req, ()))
        keep = {int(slot) for slot in orch.slot_of.values()}
        # Reserve exactly what the selection needs (+1 spare for the next token page). Asking for
        # the whole budget (budget_pages()+1 = 3329 pages at B=53248) drained the pool for every
        # request and left later requests reading pages they did not own.
        taken = {}

        own = self._own_pages(req)
        taken["own"] = own

        def _take(needed):
            need = int(needed)
            if len(own) >= need:
                # Destinations are ALWAYS the request's own pages; when there is no spare page left
                # in them, borrow exactly ONE slot from the pool instead of switching the whole
                # destination set to pool slots. The old `> need` test made the FIRST compaction
                # (row holding exactly `need` pages) fall back to the pool, so the row stopped
                # pointing at the request's own pages, later growth reused stale entries, and the
                # sink pages were overwritten (measured: `needle: false` on pages 0/1 after five
                # 200K compactions).
                slots = list(own[:need])
                if len(own) > need:
                    slots = slots + list(own[need: need + 1])
                else:
                    slots = slots + self._take_slots(req, 1, keep=keep)
                taken["slots"] = slots
                taken["own_packed"] = len(own) > need
                return slots[:need]
            slots = self._take_slots(req, need + 1, keep=keep)
            taken["slots"] = slots
            taken["own_packed"] = False
            return slots[:need]

        # Sweep: pages written during prefill can be skipped by the per-step commit when the
        # engine had not published their page id yet. They are then absent from the store and the
        # orchestrator drops them from the view (measured: 475 -> 331 pages, so the compact no
        # longer covered the prompt and was refused). Commit every page we can still map.
        self._sweep_missing_pages(req, orch, st)
        prompt_tokens = int(getattr(orch, "prompt_tokens", 0) or 0)
        _page = max(1, int(self.page_tokens))
        _budget_pages = max(1, int(self.budget_tokens) // _page)
        _sink_pages = max(0, int(getattr(self, "sink_tokens", 0) or 0) // _page)
        _prompt_fits = bool(prompt_tokens) and prompt_tokens <= int(self.budget_tokens)
        if _prompt_fits:
            # The prompt fits the budget: pin EVERY prompt page. There is nothing to choose and the
            # engine keeps `_all_token_ids` covering the prompt.
            mandatory = list(range(0, -(-prompt_tokens // _page)))
        else:
            # The prompt does NOT fit the budget: pin ONLY the two bands KVMem never drops - the
            # SINK head - and leave everything in between to the retrieval ranking, which is what
            # makes the compacted view a COMPRESSION instead of a truncation.
            #
            # Pinning the whole prompt here (the old behaviour) clamped the set to the NEWEST
            # `budget_pages - sink_pages` pages, so sink + newest filled all of B and the
            # retrieval-chosen middle could never enter: the scores were computed, the selector
            # ran, and zero middle pages were selected.
            mandatory = list(range(min(_sink_pages, _budget_pages)))
        # The RECENT tail of the OUTPUT must stay too: retrieval has no scores for freshly written
        # blocks (they are padded with -inf), so without this the model loses the lines it just
        # produced and the output degenerates into fragments (measured: long context + retrieval).
        recent = int(getattr(self, "recent_tokens", 0) or 0)
        # `st.tokens` is the WRITTEN frontier (accounting is lagged one step on purpose, see
        # advance_tokens). Adding `_inflight` here pushed the recent window past the newest written
        # token, so the pages it pinned did not exist yet and the window itself was offset.
        recent_end = int(st.tokens)
        if recent > 0 and recent_end:
            first = max(0, (recent_end - recent) // _page)
            last = (recent_end - 1) // _page
            mandatory = sorted(set(mandatory) | set(range(first, last + 1)))
        # Never let the mandatory bands alone overflow the budget (sink + recent is 10,240 tokens at
        # the shipped defaults, well below B; this only fires on an odd configuration).
        if len(mandatory) > _budget_pages:
            mandatory = mandatory[-max(1, _budget_pages - _sink_pages):]
        # Force the prompt pages only when the prompt fits the budget. If it does not, forcing them
        # would make the view exceed B and the plan would be rejected anyway (the compact also
        # refuses to go below the prompt).
        forced = mandatory if _prompt_fits else []
        score_reducer = None
        if (rebuild and self.compact_rebuild) or (
                getattr(self, "_in_quiescent", False)
                and int(getattr(self, "tp_size", 1) or 1) > 1):
            from kvmem_tp_scores import average_reconstruction_scores
            device = self.view.paged_kv()[0].device

            def _consensus(scores, count):
                # The collective raises SYMMETRICALLY on every rank (the availability/shape check
                # runs after the all_reduce), so returning None here leaves all ranks with the same
                # "no ranking" state. An escaping exception would poison the multiproc worker and
                # kill the engine (measured: one availability mismatch -> EngineDeadError).
                try:
                    return average_reconstruction_scores(scores, count, req, st.tokens, device)
                except Exception as exc:
                    if self.log is not None:
                        self.log({"kind": "tp_consensus_degraded", "req": req,
                                  "local_len": None if scores is None else len(scores),
                                  "expected": int(count),
                                  "tokens": int(st.tokens),
                                  "error": f"{type(exc).__name__}: {exc}"})
                    return None
            score_reducer = _consensus
        stats = orch.reselect(slots=_take, mandatory_pages=mandatory,
                              forced_pages=forced, prefill_pressure=prefill_pressure,
                              score_reducer=score_reducer, rebake=not self._pure_copy())
        selected = list(orch.last_view.row_pages)
        used = list(orch.slots[:len(selected)])
        spare = taken["slots"][-1]
        if taken.get("own_packed"):
            # Same memory, new order: the selection occupies ranks 0..n-1 and the request's
            # remaining own pages fill the rest of the row (unused until new tokens reach them).
            used_set = set(used)
            rest = [page for page in own if page not in used_set]
            st.row = tuple(used) + tuple(rest)
            st.logical = tuple(selected) + tuple([None] * len(rest))
            st.spare = None
        else:
            st.row = tuple(used) + (spare,)
            st.logical = tuple(selected) + (None,)
            st.spare = spare
        st.seq_len = sum(min(self.page_tokens, max(0, st.tokens - p * self.page_tokens))
                         for p in selected)
        # Only allocated native blocks remain in the row after EngineCore frees its tail.
        # Verified on the live TP2 run (2026-10-30): after each boundary compaction the NEXT
        # schedule re-appends blocks and `sync_row_from_engine` mirrors them (row 1323 -> 1421 ->
        # 1568 ...), so the adopted row does NOT need to pre-reserve the coming chunk. An earlier
        # spare-reservation attempt was reverted after the timeline disproved the gap here.
        quantum = max(1, int(getattr(self, "block_pages", self.pages_per_block) or 1))
        keep_pages = ((len(used) + quantum - 1) // quantum) * quantum
        if getattr(self, "compact_channel", None) is not None:
            st.row = st.row[:keep_pages]
            st.logical = st.logical[:keep_pages]
        # V4 needle: sample a few resident pages and verify the device copy is exactly one
        # rotation of the stored bytes. Cheap (a handful of pages) and catches any drift.
        if int(getattr(self, "full_needle", 0)) > 0:
            # Verify EVERY resident page, not a sample: with ~3.3K pages per compaction a
            # batched-rotation problem would show up on a few of them only.
            sample = sorted(orch.resident)
        else:
            sample = sorted(orch.resident)[:2] + sorted(orch.resident)[-2:]
        checks = {page: orch.needle(page) for page in sample}
        # The needle alone is NOT enough: it compares the device copy against the STORED delta, so
        # a wrong delta agrees with itself. Also check the delta the attention actually depends on:
        # the page at rank r must be baked for (r - page) * page_tokens, and the physical slot the
        # orchestrator holds must be the one the row points at.
        rank_of_all = {int(pg): r for r, pg in enumerate(selected)}
        rank_bad = []
        for page in selected:
            want = (rank_of_all[int(page)] - int(page)) * self.page_tokens
            rank = rank_of_all[int(page)]
            row_slot = int(st.row[rank]) if rank < len(st.row) else -1
            if int(orch.baked_delta.get(int(page), 0)) != want or \
                    int(orch.slot_of.get(int(page), -1)) != row_slot:
                rank_bad.append(int(page))
        self.log({"kind": "needle", "req": req, "pages": len(checks),
                  "ok": all(v is True for v in checks.values()),
                  "rot_ok": not rank_bad, "rot_bad": rank_bad[:4],
                  "detail": {str(k): bool(v) for k, v in checks.items()}})
        # Rotation reconciliation: the query tokens live at compacted positions
        # [st.seq_len, st.seq_len+n) and a page placed at rank r must be baked for
        # (r - logical_page) * page_tokens. Log a few RETRIEVED pages (outside the sink/recent
        # windows) with every number the rotation depends on.
        try:
            rank_of = {int(pg): r for r, pg in enumerate(selected)}
            window = max(0, (int(st.tokens) - int(self.recent_tokens or 0)) // self.page_tokens)
            _ = int(getattr(orch, "prompt_tokens", 0) or 0)
            sample = [pg for pg in selected if int(pg) >= window][:3] or selected[:3]
            rot = {"kind": "rot_check", "req": req,
                   "seq_len": int(st.seq_len), "tokens": int(st.tokens),
                   "engine_computed": self._engine_computed.get(req),
                   "engine_logical": self._engine_logical.get(req),
                   "rows": [{"page": int(pg), "rank": rank_of[int(pg)],
                             "slot": int(orch.slot_of.get(int(pg), -1)),
                             "want": (rank_of[int(pg)] - int(pg)) * self.page_tokens,
                             "baked": int(orch.baked_delta.get(int(pg), 0)),
                             "store_delta": int(orch.store_delta.get(int(pg), 0)),
                             "needle": orch.needle(int(pg))} for pg in sample]}
            self.log(rot)
        except Exception:
            pass
        record = {"kind": "reselect", "req": req, "selected": stats["selected"],
                  "evicted": stats["evicted"], "resident": stats["resident"],
                  "seq_len": st.seq_len, "tokens": st.tokens,
                  "prompt": prompt_tokens, "mandatory_pages": len(mandatory),
                  "planner": orch.planner is not None,
                  "plan": getattr(orch, "last_plan", None),
                  "prefill_pressure": stats["prefill_pressure"],
                  # The books this compaction targeted, plus the engine's published pair. A
                  # frontier difference here is the one-chunk lag that made the TP2 beyond-pool
                  # run move pages the in-flight chunk still referenced.
                  "engine_logical": self._engine_logical.get(req),
                  "engine_computed": self._engine_computed.get(req),
                  "prev_n": int(self._prev_n.get(req, 0) or 0),
                  # Diagnostic: the chosen logical pages, so a known needle page can be checked
                  # against what the compacted view actually kept.
                  "selected_pages": [int(p) for p in selected]}
        cap = getattr(self, "capture", None)
        if cap is not None:
            record["capture"] = {"calls": cap.calls, "wants": cap.wants,
                                 "captured": cap.captured, "errors": cap.errors}
        score_stats = getattr(orch, "last_score_stats", None)
        if score_stats:
            record["scores"] = score_stats          # proof that retrieval ranking was applied
        # The worker installed a compacted row. EngineCore rewinds only resident KV
        # while keeping the original logical tokens untouched; reconstruction alone
        # queues a new prompt for a safe output boundary. The channel reply gates
        # whether the worker may retain the selected view.
        ch = getattr(self, "compact_channel", None)
        if ch is not None and selected and send:
            try:
                # Queue reconstruction; the scheduler applies it AFTER this forward's output
                # arrives. Both TP workers receive a fresh request event on the next schedule.
                _rebuild = bool(rebuild and self.compact_rebuild and not st.rebuilt)
                if _rebuild and int(getattr(self, "tp_rank", 0)) != 0:
                    # The old KV on this rank is about to be discarded. The fresh V2
                    # admission will deliver rank zero's consensus-scored prompt to
                    # every worker, even if this rank lacks a canonical host page.
                    res = {"ok": True, "truncated": True, "rebuild": True,
                           "queued": False}
                else:
                    res = ch.compact(req, selected, self.page_tokens,
                                     visible_tokens=int(st.seq_len), rebuild=_rebuild,
                                     block_pages=int(getattr(self, "block_pages", 1) or 1),
                                     physical_pages=used)
                if _rebuild:
                    if not isinstance(res, dict) or not res.get("ok"):
                        raise RuntimeError(f"reconstruction refused: {res}")
                    st.rebuilt = True
                    record["rebuild_pending"] = True
                if isinstance(res, dict):
                    record["compact_applied"] = bool(res.get("ok"))
                    record["compact_truncated"] = bool(res.get("truncated"))
                    truncated = bool(res.get("truncated"))
                else:
                    record["compact_applied"] = bool(res)
                    truncated = bool(res)
                # The worker's block-table append cursor (`num_blocks.np[group, req]`) counts the
                # page slots written so far; installing a shorter row must move it, or the next
                # engine block is written at the old slot ('row capacity (11514 > 11312)').
                if not _rebuild and self.set_block_cursor is not None and st.row:
                    try:
                        self.set_block_cursor(req, len(st.row))
                    except Exception:
                        pass
                if not truncated:
                    # The engine's book was NOT shortened, so its own view is already correct.
                    # Installing a compacted row anyway left the two books disagreeing: the model
                    # computed at compacted positions while the scheduler kept the old lengths,
                    # num_computed then ran past num_tokens and the request was never scheduled
                    # again (measured: a long prefill stalled with the GPU idle).
                    st.row = ()
                    st.seq_len = 0
                    st.spare = None
            except Exception as exc:
                record["compact_error"] = str(exc)[:160]
                if rebuild:
                    raise
        elif selected and not send:
            # EngineCore called this on its own thread and will adopt the blocks + rewrite its
            # books from this proposal. The worker keeps the installed view and moves the append
            # cursor so the next block lands after the compacted row.
            record["compact_sent"] = False
            self._last_proposal = {
                "request": str(req), "epoch": int(getattr(self, "epoch", 0) or 0),
                "logical_frontier": int(st.tokens),
                # The PRE-compaction resident length: EngineCore compares it with its own
                # `num_computed_tokens` before the rewrite; `visible_tokens` is the new one.
                "resident_frontier": int(resident_before),
                "selected_pages": [int(p) for p in selected],
                # The rows EngineCore must keep: the worker installed `st.row` (already truncated
                # to whole native blocks), so adopting only `used` freed the block-tail pages the
                # row still references and the next chunk wrote into recycled pages.
                "physical_pages": [int(p) for p in (st.row or used)],
                "visible_tokens": int(st.seq_len), "page_tokens": int(self.page_tokens),
                "block_pages": int(getattr(self, "block_pages", 1) or 1),
                "engine_logical": self._engine_logical.get(req),
                "engine_computed": self._engine_computed.get(req),
                # Canonical-store coverage at proposal time: if the tail pages are NOT in the store
                # the selection CANNOT contain them, and Core's mandatory tail-page check will
                # (correctly) refuse - this field separates "selector dropped the tail" from
                # "the tail was never committed".
                "store_pages": len(getattr(self._orch[req], "store", None).pages
                                   if getattr(self._orch[req], "store", None) is not None else ()),
                "store_max": (max(self._orch[req].store.pages)
                              if getattr(self._orch[req], "store", None) is not None
                              and self._orch[req].store.pages else -1),
                "prepare": bool(prefill_pressure)}
            if self.set_block_cursor is not None and st.row:
                try:
                    self.set_block_cursor(req, len(st.row))
                except Exception:
                    pass
        # Slots this request no longer references go back to the pool. On a refusal only the slots
        # taken in this round are dropped; the old sources stay because the orchestrator still
        # points at them (the engine's own row was not touched).
        if probe:
            # Where SHOULD the records be? The test prompt puts KEY-MID at 1/4 of the log and
            # KEY-END twenty lines before the end; blocks are `select_block_tokens` long, so this
            # is exact enough to tell "not ranked" from "ranked but misread".
            blk = max(1, int(self.select_block_tokens))
            prompt = int(self._prompt_len.get(req, 0)
                         or getattr(orch, "prompt_tokens", 0) or 0)
            mid_block = int((prompt // 4) // blk)
            end_block = int((prompt - 20 * 17) // blk)
            sel_blk = sorted({int(p) * self.page_tokens // blk for p in selected})
            self.log({"kind": "retrieval_check", "req": req,
                      "prompt": prompt, "block_tokens": blk,
                      "mid_block": mid_block, "end_block": end_block,
                      "top_blocks": list(getattr(orch, "last_top_blocks", []) or []),
                      "selected_blocks": sel_blk[:12] + (["..."] if len(sel_blk) > 12 else []),
                      # +-2 blocks of slack: `mid_block` is an ESTIMATE (prompt//4//blk) and the
                      # chat template's prefix shifts the real position by a block or two, which is
                      # exactly how a "not selected" false negative was produced before.
                      "mid_selected": any(b in sel_blk for b in
                                          (mid_block - 2, mid_block - 1, mid_block,
                                           mid_block + 1, mid_block + 2)),
                      "end_selected": any(b in sel_blk for b in
                                          (end_block - 2, end_block - 1, end_block,
                                           end_block + 1, end_block + 2)),
                      "count": len(sel_blk)})
        st_keep = set(st.row) if st.row else old_owned
        self._release_slots(req, st_keep)
        self.log(record)
        return st

    # --- attention-only seq_lens override -----------------------------------------------
    def attention_seq_lens(self, base):
        if not self.install:
            return base
        """`base` with our requests' rows replaced by their compacted lengths.

        Only the full-attention metadata builder consumes this: the hybrid model's GDN layers
        index their recurrent state by the TRUE sequence length, so changing the shared tensor
        would corrupt them (verified: sparse-on produced gibberish until this was split).
        """
        import torch

        if not self._seq_override:
            return base
        out = base.clone()
        for row, value in self._seq_override.items():
            if row < out.numel():
                # The kernel reads ceil(seq_len / page) row entries. The row is synchronised with
                # the engine's own cursor at step entry, so this only bites when the engine is one
                # append ahead of us - and reading past the row is an illegal access, so the clamp
                # is the lesser evil (it costs the newest chunk's attention, not the process).
                try:
                    for _req, _st in self._state.items():
                        if getattr(_st, "row_index", None) == row and _st.row:
                            _cap = len(_st.row) * max(1, int(self.page_tokens))
                            if int(value) > _cap:
                                if self.log is not None:
                                    self.log({"kind": "kv_len_clamped", "req": _req,
                                              "seq_len": int(value), "cap": int(_cap)})
                                value = _cap
                            break
                except Exception:
                    pass
                out[row] = value
                # Read-side diagnostic: the kernel attends to `value` positions, i.e. it reads
                # ceil(value / page) entries of the row. More than the row has = illegal access that
                # neither the write-address probe nor the row-length probe can see.
                try:
                    need = (int(value) + int(self.page_tokens) - 1) // max(1, int(self.page_tokens))
                    for _req, _st in self._state.items():
                        if getattr(_st, "row_index", None) == row and _st.row:
                            if need > len(_st.row):
                                self.log({"kind": "kv_len_past_row", "req": _req,
                                          "seq_len": int(value), "need": int(need),
                                          "row": len(_st.row),
                                          "deficit": int(need - len(_st.row))})
                except Exception:
                    pass
        return out

    # --- page allocation ----------------------------------------------------------------
    def _engine_pages(self, req):
        """Page ids the ENGINE still references in any block-table row (ours excluded).

        The runtime installs a compacted view into its own device slots, but the engine keeps
        allocating blocks from the same pool bottom-up. Taking a slot that the engine still owns
        makes a batched page move copy onto its own source (or onto another request's live page):
        measured as a 53K-token prompt hanging with the GPU idle.
        """
        st = self._state.get(req)
        mine = getattr(st, "row_index", None) if st is not None else None
        used = set()
        try:
            for table in self.view.block_table_rows():
                rows = table.tolist()
                for row_index, row in enumerate(rows):
                    if mine is not None and row_index == mine:
                        continue
                    for value in row:
                        if value:
                            used.add(int(value))
        except Exception:
            return set()
        return used

    def _take_slots(self, req, count, keep=()):
        # Allocate DOWNWARD from the top of the pool: the engine allocates its own blocks
        # bottom-up, so scanning up (the old behaviour) handed us pages the engine was still
        # using for this very request. A batched page move then had overlapping source and
        # destination sets and clobbered pages before they were read - measured as the second
        # line of a repeated word turning into noise on the very first reselect.
        slots = []
        avoid = self._engine_pages(req)
        candidate = self.view.num_blocks() - 1
        while len(slots) < count:
            if candidate < self.first_page:
                raise RuntimeError("device slot pool exhausted")
            owner = self._slot_owner.get(candidate)
            # Another request's slot is off limits; one of OUR slots may be reused only when it is
            # not a live source (`keep` = the pages still referenced by slot_of).
            usable = candidate not in avoid and (
                owner is None or (owner == req and candidate not in keep))
            if usable:
                self._slot_owner[candidate] = req
                self._owned_slots.setdefault(req, set()).add(candidate)
                slots.append(candidate)
            candidate -= 1
        return slots

    def _release_slots(self, req, keep):
        """Drop the request's slots that are not in `keep`, returning them to the shared pool."""
        owned = self._owned_slots.get(req)
        if not owned:
            return
        for slot in list(owned):
            if slot in keep:
                continue
            owned.discard(slot)
            if self._slot_owner.get(slot) == req:
                self._slot_owner.pop(slot, None)

    def _free_slots(self, req, selected, count):
        """Physical slots not holding a selected page (the spare for a new token page)."""
        used = set(range(len(selected)))
        slots = []
        candidate = len(selected)
        while len(slots) < count:
            if candidate not in used:
                slots.append(candidate)
            candidate += 1
        return slots
