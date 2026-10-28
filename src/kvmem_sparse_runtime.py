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


class _ReqState:
    # row:      PHYSICAL slot ids written into the block table (what the kernel reads).
    # logical:  logical page id (true token page) occupying each row position, so a written
    #           token can be attributed to the history block it belongs to.
    __slots__ = ("row", "logical", "seq_len", "tokens", "next_page", "steps", "spare",
                 "row_index", "prefill_done", "query_reselected", "resume", "resumed",
                 "replay_snap", "replay_n", "replay_at", "replay_at_resident", "rebuilt")

    def __init__(self):
        self.row = ()
        self.logical = ()
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
        # req -> (hit_tokens, visible_tokens) decided by EngineCore for a resumed session.
        self._resume = {}
        # Optional hooks supplied by the plugin for the non-attention (GDN/Mamba) state, which is
        # NOT in the paged store: capture(req) -> snapshot, restore(req, snapshot) -> bool.
        self.capture_state = None
        self.restore_state = None
        self.reset_state = None          # zero the recurrent state (reconstruction)
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
        self.prefill_high_watermark = self.selector.budget_pages() + 64
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

    # --- state -------------------------------------------------------------------------
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
                    try:
                        from kvmem_retrieval_index import KvmemRetrievalIndex
                        _kv = self.view.paged_kv()
                        index = KvmemRetrievalIndex(self.select_block_tokens, self.page_tokens,
                                                    self.head_dim, len(_kv))
                        self._retrieval[req] = index
                        orch.retrieval_index = index
                        if orch.planner is not None:
                            orch.planner.pre_rope_statistics = True
                    except Exception:
                        pass
                try:
                    self.attach_statistics(req)
                except Exception:
                    pass
                return
            kv_tensors = self.view.paged_kv()
            session = max(1, len(self._orch) + 1)
            store = None
            # KVMem's host tier owns the spilled payload and enforces H. Falls back to the plain
            # host store when the native extension is unavailable.
            if getattr(self, "use_kvmem_host", True):
                try:
                    from kvmem_sparse_orchestrator import KvmemHostStore
                    # H is shared across the live sessions (KVMem contract), so each session gets
                    # its fair share of the pool instead of the whole budget.
                    if getattr(self, "budget_pool", None) is None:
                        from kvmem_session_budget import SessionBudgetPool
                        self.budget_pool = SessionBudgetPool(
                            host_bytes=int(self.host_mib) * 1024 * 1024,
                            sessions=max(1, int(self.sessions)))
                    share = self.budget_pool.open(session).capacity_bytes
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
                except Exception:
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
        return store

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

    def note_resume(self, req, hit_tokens, visible_tokens):
        """EngineCore decided `req` continues a parked session after `hit_tokens` tokens."""
        try:
            self._resume[req] = (int(hit_tokens), int(visible_tokens))
        except Exception:
            pass

    def _session_hashes(self, req):
        ids = self._prompt_ids.get(req)
        if not ids or not self.kv_block_size:
            return []
        from kvmem_session_store import block_hash_sequence
        return block_hash_sequence(ids, self.kv_block_size)

    def _park_session(self, req):
        """Keep a finished request's session (KV pages + recurrent state) instead of dropping it."""
        if not self.session_reuse or self.session_store is None:
            return False
        orch = self._orch.get(req)
        if orch is None:
            return False
        hashes = self._session_hashes(req)
        pages = orch.usable_pages()
        if not hashes or not pages:
            self.log({"kind": "session_park_skipped", "req": req,
                      "reason": "no hashes" if not hashes else "no pages",
                      "blocks": len(hashes), "pages": len(pages),
                      "prompt_ids": len(self._prompt_ids.get(req, ()) or ()),
                      "kv_block_size": int(self.kv_block_size)})
            return False
        state = None
        if self.capture_state is not None:
            try:
                state = self.capture_state(req)
            except Exception:
                state = None
        if self.has_recurrent and state is None:
            # Refuse to park: resuming without the recurrent state is worse than a full prefill.
            self.log({"kind": "session_park_skipped", "req": req,
                      "reason": "recurrent state unavailable"})
            return False
        covered = min(int(getattr(orch, "total_tokens", 0) or 0),
                      len(self._prompt_ids.get(req, ()) or ()) or int(orch.total_tokens))
        self.session_store.park(hashes, orch, covered, self.page_tokens,
                                mamba_state=state, session_id=getattr(orch, "session_id", None))
        # Tell EngineCore how much of this session may be brought back: the resident book it will
        # allocate is |sel| pages, so the number must be block-aligned and never exceed what the
        # store can actually fill.
        ppb = max(1, int(self.pages_per_block))
        budget = int(self.selector.budget_pages())
        # Whole-block rounding is REQUIRED: the engine allocates the resident book in whole KV
        # blocks (1616 tokens = pages_per_block pages of 16), and a book that ends mid-block made
        # the resumed turn answer with empty/garbage output (measured: 316 pages -> empty answer,
        # 303 pages -> correct answer). The cost is up to one block (<= pages_per_block tokens) of
        # the parked context not being brought back; allowing partial blocks needs the fork's row
        # mapping to be re-checked first.
        capacity = (min(len(pages), budget) // ppb) * ppb
        payload = {"hashes": [h.hex() for h in hashes], "covered": int(covered),
                   "page_tokens": int(self.page_tokens), "pages_per_block": ppb,
                   "capacity_pages": int(capacity), "has_state": state is not None,
                   "request": str(req)}
        if self.session_park is not None and capacity > 0:
            try:
                self.session_park(payload)
            except Exception:
                self.log({"kind": "session_park_publish_failed", "req": req})
        self.log({"kind": "session_parked", "req": req, "blocks": len(hashes),
                  "covered": covered, "pages": len(pages), "capacity": capacity,
                  "state": state is not None})
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
        st.resume = None
        orch = self._orch.get(req)
        if orch is None:
            return False
        count = int(visible_tokens) // self.page_tokens
        pages = self._resume_selection(orch, count, int(hit_tokens))
        self._absorb_engine_row(req)
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
        if not pages or len(row_values) < len(pages) or any(v <= 0 for v in row_values):
            self.log({"kind": "resume_no_room", "req": req, "pages": len(pages),
                      "row": len(row_values), "visible": int(visible_tokens)})
            return False
        for rank, page in enumerate(pages):
            try:
                orch.restore_page_rank(page, row_values[rank], rank)
            except Exception:
                self.log({"kind": "resume_restore_failed", "req": req, "page": int(page)})
                return False
        st.row = tuple(row_values[:len(pages)])
        st.seq_len = len(pages) * self.page_tokens
        st.tokens = int(hit_tokens)
        st.prefill_done = False
        st.resumed = True
        # KVMem's R cadence counts OUTPUT tokens from the resume point, not from the parked
        # session's last compaction.
        orch._tokens_at_reselect = st.seq_len
        orch.prompt_tokens = len(self._prompt_ids.get(req, ()) or ())
        orch.note_tokens(st.tokens)
        if self.restore_state is not None:
            try:
                self.restore_state(req, entry.mamba_state)
            except Exception:
                self.log({"kind": "resume_state_failed", "req": req})
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
        found = self.session_store.lookup(hashes)
        if found is None:
            self.log({"kind": "resume_miss", "req": req, "hit": hit_tokens})
            return None
        entry, _blocks = found
        if int(entry.covered_tokens) < int(hit_tokens):
            self.log({"kind": "resume_short", "req": req, "covered": entry.covered_tokens,
                      "hit": hit_tokens})
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
        """Drop a finished request and give its H share back to the shared pool.

        With session reuse on, the request's session is PARKED instead (KV pages + recurrent-state
        snapshot, keyed by the prompt's block hashes) and its H share is kept until the park is
        evicted, so a later turn can resume it instead of prefilling the whole conversation again.
        """
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
        parked = False
        if self.session_reuse and st is not None:
            try:
                parked = self._park_session(req)
            except Exception:
                parked = False
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
        self._resume.pop(req, None)
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
        self._ensure(req)
        st = self._state[req]
        if row_index is not None:
            # Bind the ROW even before a view exists: the first reselect must know which block
            # table row is the request's own (it compacts inside the request's own pages).
            st.row_index = int(row_index)
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
            self._complete_resume(req, st, int(row_index))
        if not st.row:
            return None
        row_index = req if row_index is None else int(row_index)
        if position_slots is None:
            position_slots = [req * n_query_tokens + j for j in range(n_query_tokens)]
        st.row_index = row_index
        if rows_only:
            # Pre-gather phase: only the block table row must be in place (the kernel's gathered
            # copy is taken from it). seq_lens/positions MUST be written after the engine's own
            # `prepare_pos_seq_lens`, because that runs with the optimistic device counter, which a
            # compaction does not rewind - writing ours first would be overwritten with a stale
            # length and the attention would read pages beyond the row.
            if self.install_row and st.row:
                import torch
                for table in self.view.block_table_rows():
                    table[row_index, :len(st.row)] = torch.tensor(
                        st.row, dtype=table.dtype, device=table.device)
            return st
        self._seq_override[row_index] = st.seq_len + len(position_slots)
        if not self.install:
            return st
        # seq_lens must COVER the tokens of this step: vLLM expects the number of computed
        # tokens INCLUDING the current query. Installing the previous end (what we did before)
        # made every chunk's queries blind to their own chunk — the KV was written but never
        # attended, which silently produced gibberish instead of an error.
        self._install(row_index, st.row, st.seq_len + len(position_slots), list(position_slots))
        if self.install_row:
            self._absorb_upto[req] = max(int(self._absorb_upto.get(req, 0)), len(st.row))
        return st

    def _install(self, row_index, row, seq_len, position_slots):
        import torch

        if self.install_row:
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
        if self.install_pos:
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

    def ensure_row_for_chunk(self, req, n_tokens):
        """Make the block table row cover the tokens this step is about to write.

        The row used to be extended in `after_step`, i.e. AFTER the model ran - so a prefill chunk
        that crossed a page boundary made the kernel read row entries that did not exist yet
        (measured: row=4864 pages for seq_len=77824 while the step wrote up to 81920 -> FlashInfer
        illegal memory access). Growing it at step entry, from the engine's own page map, keeps the
        row at least as long as `seq_len` for every step.
        """
        st = self._state.get(req)
        n = int(n_tokens or 0)
        if st is None or n <= 0:
            return 0
        try:
            last_rank = (int(getattr(st, "tokens", 0)) + n - 1) // max(1, self.page_tokens)
            if len(st.row) > last_rank:
                return 0
            self._absorb_engine_row(req)
            grown = list(st.row)
            page_map = self._page_map.get(req) or {}
            while len(grown) <= last_rank:
                logical = len(grown)
                page = page_map.get(logical)
                if page is None:
                    self._grow_from_engine(req, grown, logical)
                    page = page_map.get(logical)
                if page is None:
                    break
                grown.append(int(page))
            added = len(grown) - len(st.row)
            if len(grown) <= last_rank and self.log is not None:
                # Could not reserve the whole chunk: the engine's page map has no entry for the
                # logical pages this step will write. Logged loudly because the kernel would then
                # read row entries that do not exist (illegal memory access).
                self.log({"kind": "row_short_pre_step", "req": req,
                          "row": len(grown), "need": int(last_rank) + 1,
                          "tokens": int(getattr(st, "tokens", 0)), "n": n,
                          "page_map": len(page_map), "next_page": getattr(st, "next_page", None)})
            st.row = tuple(grown)
            # The attention length must equal what the ROW actually holds, including this chunk. The
            # runtime used to keep its own `seq_len`, and after an in-prefill compaction it drifted
            # away from the engine's num_computed by tens of thousands of tokens (measured:
            # seq_len=90112 while the engine was at 42114) - the kernel then masked/indexed against
            # a length the row does not have, which is the repetition seen on >B prompts.
            try:
                covered = len(st.row) * max(1, self.page_tokens)
                target = min(int(getattr(st, "tokens", 0)) + n, covered)
                if target > 0 and int(getattr(st, "seq_len", 0)) != target:
                    st.seq_len = target
                    if self.log is not None and added:
                        self.log({"kind": "seq_len_from_row", "req": req,
                                  "seq_len": int(target), "row": len(st.row),
                                  "tokens": int(getattr(st, "tokens", 0)), "n": n})
            except Exception:
                pass
            if added and self.log is not None:
                self.log({"kind": "row_grown_pre_step", "req": req, "added": int(added),
                          "row": len(st.row), "need": int(last_rank) + 1,
                          "tokens": int(getattr(st, "tokens", 0)), "n": n})
            return added
        except Exception:
            if self.log is not None:
                import traceback as _tb
                self.log({"kind": "row_grow_error", "req": req, "error": _tb.format_exc()[-400:]})
            return 0

    def advance_tokens(self, req, n_new):
        self._inflight[req] = int(n_new or 0)
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
            if not deferred and st.tokens > 0 and orch.should_reselect():
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
        # Pad to the engine's allocation quantum so its next block append is contiguous with the
        # row (the append index is len(blocks) * pages_per_block).
        quantum = int(self.pages_per_block)
        if quantum > 1 and pages:
            rem = len(pages) % quantum
            if rem:
                pages = pages[: len(pages) - rem]
        return pages

    def _grow_from_engine(self, req, grown, logical):
        """Map a page for `logical` from the engine's own allocation (best effort)."""
        try:
            rows = self.view.block_table_rows()[0]
            st = self._state[req]
            row = rows[st.row_index] if st.row_index < len(rows) else None
            if row is None:
                return
            slot = int(row[min(len(grown), len(row) - 1)])
            self._page_map[req][logical] = slot
        except Exception:
            return

    def after_step(self, req, n_new_tokens, prefill=False):
        self._ensure(req)
        # A step that schedules a whole page worth of tokens is a prefill chunk, not a decode
        # step (a decode step is 1 token, or a few with speculative decoding).
        frontier = int(self._pending_frontier.get(req, 0) or 0)
        if frontier:
            # Trust the scheduler's absolute frontier; derive this step's delta from it.
            _st = self._state[req]
            _st.tokens = max(_st.tokens, frontier - int(n_new_tokens or 0))
        logical_c = self._engine_logical.get(req)
        if logical_c is not None and logical_c > 0:
            # The SERVED count is the logical frontier. It is reset every step because
            # `num_scheduled_tokens` includes speculative drafts that may be rejected; it already
            # INCLUDES the step being committed here, so subtract it first (the commit loop adds it
            # back). With compaction this counter keeps growing while the resident book shrinks.
            self._state[req].tokens = max(0, int(logical_c) - int(n_new_tokens or 0))
        engine_c = self._engine_computed.get(req)
        if engine_c is not None and engine_c > 0:
            # RESIDENT book length (what positions/seq_lens must agree with).
            self._state[req].seq_len = max(0, int(engine_c) - int(n_new_tokens or 0))
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
                orch.commit_written_page(logical, slot, baked=0)
            st.tokens += n_new_tokens
            orch.note_tokens(st.tokens)
        elif st.row and n_new_tokens:
            self._absorb_engine_row(req)
            # Streaming tokens (decode, or a prefill continuing past a pressure reselect) land at
            # compacted positions starting at st.seq_len. A newly crossed page needs a fresh slot
            # appended to the row; the spare slot taken at reselect is the first of those.
            start = st.seq_len
            # A newly crossed page needs a slot appended to the row. This is correct for BOTH
            # prefill and decode (a decode step normally consumes the spare slot, so it is a
            # no-op there); gating it on the prefill flag broke streaming after a pressure
            # reselect made the row non-empty.
            last_rank = (start + n_new_tokens - 1) // self.page_tokens
            if len(st.row) <= last_rank:
                self._absorb_engine_row(req)
                page_map = self._page_map[req]
                grown = list(st.row)
                while len(grown) <= last_rank:
                    logical = len(grown)
                    page = page_map.get(logical)
                    if page is None:
                        break
                    grown.append(page)
                st.row = tuple(grown)
            touched = {}
            for i in range(n_new_tokens):
                logical = (st.tokens + i) // self.page_tokens
                rank = (start + i) // self.page_tokens
                if rank >= len(st.row):
                    # The spare slot handed out by the last reselect is used up: grow the row from
                    # the engine's own page map (the pages exist there, the row was just not
                    # extended). Only if even that fails is the row genuinely out of space.
                    grown = list(st.row)
                    while len(grown) <= rank:
                        logical = len(grown)
                        page = self._page_map[req].get(logical)
                        if page is None:
                            self._grow_from_engine(req, grown, logical)
                            page = self._page_map[req].get(logical)
                        if page is None:
                            raise RuntimeError("sparse row has no slot for the next token page")
                        grown.append(int(page))
                    st.row = tuple(grown)
                slot = int(st.row[rank])
                touched[logical] = (slot, rank)
            for logical, (slot, rank) in touched.items():
                # The engine wrote these tokens into COMPACTED positions, so the page's bytes are
                # baked `(rank - logical) * page_tokens` away from its logical start. Recording the
                # rank (not the engine page id) is what lets a later resume, or a second request
                # whose pool pages start at another base, restore the page to the right position.
                orch.commit_written_page(logical, slot,
                                         baked=(rank - logical) * self.page_tokens)
                if st.spare == slot:
                    st.spare = None
            st.tokens += n_new_tokens
            st.seq_len += n_new_tokens
            orch.note_tokens(st.tokens)

        st.steps += 1
        orch.tick()
        if not prefill and not st.prefill_done and st.tokens:
            # First decode step: anchor the R cadence at the prompt length. Without this the first
            # R check is already true (the logical total covers the prompt) and every request would
            # reselect before generating a single token; with it, reselection fires after R OUTPUT
            # tokens, exactly when the pool (B history + R output) is full.
            st.prefill_done = True
            orch._tokens_at_reselect = max(0, st.tokens - n_new_tokens)
            if (st.row and not st.query_reselected
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
                if _plen and int(st.tokens) >= _plen and not st.query_reselected:
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
            elif len(orch.resident) > high:
                try:
                    self._reselect(req, prefill_pressure=True)
                except Exception:
                    if self.log is not None:
                        import traceback as _tb
                        self.log({"kind": "prefill_reselect_error", "req": req,
                                  "error": _tb.format_exc()[-500:]})
                # Do not re-fire on the very next page: the reselect just rebuilt the working set,
                # so the next useful one is a reserve later (R tokens = 2K pages at B=53K). Without
                # this every prefill chunk compacted - 24 reselects on a 200K prompt, each moving
                # ~3.3K pages (measured: prefill fell to ~300 tok/s, 9x slower).
                self.prefill_high_watermark = len(orch.resident) + max(
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
        if bool(getattr(self, "inrequest_compaction", False)) and not prefill and due:
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

    def _trace_pages(self, req, pages):
        """Diagnostic: for a few pages, show every number the needle depends on."""
        orch = self._orch.get(req)
        st = self._state.get(req)
        if orch is None or st is None or self.log is None:
            return
        rows = []
        for page in pages[:4]:
            rows.append({"page": int(page),
                         "engine_page": int(self._page_map.get(req, {}).get(page, 0)),
                         "compacted_slot": (int(st.row.index(page)) if page in st.row else None),
                         "slot_of": int(orch.slot_of.get(page, -1)),
                         "store_delta": int(orch.store_delta.get(page, 0)),
                         "baked_delta": int(orch.baked_delta.get(page, 0)),
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

    def _reselect(self, req, prefill_pressure=False, probe=False):
        st = self._state[req]
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
        try:
            map_now = self._page_map.setdefault(req, {})
            known = orch.store.pages
            # The in-flight chunk must NOT be added here. `advance_tokens` deliberately LAGS
            # accounting by one step: at the entry of step k it commits step k-1 and only then
            # reselects, so `st.tokens` is exactly the WRITTEN frontier and `_inflight` holds the
            # count of a step whose KV has not been produced yet. Adding it made the runtime claim
            # pages that carry no bytes - which is the very failure the lagging was introduced to
            # fix ("selected pages without canonical bytes", see advance_tokens' docstring) - and
            # it inflated the frontier: a step logged tokens=42352 while the engine's own
            # num_computed_tokens for the same moment was 41149.
            # `after_step` has already committed every page up to `st.tokens` on both the plain and
            # the streaming branch, so there is nothing left to sweep.
            inflight = 0
            upto = -(-(int(st.tokens) + inflight) // self.page_tokens)
            for lp in range(upto):
                if lp in known:
                    continue
                slot = map_now.get(lp)
                if slot is None:
                    continue
                try:
                    # `page_map` entries are pre-compaction logical pages, whose bytes the engine
                    # wrote at TRUE positions.
                    orch.commit_written_page(lp, int(slot), baked=0)
                except Exception:
                    continue
            # The IN-FLIGHT chunk is not reachable through `page_map`: after a compaction the row is
            # rank-indexed, so those logical pages have no map entry. Their bytes ARE on the device,
            # at the row ranks where the step wrote them - commit them from the row, otherwise the
            # selection cannot see them and the whole chunk silently disappears from the context
            # (this is what made a compacted prefill answer with noise).
            if inflight:
                # The in-flight pages ARE in the store after this sweep, so the selection frontier
                # must cover them: the mandatory range is built from them, and a mandatory block
                # beyond `total_tokens` is filtered out of the selection and then refused by the
                # planner ("mandatory blocks must appear exactly once").
                try:
                    orch.note_tokens(int(st.tokens) + inflight)
                except Exception:
                    pass
            if st.row and inflight:
                base = int(st.seq_len)
                first = int(st.tokens)
                for i in range(inflight):
                    rank = (base + i) // self.page_tokens
                    if rank >= len(st.row):
                        break
                    lp = (first + i) // self.page_tokens
                    if lp in known:
                        continue
                    try:
                        # Post-compaction in-flight pages were written into COMPACTED positions.
                        orch.commit_written_page(
                            lp, int(st.row[rank]),
                            baked=(rank - lp) * self.page_tokens)
                    except Exception:
                        continue
        except Exception:
            pass
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
        stats = orch.reselect(slots=_take, mandatory_pages=mandatory,
                              forced_pages=forced, prefill_pressure=prefill_pressure)
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
        st.seq_len = min(st.tokens, len(selected) * self.page_tokens)
        # V4 needle: sample a few resident pages and verify the device copy is exactly one
        # rotation of the stored bytes. Cheap (a handful of pages) and catches any drift.
        if int(getattr(self, "full_needle", 0)) > 0:
            # Verify EVERY resident page, not a sample: with ~3.3K pages per compaction a
            # batched-rotation problem would show up on a few of them only.
            sample = sorted(orch.resident)
        else:
            sample = sorted(orch.resident)[:2] + sorted(orch.resident)[-2:]
        checks = {page: orch.needle(page) for page in sample}
        self.log({"kind": "needle", "req": req, "pages": len(checks),
                  "ok": all(v is True for v in checks.values()),
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
                  "prefill_pressure": stats["prefill_pressure"]}
        cap = getattr(self, "capture", None)
        if cap is not None:
            record["capture"] = {"calls": cap.calls, "wants": cap.wants,
                                 "captured": cap.captured, "errors": cap.errors}
        score_stats = getattr(orch, "last_score_stats", None)
        if score_stats:
            record["scores"] = score_stats          # proof that retrieval ranking was applied
        # Three-book reconciliation (Route A): the worker has rewritten the block-table row to the
        # compacted working set, but the engine still tracks the monotone sequence length. Tell
        # EngineCore to truncate `_all_token_ids` to the visible tokens and rewind
        # `num_computed_tokens`, so the scheduler sees no gap and does not recompute the spilled
        # tokens. Synchronous: the worker blocks until the engine applies it, so the next schedule
        # (step k+1) is already consistent.
        ch = getattr(self, "compact_channel", None)
        if ch is not None and selected:
            try:
                # Reconstruction: zero the recurrent state and rewrite the engine's token book to
                # the compacted sequence, so the selected pages are prefilled from scratch. Only at
                # the prefill boundary, where the sequence may legitimately change.
                _st_r = self._state.get(req)
                _rebuild = bool(getattr(self, "compact_rebuild", False)) and not bool(
                    getattr(_st_r, "rebuilt", False))
                if _rebuild and self.reset_state is not None:
                    try:
                        self.reset_state(req)
                    except Exception:
                        _rebuild = False
                res = ch.compact(req, selected, self.page_tokens,
                                 visible_tokens=int(st.seq_len), rebuild=_rebuild)
                if isinstance(res, dict):
                    record["compact_applied"] = bool(res.get("ok"))
                    record["compact_truncated"] = bool(res.get("truncated"))
                    truncated = bool(res.get("truncated"))
                else:
                    record["compact_applied"] = bool(res)
                    truncated = bool(res)
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
                      "mid_selected": mid_block in sel_blk,
                      "end_selected": end_block in sel_blk,
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
                out[row] = value
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
