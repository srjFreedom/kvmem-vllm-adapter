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
                 "row_index", "prefill_done")

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


class VllmSparseRuntime:
    def __init__(self, view: RunnerView, selector, page_tokens, head_dim, rotary_dim, base,
                 k_slice=None, reselect_interval=64, log=None, first_page=0,
                 select_block_tokens=128, budget_tokens=32768, gen_reserve=16384,
                 host_mib=16384, use_kvmem_plan=False, needle_trace=0, use_kvmem_host=True,
                 sessions=4):
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
        self.host_mib = int(host_mib)
        self.use_kvmem_plan = bool(use_kvmem_plan)
        self.needle_trace = int(needle_trace)
        self.use_kvmem_host = bool(use_kvmem_host)
        self.sessions = int(sessions)
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
        # Engine-reported num_computed_tokens per request: authoritative for ACCEPTED tokens.
        # Counting `num_scheduled_tokens` drifts under MTP (draft tokens that get rejected are
        # counted but never appended), which fires the R cadence early.
        self._engine_computed = {}
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
        for logical, page in enumerate(values):
            page = int(page)
            if page > 0 and logical not in page_map:
                page_map[logical] = page
            elif page > 0 and logical in page_map and page_map[logical] != page:
                # The engine moved this logical page (its own reordering): trust the engine.
                page_map[logical] = page

    def _ensure(self, req):
        if req not in self._orch:
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
            if self.retrieval_enabled:
                from kvmem_retrieval_index import KvmemRetrievalIndex
                index = KvmemRetrievalIndex(self.select_block_tokens, self.page_tokens,
                                            self.head_dim, len(kv_tensors))
                self._retrieval[req] = index
                orch.retrieval_index = index
                if orch.planner is not None:
                    orch.planner.pre_rope_statistics = True
            orch.session_id = session      # identity used by the shared H pool on release
            self._orch[req] = orch
            self._owned_slots[req] = set()
            self._state[req] = _ReqState()

    def retrieval_index(self, req):
        return self._retrieval.get(req)

    def live_requests(self):
        """Requests the runtime still holds state for (for the worker-side sweep)."""
        return list(self._orch.keys())

    def release(self, req):
        """Drop a finished request and give its H share back to the shared pool."""
        st = self._state.get(req)
        row = getattr(st, "row_index", None) if st is not None else None
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
        self._dirty_pages.pop(req, None)
        self._engine_computed.pop(req, None)
        self._pending_frontier.pop(req, None)
        self._pending_new.pop(req, None)
        self._prev_n.pop(req, None)
        if orch is not None and getattr(self, "budget_pool", None) is not None:
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
    def before_step(self, req, n_query_tokens=1, row_index=None, position_slots=None):
        """Install the request's sparse view. `row_index` is the batch row of the request and
        `position_slots` the positions[] indices of its query tokens (live runner: derived from
        idx_mapping and query_start_loc)."""
        self._ensure(req)
        st = self._state[req]
        # Without in-request compaction the sparse view must NOT be installed: the block table is
        # vLLM's own, and overriding seq_lens/positions on top of an uncompacted row is exactly the
        # half-initialised state that produced gibberish answers (measured with
        # --kvmem-inrequest-compaction off). Retrieval statistics can still be collected.
        if not bool(getattr(self, "inrequest_compaction", False)):
            return None
        if not st.row:
            return None
        row_index = req if row_index is None else int(row_index)
        if position_slots is None:
            position_slots = [req * n_query_tokens + j for j in range(n_query_tokens)]
        st.row_index = row_index
        self._seq_override[row_index] = st.seq_len + len(position_slots)
        if not self.install:
            return st
        # seq_lens must COVER the tokens of this step: vLLM expects the number of computed
        # tokens INCLUDING the current query. Installing the previous end (what we did before)
        # made every chunk's queries blind to their own chunk — the KV was written but never
        # attended, which silently produced gibberish instead of an error.
        self._install(row_index, st.row, st.seq_len + len(position_slots), list(position_slots))
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

    def advance_tokens(self, req, n_new):
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
                      "due": bool(orch.should_reselect())})
        return st

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
        engine_c = self._engine_computed.get(req)
        if engine_c is not None and engine_c > 0:
            # Reset to the engine's count every step: `num_scheduled_tokens` includes speculative
            # drafts that may be rejected, so counting them drifts ahead of the real sequence.
            # `engine_c` already INCLUDES the step being committed here, so subtract it first - the
            # commit loop below adds it back. Setting it wholesale double-counted every step
            # (measured: runtime tokens 9638 vs engine 8112, and the two TP ranks disagreed about
            # which pages to select).
            self._state[req].tokens = max(0, int(engine_c) - int(n_new_tokens or 0))
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
                orch.commit_written_page(logical, slot)
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
                touched[logical] = slot
            for logical, slot in touched.items():
                orch.commit_written_page(logical, slot)
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
            # The prompt is IMMUTABLE for this backend: vLLM derives the output length from
            # `num_tokens - prompt_len` and requires `_all_token_ids` to cover the prompt, so a
            # compaction may only ever drop GENERATED tokens. Remember the prefill length here so
            # every reselect can declare the prompt pages mandatory.
            orch.prompt_tokens = max(0, st.tokens - n_new_tokens)
        # During a long prefill the resident set must stay bounded even though relevance is not
        # meaningful yet: keep the sink prefix + newest tail and spill the middle (kvmem's
        # prefill-pressure selection). Without this a 200K prefill would consume the whole pool.
        if (prefill and st.tokens and orch.selector is not None
                and bool(getattr(self, "inrequest_compaction", False))):
            high = self.prefill_high_watermark
            if len(orch.resident) > high:
                self._reselect(req, prefill_pressure=True)
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
                      "planner": orch.planner is not None, "due": bool(due)})
        # Compaction is DECODE-only and TRUNCATION-driven: reselect when the R cadence is due, which
        # is exactly when the pool (B history + R output) is full and the visible set must shrink.
        # Building an "initial view" on the first decode step is NOT done: it truncates nothing
        # (visible == the whole book) and installing a compacted row anyway desynchronised the two
        # books - measured as an 8K prompt stalling right after prefill while the same prompt
        # completed with compaction off.
        if bool(getattr(self, "inrequest_compaction", False)) and not prefill and due:
            result = self._reselect(req)
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

    def _reselect(self, req, prefill_pressure=False):
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
        for slot in self._owned_slots[req]:
            self._slot_owner.pop(slot, None)
        self._owned_slots[req] = set()
        # Reserve exactly what the selection needs (+1 spare for the next token page). Asking for
        # the whole budget (budget_pages()+1 = 3329 pages at B=53248) drained the pool for every
        # request and left later requests reading pages they did not own.
        taken = {}

        def _take(needed):
            slots = self._take_slots(req, int(needed) + 1)
            taken["slots"] = slots
            return slots[:-1]

        prompt_tokens = int(getattr(orch, "prompt_tokens", 0) or 0)
        mandatory = list(range(0, -(-prompt_tokens // self.page_tokens))) if prompt_tokens else []
        # Force the prompt pages only when the prompt fits the budget. If it does not, the entry
        # policy failed to trim it; forcing them would make the view exceed B and the plan would be
        # rejected anyway (the compact also refuses to go below the prompt).
        forced = mandatory if prompt_tokens and prompt_tokens <= int(self.budget_tokens) else []
        stats = orch.reselect(slots=_take, mandatory_pages=mandatory,
                              forced_pages=forced, prefill_pressure=prefill_pressure)
        selected = list(orch.last_view.row_pages)
        used = list(orch.slots[:len(selected)])
        spare = taken["slots"][-1]
        st.row = tuple(used) + (spare,)
        st.logical = tuple(selected) + (None,)
        st.spare = spare
        st.seq_len = min(st.tokens, len(selected) * self.page_tokens)
        # V4 needle: sample a few resident pages and verify the device copy is exactly one
        # rotation of the stored bytes. Cheap (a handful of pages) and catches any drift.
        sample = sorted(orch.resident)[:2] + sorted(orch.resident)[-2:]
        checks = {page: orch.needle(page) for page in sample}
        self.log({"kind": "needle", "req": req, "pages": len(checks),
                  "ok": all(v is True for v in checks.values()),
                  "detail": {str(k): bool(v) for k, v in checks.items()}})
        record = {"kind": "reselect", "req": req, "selected": stats["selected"],
                  "evicted": stats["evicted"], "resident": stats["resident"],
                  "seq_len": st.seq_len, "tokens": st.tokens,
                  "prompt": prompt_tokens, "mandatory_pages": len(mandatory),
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
                res = ch.compact(req, selected, self.page_tokens)
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

    def _take_slots(self, req, count):
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
            if candidate not in self._slot_owner and candidate not in avoid:
                self._slot_owner[candidate] = req
                self._owned_slots[req].add(candidate)
                slots.append(candidate)
            candidate -= 1
        return slots

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
