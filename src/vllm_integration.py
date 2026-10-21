# Out-of-tree factory-hook integration: injects the KVMem adapter into a running vLLM engine
# at engine-safe boundaries, without modifying the vLLM source. The hook wraps
# GPUModelRunner.execute_model so the spill/restore cycle runs after the KV writes and before
# the next step, and is disabled while CUDA graphs are being captured.
#
# The sync/async choice comes from KVMEM_ASYNC_TRANSFER (1/true => the pinned-staging transfer
# arm); unset/0 keeps the blocking arm. Every rank reads the same env var, so a TP=2 pair runs
# the same arm.
import os

from vllm_handle import VllmKvHandleVllm
from policy_store import DeterministicPolicyStore

import vllm_kvmem_ext as ext
from vllm_kvmem_ext import (
    VllmMemoryBackend,
    MemorySession,
    MemoryBudget,
    MemoryPhase,
)


def _async_transfer_requested():
    value = os.environ.get("KVMEM_ASYNC_TRANSFER", "0").strip().lower()
    return value not in ("", "0", "false", "no", "off")


class KvmemAdapter:
    """Per-rank KVMem adapter bound to one vLLM engine (one TP rank)."""

    def __init__(self, runner, pool_capacity_bytes, session_id, block_tokens,
                 num_full_attn_layers, num_kv_heads, head_dim, dtype_bytes,
                 resident_budget_tokens=None, async_transfer=None, pages_per_block=1,
                 kv_caches=None):
        self._runner = runner
        self._geometry = None
        self._handle = None
        self._backend = None
        self._session = None
        self._pool_capacity = pool_capacity_bytes
        self._session_id = session_id
        self._block_tokens = block_tokens
        self._num_layers = num_full_attn_layers
        self._kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._dtype_bytes = dtype_bytes
        self._capture_depth = 0  # >0 while CUDA graphs are capturing
        # The GPU KV budget in tokens (how much of the conversation stays resident).
        self._resident_budget_tokens = resident_budget_tokens
        # The deterministic policy: a pure function of the conversation state, so both ranks
        # decide the same working set without any cross-rank communication.
        self._policy = DeterministicPolicyStore(block_tokens)
        self._async_transfer = (_async_transfer_requested() if async_transfer is None
                                else async_transfer)
        # mamba_cache_mode=align pads a block to several 16-token pages; the KV tensor's first
        # dim is PAGES, so a logical block maps to a page range.
        self._pages_per_block = max(1, int(pages_per_block))
        # Only the paged full-attention tensors are spillable token payload; the GDN layers carry
        # Mamba state blocks of a different shape and must never enter the handle.
        self._kv_caches = kv_caches

    def bind(self):
        """Build the handle + backend + session from the engine's live KV tensors."""
        geom = ext.Geometry(self._num_layers, self._kv_heads, self._head_dim,
                            self._block_tokens, self._dtype_bytes)
        kv_caches = self._kv_caches if self._kv_caches is not None else self._runner.kv_caches
        # Both model runners: V1 keeps input_batch.block_table.block_tables, V2 keeps a
        # BlockTables object at runner.block_tables (its entries expose `.gpu`).
        block_tables = self._block_table_tensors(self._runner)
        handle = VllmKvHandleVllm(
            kv_caches=kv_caches,
            block_tables=block_tables,
            geometry=geom,
            session_id=self._session_id,
            incarnation=1,
            catalog=[],        # populated by the engine as blocks commit
            resident=[],       # populated by the engine
            frontier=0,
            execution_history=1,
            history_revision=1,
            pages_per_block=self._pages_per_block,
        )
        self._handle = handle
        self._backend = VllmMemoryBackend(handle, self._pool_capacity,
                                          f"vllm-rank-{self._session_id}",
                                          self._async_transfer)
        self._session = MemorySession(self._backend)
        self._geometry = geom
        return self

    def on_execute_model_done(self, scheduler_output):
        """Called after execute_model's KV writes; runs the spill/restore cycle if due."""
        if self._capture_depth > 0 or self._session is None:
            return None
        # The policy store decides the working set (a pure function of conversation state);
        # the session then spills/restores to match it.
        stamp = self._backend.stamp()
        selected, mandatory, budget = self._policy_next_working_set(scheduler_output)
        if selected is None:
            return None
        self._session.prepare(stamp, selected, mandatory, budget)
        phase = self._session.advance()
        while phase in (MemoryPhase.Spilling, MemoryPhase.Restoring):
            phase = self._session.advance()
        if phase == MemoryPhase.Ready:
            return self._session.publish()
        return None

    def begin_capture(self):
        self._capture_depth += 1

    def end_capture(self):
        self._capture_depth = max(0, self._capture_depth - 1)

    # --- policy: a pure function of the conversation state ------------------------------
    def _policy_next_working_set(self, scheduler_output):
        """Derive (selected, mandatory, budget) from the committed catalog + the step's blocks.

        Pure: reads only the handle's catalog (the conversation state) and the blocks the step
        touched; never the host-pool occupancy. Both ranks therefore decide identically.
        """
        if self._handle is None or self._resident_budget_tokens is None:
            return None, None, None
        catalog = self._handle.catalog()
        if not catalog:
            return None, None, None
        committed = [(b.id, b.original_start) for b in catalog]
        mandatory = tuple(sorted(self._step_block_ids(scheduler_output, catalog)))
        decision = self._policy.next_working_set(
            committed, self._resident_budget_tokens, mandatory)
        budget = self._budget_for(decision, catalog)
        return list(decision.selected), list(decision.mandatory), budget

    def _step_block_ids(self, scheduler_output, catalog):
        """The blocks this step needs resident (the mandatory frontier), deterministically.

        Preferred source: the active requests' block tables (what the engine will read). If the
        runner does not expose them, fall back to the newest blocks of the catalog, which is a
        pure function of the conversation state as well.
        """
        ids = set()
        try:
            for table in self._block_table_tensors(self._runner):
                raw = table.tolist() if hasattr(table, "tolist") else table
                for row in raw:
                    ids.update(int(b) for b in row if b >= 0)
        except Exception:
            ids = set()
        known = {b.id for b in catalog}
        ids &= known
        if not ids:
            # Deterministic fallback: the newest committed block(s).
            ids = {max(catalog, key=lambda b: b.original_start).id}
        return ids

    @staticmethod
    def _block_table_entries(runner):
        """(cache_group_id, block_table_tensor) for both model runners.

        V1 keeps them under input_batch.block_table.block_tables; the V2 runner keeps a
        BlockTables object at runner.block_tables whose entries expose `.gpu`. The group id
        matters: the Mamba state groups have their own numbering, so mixing them in would invent
        block ids the paged attention tensors do not have.
        """
        holders = []
        batch = getattr(runner, "input_batch", None)
        holders.append(getattr(getattr(batch, "block_table", None), "block_tables", None))
        holders.append(getattr(getattr(runner, "block_tables", None), "block_tables", None))
        entries = []
        for holder in holders:
            for gi, table in enumerate(holder or []):
                for attr in ("gpu", "block_table"):
                    raw = getattr(table, attr, None)
                    if raw is not None:
                        entries.append((gi, raw))
                        break
        return entries

    @staticmethod
    def _block_table_tensors(runner):
        return [tensor for _, tensor in KvmemAdapter._block_table_entries(runner)]

    def _budget_for(self, decision, catalog):
        by_id = {b.id: b for b in catalog}
        keep = list(decision.selected)
        selected_tokens = sum(by_id[b].valid_tokens for b in keep)
        # The plan requires the working set PLUS the per-step reserve, so the device budget must
        # cover both (the session rounds each independently to a page).
        reserve_tokens = max(by_id[b].valid_tokens for b in keep)
        # The core budgets the WHOLE retained host set plus this plan's new spills, so the host
        # budget covers every committed block that is not both resident AND kept.
        resident = set(self._handle.resident())
        hot = set(keep) & resident
        return MemoryBudget(selected_tokens, reserve_tokens,
                            selected_tokens + 2 * reserve_tokens,
                            (len(by_id) - len(hot)) * self._record_bytes())

    def _record_bytes(self):
        g = self._geometry
        return g.block_tokens * g.num_kv_heads * g.head_dim * g.dtype_bytes * 2 * g.num_layers
