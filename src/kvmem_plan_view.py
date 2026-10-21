"""Take the attention view from KVMem itself instead of building one by hand.

KVMem's `plan_working_set` is the authority for "which blocks stay, in which compact order, how
many tokens are visible". This module wraps the descriptor/snapshot/budget plumbing so the vLLM
runtime can use KVMem's answer directly.

Two facts learned the hard way (see NOTES 38):
  * pybind's def_readwrite returns a COPY for a std::vector member, so containers must be assigned
    whole (`snap.blocks = [...]`), never appended to in place;
  * plan_working_set validates hard: complete identity, a payload layout with Key AND Value planes
    per layer, three capability bits, a stamp with all revisions non-zero, a catalog whose blocks
    chain head-to-tail up to the frontier, and a current host copy for every block.
"""

import os

from dataclasses import dataclass

PAGE_TOKENS_DEFAULT = 16
BLOCK_TOKENS_DEFAULT = 128


def _ext():
    import vllm_kvmem_ext  # imported lazily so CPU-only tests can stub it
    return vllm_kvmem_ext


@dataclass(frozen=True)
class PlanResult:
    pages: tuple[int, ...]          # selected pages, in KVMem's compact order
    visible_tokens: int
    kept_blocks: tuple[int, ...]
    evict_blocks: tuple[int, ...]


class KvmemPlanner:
    """Builds the KVMem descriptor/catalog once and asks it for a view on every reselect."""

    def __init__(self, page_tokens=PAGE_TOKENS_DEFAULT, block_tokens=BLOCK_TOKENS_DEFAULT,
                 budget_tokens=32768, gen_reserve=16384, host_mib=16384,
                 record_bytes=None, layers=17, device_tokens=None,
                 pre_rope_statistics=False):
        if block_tokens % page_tokens:
            raise ValueError("a selection block must be a whole number of pages")
        self.page_tokens = int(page_tokens)
        self.block_tokens = int(block_tokens)
        self.budget_tokens = int(budget_tokens)
        self.gen_reserve = int(gen_reserve)
        self.layers = int(layers)
        self.pages_per_block = self.block_tokens // self.page_tokens
        # One record = every paged attention layer of one block, K and V halves.
        self.record_bytes = int(record_bytes or (self.block_tokens * 512 * self.layers))
        # The DEVICE budget is the KV pool capacity in tokens, not the conversation length:
        # the plan must fit visible + reserve into the pool.
        self.device_tokens = int(device_tokens) if device_tokens else (1 << 21)
        # Only claim the capability when a pre-RoPE capture actually feeds scores; kvmem rejects
        # retrieval selection from a backend that cannot supply statistics.
        self.pre_rope_statistics = bool(pre_rope_statistics)
        self._descriptor = None

    # --- descriptor / snapshot ---------------------------------------------------------
    def descriptor(self):
        ext = _ext()
        if self._descriptor is not None:
            return self._descriptor
        d = ext.MemoryDescriptor()
        d.identity.backend = "vllm"
        d.identity.revision = "sm75-kvmem-1"
        d.identity.model = "vllm-kvmem-backend"
        d.identity.state_schema = "fp8-e4m3-kv-1"
        d.layout.format = "fp8-e4m3-kv"
        d.layout.version = 1
        d.layout.block_tokens = self.block_tokens
        d.layout.record_bytes = self.record_bytes
        d.layout.alignment = 16
        # record = one block of EVERY layer; each layer holds a Key and a Value half.
        layer_bytes = self.record_bytes // self.layers
        plane_bytes = layer_bytes // 2
        stride = plane_bytes // self.block_tokens
        planes = []
        for layer in range(self.layers):
            for kind_name, offset in (("Key", 0), ("Value", plane_bytes)):
                plane = ext.PayloadPlane()
                plane.layer = layer
                plane.kind = getattr(ext.PayloadPlaneKind, kind_name)
                plane.offset = layer * layer_bytes + offset
                plane.bytes = plane_bytes
                plane.alignment = 16
                plane.token_stride = stride
                planes.append(plane)
        d.layout.planes = planes
        d.capabilities.page_tokens = self.page_tokens
        d.capabilities.max_position = 1 << 21
        d.capabilities.native_payload = True
        d.capabilities.sparse_view = True
        d.capabilities.original_rope_positions = True
        d.capabilities.pre_rope_statistics = self.pre_rope_statistics
        self._descriptor = d
        return d

    def snapshot(self, total_tokens, session_id=1, resident_blocks=()):
        """Catalog of every committed block.

        `host_version` marks whether a CURRENT host copy exists. Blocks kept on the device do not
        need one (the device copy is authoritative), the rest are the retained cold set that the H
        budget has to cover.
        """
        ext = _ext()
        snap = ext.MemorySnapshot()
        snap.stamp.valid = True
        snap.stamp.session.id = max(1, int(session_id))
        snap.stamp.session.incarnation = 1
        snap.stamp.history_revision = 1
        snap.stamp.execution_history = 1
        snap.stamp.frontier = int(total_tokens)
        resident = sorted(set(int(b) for b in resident_blocks))
        blocks = []
        start = 0
        block_id = 0
        while start < int(total_tokens):
            copies = ext.BlockCopies()
            copies.block.session = snap.stamp.session
            copies.block.id = block_id
            copies.block.original_start = start
            copies.block.valid_tokens = min(self.block_tokens, int(total_tokens) - start)
            copies.block.content_version = 1
            copies.host_version = 0 if block_id in resident_blocks else 1
            blocks.append(copies)
            start += copies.block.valid_tokens
            block_id += 1
        snap.blocks = blocks                  # whole-list assignment (see module docstring)
        snap.resident = resident              # which blocks the device view currently owns
        return snap

    def plan(self, selected_pages, total_tokens, session_id=1, mandatory_pages=()):
        """Ask KVMem for the compact view over the pages the policy already chose."""
        ext = _ext()
        # The selector works at block granularity, so its choice can cross B by up to one block
        # (a 128-token block whose tail sits past the budget). kvmem's contract is strict -
        # `visible_tokens > selected_tokens` aborts the plan - so drop the lowest-ranked blocks of
        # the selection until it fits, never touching mandatory ones.
        ordered, seen = [], set()
        for page in selected_pages:
            block = int(page) // self.pages_per_block
            if block not in seen:
                seen.add(block)
                ordered.append(block)
        mandatory = sorted({int(p) // self.pages_per_block for p in mandatory_pages})
        mandatory_set = set(mandatory)

        def tokens_of(blocks):
            total = 0
            for block in blocks:
                start = block * self.block_tokens
                if start < int(total_tokens):
                    total += min(self.block_tokens, int(total_tokens) - start)
            return total

        trimmed = list(ordered)
        while tokens_of(trimmed) > self.budget_tokens:
            for index in range(len(trimmed) - 1, -1, -1):
                if trimmed[index] not in mandatory_set:
                    trimmed.pop(index)
                    break
            else:
                break                       # only mandatory blocks left: let kvmem report it
        selected_blocks = sorted(trimmed)
        # The host budget must cover EVERY retained cold block, not just the new spills: the plan
        # reports host_payload_bytes_required over all blocks that are not kept resident.
        total_blocks = -(-int(total_tokens) // self.block_tokens)
        retained = max(0, total_blocks - len(selected_blocks))
        budget = ext.MemoryBudget(self.budget_tokens, self.gen_reserve, self.device_tokens,
                                  self.record_bytes * retained)
        plan = ext.plan_working_set(
            self.descriptor(),
            self.snapshot(total_tokens, session_id, resident_blocks=set(selected_blocks)),
            selected_blocks, mandatory, budget)
        if os.environ.get("KVMEM_PLAN_TRACE") == "1":
            try:
                import sys as _sys
                print(f"[kvmem plan] total={int(total_tokens)} sel_blocks={len(selected_blocks)} "
                      f"sel_tokens={tokens_of(selected_blocks)} mandatory_blocks={len(mandatory)} "
                      f"visible={int(plan.visible_tokens)} kept_blocks={len(plan.next.blocks)} "
                      f"budget_tokens={self.budget_tokens}", file=_sys.stderr, flush=True)
            except Exception:
                pass
        kept = tuple(vb.block.id for vb in plan.next.blocks)
        pages = []
        for block in kept:
            pages.extend(range(block * self.pages_per_block,
                               (block + 1) * self.pages_per_block))
        return PlanResult(pages=tuple(pages),
                          visible_tokens=int(plan.next.visible_tokens),
                          kept_blocks=kept,
                          evict_blocks=tuple(b.block.id for b in plan.evict))
