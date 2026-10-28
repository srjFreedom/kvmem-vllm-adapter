"""Offline end-to-end check of park -> resume (no engine, no ext build).

Turn 1 fills a session's KV into the host store and is RELEASED; with session reuse on, release
must PARK it (KV pages + the request's host share kept alive) under the prompt's block-hash
sequence instead of dropping it.

Turn 2 is the same conversation plus new tokens. EngineCore has already turned the hit into the two
books (`num_logical_tokens` = hit, `num_computed_tokens` = |sel| pages), and the engine allocated
that many pages. `before_step` must therefore:
  * adopt the parked orchestrator (not build a new one),
  * restore exactly |sel| pages from the store into the engine's own row, at their compacted ranks,
  * leave the served/resident books consistent with what the scheduler decided.

Run: python tests/session_reuse_flow_test.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kvmem_session_store import SessionStore  # noqa: E402
from kvmem_sparse_orchestrator import SparseOrchestrator  # noqa: E402
from kvmem_sparse_rope import _quantize  # noqa: E402
from kvmem_sparse_runtime import RunnerView, VllmSparseRuntime  # noqa: E402

PAGE, HEADS, HD, ROT, CONTENT, PAGES = 16, 2, 256, 64, 512, 4096
BLOCK = 32          # hash block size for the test (the engine uses 1616)
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


class StubSelector:
    def set_context_tokens(self, tokens):
        pass

    def set_block_scores(self, scores):
        pass

    def budget_pages(self):
        return 256

    def select_pages(self, mandatory=()):
        return sorted(int(p) for p in mandatory)

    def select_prefill_pressure_pages(self, mandatory=()):
        return self.select_pages(mandatory)


class FakeView(RunnerView):
    def __init__(self):
        self.kvs = [torch.zeros(PAGES, HEADS, PAGE, CONTENT, dtype=torch.uint8)]
        self._rows = [torch.zeros(1, PAGES, dtype=torch.int32)]
        self._seq = torch.zeros(1, dtype=torch.int32)
        self._pos = torch.zeros(8192, dtype=torch.int64)
        self.row = [int(v) for v in self._rows[0][0].tolist()]

    def paged_kv(self):
        return self.kvs

    def block_table_rows(self):
        return self._rows

    def seq_lens(self):
        return self._seq

    def positions(self):
        return self._pos

    def num_blocks(self):
        return PAGES

    def allocate(self, count, base):
        """Give the request `count` pages starting at `base`, as the engine's allocator would."""
        for i in range(count):
            self._rows[0][0, i] = base + i


def canon(page):
    g = torch.Generator().manual_seed(4242 + int(page))
    k = torch.randn(HEADS, PAGE, HD, generator=g) * 0.5
    v = torch.randn(HEADS, PAGE, HD, generator=g) * 0.5
    return _quantize(torch.cat([k, v], dim=-1).to(torch.float32))


def main():
    view = FakeView()
    rt = VllmSparseRuntime(
        view, StubSelector(), PAGE, HD, ROT, 1e7,
        budget_tokens=2048, gen_reserve=256, sink_tokens=256, recent_tokens=128,
        sessions=2, host_mib=4096, pages_per_block=1, no_prefill_compact=True)
    rt.inrequest_compaction = True          # the view is only installed in this mode
    store = SessionStore(max_sessions=2)
    rt.enable_session_reuse(store, BLOCK)

    # --- turn 1: a 256-token prompt prefilled at TRUE positions ---------------------------
    ids1 = list(range(1000, 1256))
    total1 = len(ids1)
    pages1 = total1 // PAGE
    rt.note_prompt_ids("turn1", ids1)
    rt._ensure("turn1")
    orch = rt._orch["turn1"]
    orch.session_id = 1
    for p in range(pages1):
        orch.write_page(p, canon(p), p + 101, baked=0)   # engine ids start at 101
    orch.note_tokens(total1)
    rt._state["turn1"].row_index = 0

    check(rt.release("turn1") is True, "turn 1 is released")
    check(len(store) == 1, f"the session is parked, not dropped (len={len(store)})")
    entry, blocks = store.lookup(__import__("kvmem_session_store").block_hash_sequence(ids1, BLOCK))
    check(blocks == len(ids1) // BLOCK, f"the parked key covers the whole prompt ({blocks} blocks)")
    check(entry.orchestrator is orch, "the parked entry IS the turn-1 orchestrator")

    # --- turn 2: same conversation + 64 new tokens ---------------------------------------
    ids2 = ids1 + list(range(5000, 5064))
    hit = (len(ids1) // BLOCK) * BLOCK          # EngineCore's hit length
    selected = 8                                # |sel| pages the resident book was set to
    visible = selected * PAGE
    check(visible < hit, "the resident book is SHORTER than the served count (compression)")

    for i in range(selected):
        view._rows[0][0, i] = 0
    view.allocate(selected, 700)                # the engine's fresh pages for this request

    rt.note_prompt_ids("turn2", ids2)
    rt.note_resume("turn2", hit, visible)
    rt._ensure("turn2")
    check(rt._orch.get("turn2") is orch, "turn 2 ADOPTS the parked orchestrator")
    check(len(store) == 0, "the parked entry is taken out of the store (no double use)")

    st = rt.before_step("turn2", n_query_tokens=1, row_index=0, position_slots=[0])
    check(st is not None, "before_step installs a view on the resumed request")
    check(len(st.row) == selected, f"the row holds exactly |sel| pages (n={len(st.row)})")
    check(st.seq_len == visible, f"seq_len is the resident book ({st.seq_len} == {visible})")
    check(st.tokens == hit, f"the served book is the hit length ({st.tokens} == {hit})")
    check(st.row == tuple(range(700, 700 + selected)),
          "the row is the engine's own freshly allocated pages")
    restored = sum(1 for p in orch.store.pages if orch.slot_of.get(p) is not None)
    check(restored >= selected, f"at least |sel| pages are back on the device (n={restored})")
    ok_needle = all(orch.needle(p) is True for p in orch.slot_of)
    check(ok_needle, "every restored page is byte-consistent with the store")

    # The restored page must carry the RANK rotation, not an engine-id rotation.
    p = sorted(orch.slot_of)[0]
    rank = list(st.row).index(orch.slot_of[p])
    check(orch.baked_delta[p] == (rank - p) * PAGE,
          f"page {p} carries (rank-page)*page_tokens = {(rank - p) * PAGE}")

    print("session_reuse_flow_test: OK" if not failed else "session_reuse_flow_test: FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
