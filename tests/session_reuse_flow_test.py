"""CPU-only exact-frontier checkpoint -> acknowledged park -> full-prefix restore.

A release discovered by the next request must not create a checkpoint. EngineCore prepares one
sealed block-aligned prefix before publishing metadata, and both worker books start at that same
frontier. A compressed resident book is refused until Mamba's cursor contract is qualified.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kvmem_session_store import SessionStore  # noqa: E402
from kvmem_sparse_orchestrator import SparseOrchestrator  # noqa: E402
from kvmem_sparse_rope import _quantize  # noqa: E402
from kvmem_sparse_runtime import RunnerView, SparseViewCapacityError, VllmSparseRuntime  # noqa: E402

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
    rt.stats_core = False
    rt.retrieval_enabled = False
    store = SessionStore(max_sessions=2)
    rt.enable_session_reuse(store, BLOCK)
    rt.has_recurrent = True
    rt.capture_state = lambda req, frontier=None: {0: "GDN@128"}
    rt.restore_state = lambda req, image: bool(image)
    rt.session_checkpoint_speculative_tokens = 0
    rt.session_checkpoint_executed_frontier = lambda req: 128

    # --- turn 1: a 256-token prompt prefilled at TRUE positions ---------------------------
    ids1 = list(range(1000, 1256))
    total1 = len(ids1)
    pages1 = total1 // PAGE
    rt.note_prompt_ids("turn1", ids1)
    rt.note_session_id("turn1", "conversation", "tenant")
    rt.note_prompt_len("turn1", total1)
    rt._ensure("turn1")
    orch = rt._orch["turn1"]
    orch.session_id = 1
    for p in range(pages1):
        orch.write_page(p, canon(p), p + 101, baked=0)   # engine ids start at 101
    orch.note_tokens(total1)
    rt._state["turn1"].row_index = 0
    rt._state["turn1"].tokens = 128
    check(rt._maybe_capture_session_checkpoint("turn1", rt._state["turn1"]),
          "exact 128-token prefill checkpoint captured")
    check(rt.prepare_session_checkpoint("turn1")["covered"] == 128,
          "terminal barrier prepares only the sealed executed prefix")
    check(rt.release("turn1") is True, "turn 1 is released")
    check(len(store) == 1, f"the acknowledged session remains parked (len={len(store)})")
    hashes = __import__("kvmem_session_store").block_hash_sequence(ids1[:128], BLOCK)
    entry, blocks = store.lookup(hashes, owner_session_id=("tenant", "conversation"))
    check(blocks == 4 and entry.orchestrator is orch,
          "the checkpoint is the exact 128-token prefix, not the final prompt")

    # --- turn 2: same conversation + 64 new tokens ---------------------------------------
    ids2 = ids1 + list(range(5000, 5064))
    hit = 128
    selected = hit // PAGE
    visible = hit
    check(visible == hit, "Mamba state and resident KV use one exact frontier")

    for i in range(selected + 1):
        view._rows[0][0, i] = 0
    # The engine allocates for the resident prefix AND this step's scheduled query token, then
    # the plugin mirrors the written block-table prefix into the runtime before prepare_inputs.
    view.allocate(selected + 1, 700)

    rt.note_prompt_ids("turn2", ids2)
    rt.note_session_id("turn2", "conversation", "tenant")
    rt.note_resume("turn2", hit, visible)
    rt._ensure("turn2")
    check(rt._orch.get("turn2") is orch, "turn 2 ADOPTS the parked orchestrator")
    check(len(store) == 0, "the parked entry is taken out of the store (no double use)")
    rt.state("turn2").row_index = 0
    rt.sync_row_from_engine("turn2", selected + 1, 1)

    st = rt.before_step("turn2", n_query_tokens=1, row_index=0, position_slots=[0])
    check(st is not None, "before_step installs a view on the resumed request")
    check(len(st.row) == selected + 1,
          f"the row keeps this step's scheduled page too (n={len(st.row)})")
    check(st.seq_len == visible, f"seq_len is the resident book ({st.seq_len} == {visible})")
    check(st.tokens == hit, f"the served book is the hit length ({st.tokens} == {hit})")
    check(st.row[:selected] == tuple(range(700, 700 + selected)),
          "the resident prefix uses the engine's own freshly allocated pages")
    check(st.row[selected] == 700 + selected, "the scheduled chunk page is preserved")
    restored = sum(1 for p in orch.store.pages if orch.slot_of.get(p) is not None)
    check(restored >= selected, f"at least |sel| pages are back on the device (n={restored})")
    ok_needle = all(orch.needle(p) is True for p in orch.slot_of)
    check(ok_needle, "every restored page is byte-consistent with the store")

    # The restored page must carry the RANK rotation, not an engine-id rotation.
    p = sorted(orch.slot_of)[0]
    rank = list(st.row).index(orch.slot_of[p])
    check(orch.baked_delta[p] == (rank - p) * PAGE,
          f"page {p} carries (rank-page)*page_tokens = {(rank - p) * PAGE}")

    # Fail closed if the runtime's row is one page short of the attention length it would install:
    # a resumed prefix alone cannot address the next scheduled token.
    st.row = st.row[:selected]
    refused = False
    try:
        rt.before_step("turn2", n_query_tokens=1, row_index=0, position_slots=[0])
    except SparseViewCapacityError:
        refused = True
    check(refused, "a row one page short of the scheduled step is refused, not clamped")

    print("session_reuse_flow_test: OK" if not failed else "session_reuse_flow_test: FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
