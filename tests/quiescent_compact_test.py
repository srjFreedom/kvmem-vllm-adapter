"""Contract tests for the EngineCore-callable quiescent compaction.

EngineCore calls this on its own thread after the previous output and before the next schedule:
the worker must first commit the completed chunk, refuse a stale frontier, then move pages and
return a proposal WITHOUT touching the BlockChannel (Core applies the books).
"""
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, "src")
from kvmem_sparse_runtime import VllmSparseRuntime, _ReqState

failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def main():
    rt = VllmSparseRuntime.__new__(VllmSparseRuntime)
    st = _ReqState()
    st.tokens = 147456
    st.seq_len = 36864
    calls = []

    def commit(req, n, prefill=False):
        calls.append(("commit", int(n), bool(prefill)))
        st.tokens += int(n)

    def reselect(req, prefill_pressure=False, send=True, **kwargs):
        calls.append(("reselect", bool(prefill_pressure), bool(send),
                      bool(getattr(rt, "_in_quiescent", False))))
        calls.append(("entry_books", int(st.tokens), int(st.seq_len)))
        # Simulate the move: the real _reselect rewrites the row/books to the compacted view.
        st.row = (9, 8, 7)
        st.logical = (4608, 4609, 4610)
        st.seq_len = 3 * 32
        rt._orch["r"].resident = {4608, 4609, 4610}
        rt._orch["r"].slot_of = {4608: 9, 4609: 8, 4610: 7}
        rt._last_proposal = {"visible_tokens": int(st.seq_len),
                             "physical_pages": [1, 2], "logical_frontier": int(st.tokens),
                             "resident_frontier": calls[-1][2]}

    class FakeStore:
        def __init__(self):
            self.pages = {4608: 1, 4609: 1, 4610: 1, 4631: 1, 5153: 1}

    class FakeOrch:
        def __init__(self):
            self.store = FakeStore()
            self.resident = set()
            self.slot_of = {}
            self.baked_delta = {}
            self.restored = []

        def restore_page(self, page, slot):
            self.restored.append((page, slot))

    rt._state = {"r": st}
    rt._prev_n = {"r": 4096}
    rt._prompt_len = {"r": 0}
    rt._orch = {"r": FakeOrch()}
    rt._owned_slots = {"r": [9, 8, 7]}
    rt.page_tokens = 32
    rt.after_step = commit
    rt._reselect = reselect
    rt.log = lambda record: None

    # The chunk completed but was not committed yet: the quiescent call commits it first, then
    # moves pages without the socket.
    st.tokens = 143360
    proposal = rt.quiescent_compact("r", 147456)
    check(calls[0] == ("commit", 4096, True), f"completed chunk is flushed first ({calls[0]})")
    check(calls[1] == ("reselect", True, False, True), f"pages move without BlockChannel ({calls[1]})")
    check(proposal and proposal.get("resident_frontier") == 36864
          and proposal.get("visible_tokens") == 96,
          f"proposal reports the pre-move resident book ({proposal})")
    check(rt._prev_n["r"] == 0, "the flushed chunk is not committed twice")

    # A frontier that no longer matches the executed book must be refused, not compacted.
    calls.clear()
    st.tokens = 147456
    rt._prev_n["r"] = 0
    check(rt.quiescent_compact("r", 151552) is None, "stale frontier is refused")
    check(calls == [], f"no work happens for a stale frontier ({calls})")

    # An unknown request (already finished) is a no-op for Core.
    check(rt.quiescent_compact("missing", 0) is None, "unknown request is a no-op")

    # With the prompt complete, the reselect must use the SCORE-driven picker (pressure=False).
    calls.clear()
    st.tokens = 147456
    rt._prev_n["r"] = 0
    rt._prompt_len = {"r": 147456}
    rt.quiescent_compact("r", 147456)
    check(calls[0] == ("reselect", False, False, True),
          f"final reselect routes to select_pages, not the pressure picker ({calls[0]})")
    # Mid-prefill (prompt incomplete) keeps the pressure path.
    calls.clear()
    st.tokens = 61440
    rt._prompt_len = {"r": 147456}
    rt.quiescent_compact("r", 61440)
    check(calls[0] == ("reselect", True, False, True),
          f"mid-prefill keeps the pressure picker ({calls[0]})")

    # --- refusal must not diverge the two books ----------------------------------------
    # Core's resident book that disagrees with the worker's is refused BEFORE anything moves:
    # one moved-then-refused proposal used to poison every later boundary for the request
    # (measured live: worker 20,161 vs Core 54,326, twelve consecutive "frontier differs").
    calls.clear()
    st.tokens = 164918
    st.seq_len = 54326
    rt._prev_n["r"] = 0
    rt._prompt_len = {"r": 164918}
    check(rt.quiescent_compact("r", 164918, 143360) is None,
          "resident disagreement refuses the compact")
    check(calls == [], f"refusal happens BEFORE any move ({calls})")

    # A proposal that Core then refuses is rolled back: the moved row and books return to the
    # snapshot and the store re-installs the old pages (pure copies).
    calls.clear()
    # Pre-move state captured BEFORE the proposal runs its move.
    st.row = (101 + i for i in range(0))  # placeholder cleared below
    st.row = tuple(range(101, 101 + 1152))
    st.logical = tuple(range(1152))
    st.seq_len = 54326
    rt._orch["r"].resident = set(range(1152))
    rt._orch["r"].slot_of = {p: 101 + p for p in range(1152)}
    row_before, seq_before = st.row, st.seq_len
    res_before = set(rt._orch["r"].resident)
    ok = rt.quiescent_compact("r", 164918, 54326)
    check(ok is not None, "matched books propose")
    check(st.seq_len == 96 and st.row == (9, 8, 7), "the proposal moved the view")
    check(rt.revert_quiescent("r") is True, "revert restores after a refusal")
    check(st.row == row_before and st.seq_len == seq_before,
          f"books back to the pre-proposal state (seq={st.seq_len} row={len(st.row)})")
    check(rt._orch["r"].resident == res_before, "orchestrator resident rewound")
    check(len(rt._orch["r"].restored) >= 3,
          f"old view re-installed from the store ({len(rt._orch['r'].restored)} copies)")
    check(rt.revert_quiescent("r") is False, "revert is one-shot (snapshot consumed)")

    # Core's `final` flag is the authority even when the worker's `_prompt_len` book is empty
    # (the plen hook did not fire live: every prompt-end reselect took the recency picker).
    calls.clear()
    st.tokens = 164918
    st.seq_len = 54326
    rt._prev_n["r"] = 0
    rt._prompt_len = {}
    rt.quiescent_compact("r", 164918, 54326, final=True)
    check(calls[0] == ("reselect", False, False, True),
          f"final flag selects the score picker without _prompt_len ({calls[0]})")
    calls.clear()
    # reset the books the previous proposal moved before asking again with a different flag.
    st.tokens = 164918
    st.seq_len = 54326
    rt.quiescent_compact("r", 164918, 54326, final=False)
    check(calls[0] == ("reselect", True, False, True),
          f"Core says not final -> pressure picker ({calls[0]})")

    # ---------- rewind notify: view-coord basis flag on the state ----------
    check(getattr(st, "applied_view_coord", False) is not True,
          "fresh state starts in true-position coordinates")
    rt.log = lambda record: None
    rt.book_rewound("r", 20160)
    check(getattr(rt._state["r"], "applied_view_coord", False) is True,
          "book_rewound marks the request as living in VIEW coordinates")

    print("PASS quiescent_compact_test" if not failed else "FAILED quiescent_compact_test")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
