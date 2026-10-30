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
        rt._last_proposal = {"visible_tokens": int(st.seq_len),
                             "physical_pages": [1, 2], "logical_frontier": int(st.tokens),
                             "resident_frontier": int(st.seq_len)}

    rt._state = {"r": st}
    rt._prev_n = {"r": 4096}
    rt._prompt_len = {"r": 0}
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
    check(proposal and proposal.get("visible_tokens") == 36864,
          f"proposal returned ({proposal})")
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

    print("PASS quiescent_compact_test" if not failed else "FAILED quiescent_compact_test")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
