"""Every SOCKET compaction path (the worker-side `_reselect` with send=True) must stay silent
while Core-boundary mode (`sched_compact`) owns compaction. The 169,998-token TP2 run proved the
ungated paths fired anyway (36 applied / 12 query_reselect), moving the block table without the
EngineCore books - the double path behind the corrupted compacted view.

Offline contract per gated site:
  A. advance_tokens: the R-cadence decode reselect.
  B. after_step first-decode query_reselect.
  C. after_step post_prefill_shrink (resident prefill, prompt complete).
  D. after_step decode-due reselect.
Each runs with sched_compact OFF (call must land) and ON (no call may leave the socket path).
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


PAGE = 32
REQ = "r"


def make_rt(sched_compact, scenario):
    rt = VllmSparseRuntime.__new__(VllmSparseRuntime)
    st = _ReqState()
    calls = []

    def reselect(req, prefill_pressure=False, probe=False, rebuild=False, send=True):
        calls.append({"probe": bool(probe), "send": bool(send)})
        return st

    orch = NS(selector=object(), retrieval_index=object(), store=NS(pages={}),
              resident=list(range(5313)), tick=lambda: None,
              note_tokens=lambda n: None, prompt_tokens=0,
              _tokens_at_reselect=0, should_reselect=lambda: scenario in ("A", "D"),
              gen_reserve=16384, reselect_token_budget=36864, planner=None, last_plan=None)
    st.tokens = 169998
    st.seq_len = 36864
    st.row = tuple(range(1152))
    st.prefill_done = scenario not in ("B",)
    rt._state = {REQ: st}
    rt._orch = {REQ: orch}
    rt._reselect = reselect
    rt.state = lambda req: rt._state[req]
    rt._ensure = lambda req: None
    rt.send_replay = lambda req, n=2048: None
    rt.log = lambda record: None
    rt.page_tokens = PAGE
    rt.budget_tokens = 36864
    rt.gen_reserve = 16384
    rt.inrequest_compaction = True
    rt.no_prefill_compact = True
    rt.compact_rebuild = False
    rt.reselect_at_prompt_end = False
    rt.sched_compact = bool(sched_compact)
    rt._prompt_len = {REQ: 169998}
    rt._retrieval = {}
    rt._over_budget_logged = set()
    rt._inflight = {}
    rt._prev_n = {}
    rt._adv_logged = set()
    rt.spec_reqs = set()
    rt._pending_frontier = {}
    rt._engine_logical = {}
    rt._engine_computed = {}
    rt._pending_new = {}
    rt._prev_chunk = {}
    return rt, st, calls


def run(scenario, sched_compact, prefill):
    rt, st, calls = make_rt(sched_compact, scenario)
    if scenario == "A":
        rt.advance_tokens(REQ, 1)
    else:
        rt.after_step(REQ, 0, prefill=prefill)
    return st, calls


def main():
    # A. R-cadence decode reselect through advance_tokens. With the prompt already complete, the
    # prompt-complete socket reselect fires too (pre-existing gate) - both must land when off and
    # both must be silent when Core-boundary mode is on.
    _, calls = run("A", False, prefill=False)
    check(len(calls) == 2 and any(not c["probe"] for c in calls),
          f"A off: prompt-complete and R-cadence socket reselects both run ({calls})")
    _, calls = run("A", True, prefill=False)
    check(calls == [], f"A on: advance_tokens stays silent ({calls})")

    # B. First-decode query_reselect.
    st, calls = run("B", False, prefill=False)
    check(len(calls) == 1 and calls[0]["probe"], f"B off: first decode reselects with the query ({calls})")
    st, calls = run("B", True, prefill=False)
    check(calls == [] and not st.query_reselected,
          f"B on: no socket reselect and the flag stays free for Core ({calls},{st.query_reselected})")

    # C. post_prefill_shrink on the final prefill step.
    _, calls = run("C", False, prefill=True)
    check(len(calls) == 1 and calls[0]["probe"], f"C off: prompt-complete shrink runs ({calls})")
    _, calls = run("C", True, prefill=True)
    check(calls == [], f"C on: shrink defers to the boundary protocol ({calls})")

    # D. Decode-due reselect at the tail of after_step.
    _, calls = run("D", False, prefill=False)
    check(len(calls) == 1, f"D off: decode-due reselect runs ({calls})")
    _, calls = run("D", True, prefill=False)
    check(calls == [], f"D on: decode-due reselect is Core-owned ({calls})")

    print("sched_gate: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
