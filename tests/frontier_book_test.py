"""CPU contract for the executed-frontier book: no one-chunk lag, no logical seq_len.

Measured on the 169,998-token TP2 beyond-pool run: at the first pressure compaction the worker's
executed book was 145,088 while EngineCore's resident book was already 153,280 - exactly the
8,192-token chunk that had just completed. `after_step` derived its base from the PREVIOUS
schedule's logical count and then subtracted the chunk again, so the commit landed one chunk
behind. After compaction the same code overwrote the resident `seq_len` with V2's CPU
`num_computed_tokens_np`, which is the sampler's LOGICAL mirror.
"""
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, "src")
from kvmem_sparse_runtime import VllmSparseRuntime, _ReqState

PAGE = 32
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


class Orch:
    gen_reserve = 16384
    reselect_token_budget = 0
    _tokens_at_reselect = 0
    resident = set()
    planner = None
    retrieval_index = None

    def __init__(self):
        self.pages = set()
        self.committed = []
        self.store = NS(pages=self.pages, has=lambda p: p in self.pages)
        self.notes = []

    def commit_written_page(self, page, slot, baked=0):
        self.committed.append((int(page), int(slot), int(baked)))
        self.pages.add(int(page))

    def note_tokens(self, tokens):
        self.notes.append(int(tokens))

    def tick(self):
        pass

    def should_reselect(self):
        return False


def build(st):
    rt = VllmSparseRuntime.__new__(VllmSparseRuntime)
    orch = Orch()
    rt.page_tokens = PAGE
    rt._state = {"r": st}
    rt._orch = {"r": orch}
    rt._prev_n = {}
    rt._pending_new = {}
    rt._pending_frontier = {}
    rt._engine_logical = {}
    rt._engine_computed = {}
    rt._page_map = {"r": {}}
    rt._inflight = {}
    rt._prev_chunk = {}
    rt._prompt_len = {"r": 0}
    rt._deferred = 0
    rt.spec_reqs = set()
    rt.gen_reserve = 16384
    rt.prefill_high_watermark = 10 ** 9
    rt.pool_ceiling_tokens = 0
    rt.inrequest_compaction = True
    rt.compact_rebuild = False
    rt.reselect_at_prompt_end = False
    rt.checkpoint_replay = False
    rt.session_reuse = False
    rt.stats_core = False
    rt._retrieval = {}
    rt._adv_logged = set()
    rt._absorb_engine_row = lambda req: None
    rt.log = lambda record: None
    return rt, orch


def main():
    # --- 1. a completed 8,192-token chunk must land on the true executed frontier -------------
    st = _ReqState()
    st.tokens = st.seq_len = 145088
    rt, orch = build(st)
    rt._page_map["r"] = {p: 101 + p for p in range(145088 // PAGE, 153280 // PAGE)}
    rt._engine_logical["r"] = 145088        # previous schedule's served frontier = before chunk
    rt._engine_computed["r"] = 145088
    rt._prev_n["r"] = 8192                   # the completed chunk pending a commit
    rt.advance_tokens("r", 8192)            # entry of the next step: commit, then record
    check(st.tokens == 153280, f"executed book reaches the true frontier (got {st.tokens})")
    check(orch.committed and orch.committed[0][0] == 145088 // PAGE,
          f"the committed pages start at the chunk's first page (got {orch.committed[:1]})")
    check(orch.committed[-1][0] == 153280 // PAGE - 1,
          f"the committed pages end at the chunk's last page (got {orch.committed[-1:]})")
    check(all(baked == 0 for _p, _s, baked in orch.committed),
          "a still-dense prefix is committed with zero rotation")

    # --- 2. after compaction the logical mirror must not become the resident seq_len ----------
    st = _ReqState()
    st.tokens = 145088
    st.seq_len = 36864
    st.row = tuple(range(101, 101 + 1280))     # room for the resident prefix + a spare chunk
    rt, orch = build(st)
    rt._engine_logical["r"] = 145088
    rt._engine_computed["r"] = 153280          # V2 CPU mirror: logical, NOT resident
    rt._page_map["r"] = {1152: st.row[1152]}
    rt._prev_n["r"] = 1                        # one decoded token pending a commit
    rt.advance_tokens("r", 1)                  # entry of the next step: commit, then record
    check(st.seq_len == 36865,
          f"compacted seq_len advances by the decoded token only (got {st.seq_len})")
    check(st.tokens == 145089, f"served book advances by one token (got {st.tokens})")

    print("PASS frontier_book_test" if not failed else "FAILED frontier_book_test")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
