"""Store semantics: every committed page must carry the bytes the ENGINE really wrote there.

The 164,918/169,998-byte cold runs proved the compacted store contains ZERO pages for every
logical page whose row slot was grown after the first compaction (store sha == sha256 of the
all-zero page), while the dense prefix committed correctly. Mechanism: both growth loops key
`_page_map` by `len(grown) + page_shift`, but `page_shift` is never assigned (always 0), so
after a compaction the RANK position is mistaken for a LOGICAL page and the stale dense page id
(a released physical page, still zero) is committed as the new page's bytes.

Math: a token at compact rank position c corresponds to logical position c + D where
D = st.tokens - st.seq_len at commit start; for whole pages `logical = rank + D//page` exactly.

The offline model replays the live branch order: engine table position -> physical page (the
engine appends beyond the installed packed prefix), then `after_step` growth + commit.
"""
import sys
import types

sys.path.insert(0, "src")
from kvmem_sparse_runtime import VllmSparseRuntime, _ReqState

PAGE = 32
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


class Store:
    def __init__(self):
        self.pages = {}

    def has(self, p):
        return p in self.pages


class FakeOrch:
    def __init__(self):
        self.store = Store()
        self.slot_of = {}
        self.store_delta = {}
        self.baked_delta = {}
        self.resident = set()
        self.selector = None
        self.commits = []          # (logical, slot)

    def tick(self):
        pass

    def note_tokens(self, n):
        pass

    def should_reselect(self):
        return False

    def commit_written_page(self, logical, slot, baked=0):
        self.commits.append((int(logical), int(slot)))
        self.store.pages[int(logical)] = int(slot)
        self.slot_of[int(logical)] = int(slot)


def make_rt():
    rt = VllmSparseRuntime.__new__(VllmSparseRuntime)
    st = _ReqState()
    orch = FakeOrch()
    rt._state = {"r": st}
    rt._orch = {"r": orch}
    rt._page_map = {"r": {}}
    rt._absorb_upto = {"r": 0}
    rt._pending_frontier = {}
    rt._engine_logical = {}
    rt._engine_computed = {}
    rt._pending_new = {}
    rt._retrieval = {}
    rt._prompt_len = {}
    rt._over_budget_logged = set()
    rt.page_tokens = PAGE
    rt.budget_tokens = 36864
    rt.gen_reserve = 16384
    rt.inrequest_compaction = False
    rt.compact_rebuild = False
    rt.sched_compact = False
    rt.install = False
    rt.log = None
    rt._reselect = lambda *a, **k: None
    # a pure-copy compaction: runtime owns (tokens-seq_len)//page
    def pure():
        return True
    rt._pure_copy = pure
    return rt, st, orch


def engine_table_setup(physical_map):
    """Fake view: one row tensor indexed by engine position, values = physical page ids."""
    width = 4096
    table = [[0] * width]

    class Vec:
        def __init__(self, lst):
            self.lst = lst

        def tolist(self):
            return list(self.lst)

        def __len__(self):
            return len(self.lst)

        def __getitem__(self, i):
            return self.lst[i]

    class Row:
        def __init__(self, tbl):
            self.tbl = tbl

        def __len__(self):
            return 1

        def __getitem__(self, ridx):
            return Vec(self.tbl[0])

    rows = [Row(table)]

    class V:
        def block_table_rows(self):
            return rows
    return V(), table[0]


def main():
    # ---------- A) dense prefix commits from the engine table (must already work) ----
    rt, st, orch = make_rt()
    st.tokens = 0
    st.seq_len = 0
    st.row_index = 0
    view, tbl = engine_table_setup(None)
    # engine allocated 128 pages [196..323] for chunk 0 (logical 0..127)
    for pos in range(128):
        tbl[pos] = 196 + pos
    # row non-empty: the live model syncs the row from the engine BEFORE the first commit
    st.row = tuple(tbl[:128])
    rt.view = view
    rt.after_step("r", 4096, prefill=True)
    committed = dict((lp, sp) for lp, sp in orch.commits)
    check(all(committed.get(lp) == 196 + lp for lp in range(128)),
          f"A: dense commits map logical->engine page (got {len(committed)})")

    # ---------- B) post-compaction growth must take the ENGINE's appended slots ------
    # Compaction packed the row: rank 0..3 = [196, 900, 901, 902] (sink + selected tail of
    # the view); logical 1..3 are no longer their rank, and the resident length is 4 pages.
    rt, st, orch = make_rt()
    st.tokens = 64 * PAGE              # logical frontier 256 tokens... use pages: 64 pages
    st.seq_len = 4 * PAGE              # visible 4 pages after compaction
    st.row_index = 0
    packed = [196, 900, 901, 902]
    st.row = tuple(packed)
    view, tbl = engine_table_setup(None)
    for pos, p in enumerate(packed):
        tbl[pos] = p
    # the engine appends for the next chunk at rank positions 4..5: fresh physical pages
    tbl[4] = 700
    tbl[5] = 701
    # page_map holds STALE dense ids from before the compaction: logical 4 -> 200 (released,
    # zero-filled). The absorb cursor already passed these positions (live: prefill scanned the
    # whole dense table before the compaction packed it), so nothing re-absorbs the truth.
    rt._page_map["r"] = {lp: 196 + lp for lp in range(64)}
    rt._absorb_upto["r"] = 64
    rt.view = view
    # commit the 2 pages the engine just wrote (64 tokens): compact ranks 4 and 5,
    # logical pages 64 and 65 (D = tokens-seq = 60 pages -> logical = rank + 60)
    rt.after_step("r", 64, prefill=True)
    committed = dict(orch.commits)
    want = {(4 + 60): 700, (5 + 60): 701}
    check(committed.get(64) == 700 and committed.get(65) == 701,
          f"B: post-compaction commit takes the engine appended slot, not the stale dense id "
          f"(got {committed.get(64)},{committed.get(65)} want 700,701)")
    check(st.row[4:6] == (700, 701), f"B: row grew with the engine slots ({st.row[4:6]})")

    # ---------- C) the row must not double-append the last slot past the table -------
    rt, st, orch = make_rt()
    st.tokens = 64 * PAGE
    st.seq_len = 4 * PAGE
    st.row = tuple(packed)
    st.row_index = 0
    view, tbl = engine_table_setup(None)
    for pos, p in enumerate(packed):
        tbl[pos] = p
    tbl[4] = 700          # only rank 4 is published; rank 5 is NOT (engine hasn't appended)
    rt._page_map["r"] = {}
    rt.view = view
    try:
        rt.after_step("r", 96, prefill=True)   # crosses ranks 4,5,6
        ok = False
        err = ""
    except RuntimeError as e:
        ok = "no slot" in str(e)
        err = str(e)[:60]
    check(ok, f"C: unpublished engine slot raises instead of clamping/duplicating ({err})")
    check(all(sp > 0 for _, sp in orch.commits) and len(orch.commits) == 0,
          f"C: nothing is committed from an unpublished slot (zero-bytes never enter store) "
          f"({orch.commits})")

    print("store_semantics_test: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
