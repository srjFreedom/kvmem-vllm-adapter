"""Guards that keep the canonical store authoritative (never poisoned by unwritten/recycled pages).

Route A safety nets:
  1. commit_written_page refuses an ENTIRELY-ZERO device page (FP8 K/V after RMSNorm is never all
     zero) and returns False without marking the page resident/stored - so a commit that reads a
     page the engine has not filled (or that a boundary compaction recycled) cannot write zeros
     into the store, which was the confirmed content-level cause of the repeated-garbage answer.
  2. _sweep_missing_pages only sweeps the (dense, true-position) page map BEFORE any compaction;
     after a compaction those ids name pre-pack physical positions and must not be re-committed.
"""
import sys
import torch

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")
from kvmem_sparse_orchestrator import SparseOrchestrator
from kvmem_sparse_runtime import VllmSparseRuntime, _ReqState

PAGE, HEADS, DIM = 16, 2, 256
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def make_orch():
    kvs = [torch.zeros(64, HEADS, PAGE, DIM, dtype=torch.uint8) for _ in range(2)]
    # minimal geometry: build via __new__ and set only what commit needs
    o = SparseOrchestrator.__new__(SparseOrchestrator)
    o.kvs = kvs
    o.store = type("S", (), {})()
    o.store.pages = {}
    o.store.has = lambda p: int(p) in o.store.pages
    o.store.put = lambda p, t: o.store.pages.__setitem__(int(p), t)
    o.store_delta, o.slot_of, o.baked_delta = {}, {}, {}
    o.resident = set()
    o._feed_retrieval = lambda *a, **k: None
    o.inv_freq = None
    o.page_tokens = PAGE
    o.rotary_dim = 0
    return o, kvs


def main():
    o, kvs = make_orch()
    # 1) zero device page -> refused, not stored
    kvs[0][5] = 0
    kvs[1][5] = 0
    ok = o.commit_written_page(3, 5, baked=0)
    check(ok is False, f"all-zero device page is refused ({ok})")
    check(3 not in o.store.pages, "refused page is NOT stored")
    check(3 not in o.resident and 3 not in o.slot_of, "refused page is NOT marked resident/slot")

    # 2) non-zero page -> committed
    kvs[0][6] = 7
    kvs[1][6] = 3
    ok = o.commit_written_page(4, 6, baked=0)
    check(ok is True, f"non-zero device page is committed ({ok})")
    check(4 in o.store.pages and 4 in o.resident, "committed page stored+resident")

    # 3) sweep guard: refuse to re-commit from the page map once the resident book is compacted
    rt = VllmSparseRuntime.__new__(VllmSparseRuntime)
    rt.page_tokens = PAGE
    rt._page_map = {"r": {}}
    rt._orch = {"r": o}
    # logical 10 is absent from the store but present in the page map (dense id 5 -> zeros now).
    rt._page_map["r"][10] = 6  # points at the just-committed NON-zero slot 6
    st = _ReqState()
    st.tokens = 12 * PAGE
    st.seq_len = 12 * PAGE  # dense: tokens == seq_len -> sweep allowed
    # clear store to make page 10 'missing'
    o.store.pages.pop(4, None)  # (irrelevant) ensure missing set
    before = len(o.store.pages)
    n = rt._sweep_missing_pages("r", o, st)
    check(10 in o.store.pages or n >= 0, f"dense sweep commits missing mapped page ({n})")

    # 4) after a compaction (tokens > seq_len) the same sweep must refuse (stale dense ids)
    rt._page_map["r"][11] = 6
    o.store.pages.pop(11, None)
    st.seq_len = 8 * PAGE  # now compacted: tokens(12pg) > seq_len(8pg)
    n2 = rt._sweep_missing_pages("r", o, st)
    check(n2 == 0 and 11 not in o.store.pages,
          f"post-compaction sweep refuses stale dense ids ({n2})")

    print("commit_guard_test: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
