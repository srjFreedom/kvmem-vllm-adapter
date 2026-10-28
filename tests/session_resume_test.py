"""Offline check of the cross-request RESUME rotation frame (no engine, no ext build).

The invariant under test: `store_delta[p]` is the rotation the stored bytes carry **relative to page
p's logical start** -- not the engine page-id difference `(slot - p)`. The two only coincide while a
request's page ids run parallel to its compacted ranks; a resume writes the same logical page at a
different base, and an id-derived delta then bakes a constant `(base_new - base_old)` rotation into
the whole restored context (silent degeneration, no error).

What this pins down:
  1. A session prefilled at TRUE positions stores `store_delta == 0` regardless of which engine page
     ids it happened to get, and its bytes are the canonical page.
  2. `restore_page_rank(p, dst, rank)` lands in tensor page `dst`, carries `(rank - p) * page_tokens`
     and `needle()` confirms it byte-for-byte (one hop from the store, no extra rounding).
  3. The id-derived convention would carry the engine-base offset instead -- exactly the bug.

Run: python tests/session_resume_test.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kvmem_sparse_orchestrator import SparseOrchestrator  # noqa: E402
from kvmem_sparse_rope import _quantize  # noqa: E402

PAGE, HEADS, HD, ROT, CONTENT, PAGES = 16, 2, 256, 64, 512, 8192
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


class StubSelector:
    """The selector surface the rotation paths need (they score nothing)."""

    def set_context_tokens(self, tokens):
        pass

    def set_block_scores(self, scores):
        pass

    def budget_pages(self):
        return 512

    def select_pages(self, mandatory=()):
        return sorted(int(p) for p in mandatory)

    def select_prefill_pressure_pages(self, mandatory=()):
        return self.select_pages(mandatory)


def canon(page):
    g = torch.Generator().manual_seed(9000 + int(page))
    k = torch.randn(HEADS, PAGE, HD, generator=g) * 0.5
    v = torch.randn(HEADS, PAGE, HD, generator=g) * 0.5
    return _quantize(torch.cat([k, v], dim=-1).to(torch.float32))


def build():
    kv = [torch.zeros(PAGES, HEADS, PAGE, CONTENT, dtype=torch.uint8) for _ in range(2)]
    return SparseOrchestrator(StubSelector(), kv, PAGE, HD, ROT, 1e7, reselect_interval=4096)


def main():
    written = list(range(0, 40))
    base_old = 101          # the parked request's engine pages started here
    base_new = 707          # the resuming request's engine pages start somewhere else

    # --- 1. park a session that was prefilled at TRUE positions ------------------------------
    orch = build()
    for p in written:
        # Bytes as the engine wrote them (true positions), stored under a shifted engine page id.
        orch.write_page(p, canon(p), p + base_old, baked=0)
    check(all(orch.store_delta[p] == 0 for p in written),
          "a true-position prefill records store_delta 0 whatever the engine page ids are")
    check(all(torch.equal(orch.store.get(p)[0], canon(p)) for p in (0, 17, 39)),
          "the stored bytes are the canonical page (no offset baked in)")

    # --- 2. resume into a different base and a different rank order --------------------------
    working = list(range(0, 4)) + list(range(36, 40))     # sink + recent
    for rank, p in enumerate(working):
        dst = base_new + rank
        needed = orch.restore_page_rank(p, dst, rank)
        check(needed == (rank - p) * PAGE,
              f"page {p} at rank {rank} carries (rank-page)*page_tokens = {needed}")
        check(orch.slot_of[p] == dst, f"page {p} lands in tensor page {dst}")
        check(orch.baked_delta[p] == (rank - p) * PAGE, f"page {p} records the rank rotation")
        check(orch.needle(p) is True, f"page {p} restores byte-exactly at rank {rank}")

    # --- 3. the id-derived convention would bake the engine-base offset ----------------------
    naive = build()
    for p in written:
        naive.write_page(p, canon(p), p + base_old)      # no `baked`: legacy id fallback
    check(all(naive.store_delta[p] == base_old * PAGE for p in written),
          "the legacy fallback records the engine id difference as if it were a rotation")
    p, rank = working[3], 3
    naive_needed = (rank - p) * PAGE
    check(naive_needed == orch.baked_delta[p],
          "the semantic frame is what the compaction view actually expects")
    legacy_step = (base_new + rank - p) * PAGE - naive.store_delta[p]
    check(legacy_step - naive_needed == (base_new - base_old) * PAGE,
          "the id-based restore is off by exactly the engine-base difference")

    print("session_resume_test: OK" if not failed else "session_resume_test: FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
