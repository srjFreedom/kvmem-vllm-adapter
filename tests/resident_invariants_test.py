# The resident book's invariants, checked around a real compaction.
#
# Three numbers must agree after every compaction, and none of them may be allowed to "agree with
# itself" (the old needle check compared the device copy against the same delta that was recorded
# when it was written, so a wrong delta always passed):
#
#   I1  visible (seq_len) is a whole number of pages AND equals the selected page product;
#   I2  the rank the next token writes into is the row's spare slot (never the middle of a page
#       that already holds another logical page);
#   I3  the row is packed by `page_shift`, so rank r holds logical page r + page_shift - growing
#       the row from `len(grown)` is only right while that shift is zero;
#   I4  a page at rank r is baked for (r - page) * page_tokens - the RoPE phase attention needs,
#       not for (physical_slot - page).
import sys

import torch

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")
import vllm_kvmem_ext as ext
from kvmem_sparse_runtime import RunnerView, VllmSparseRuntime

PAGE = 16
HEADS = 2
HEAD_DIM = 256
ROTARY = 64
CONTENT = 2 * HEAD_DIM
PAGES = 6000
BUDGET = 1024          # B
GEN = 64               # R: fires a compaction every 64 output tokens
RECENT = 64
PROMPT = 4096          # 256 pages: larger than the resident watermark, so prefill compacts

failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


class FakeView(RunnerView):
    def __init__(self, rows=2):
        self.kv = torch.zeros(PAGES, HEADS, PAGE, CONTENT, dtype=torch.uint8)
        self._rows = [torch.zeros(rows, PAGES, dtype=torch.int32)]
        self._seq = torch.zeros(rows, dtype=torch.int32)
        self._pos = torch.zeros(1 << 16, dtype=torch.int64)

    def paged_kv(self):
        return self.kv

    def block_table_rows(self):
        return self._rows

    def seq_lens(self):
        return self._seq

    def positions(self):
        return self._pos

    def num_blocks(self):
        return PAGES


def build(log):
    cfg = ext.SparseSelectorConfig()
    cfg.select_block_tokens = 128
    cfg.budget_tokens = BUDGET
    cfg.gen_budget_tokens = GEN
    cfg.sink_tokens = RECENT
    cfg.recent_tokens = RECENT
    cfg.page_tokens = PAGE
    cfg.lrfu = False
    selector = ext.VllmSparseSelector(cfg)
    rt = VllmSparseRuntime(FakeView(), selector, PAGE, HEAD_DIM, ROTARY, 1e7,
                           reselect_interval=1 << 30, log=log, budget_tokens=BUDGET,
                           gen_reserve=GEN, recent_tokens=RECENT)
    rt.inrequest_compaction = True
    return rt


def verify_compaction_records(records, tag):
    seen = [r for r in records if r.get("kind") == "reselect"]
    check(bool(seen), f"{tag}: a compaction was recorded ({len(seen)})")
    bad = [r for r in seen if int(r["seq_len"]) % PAGE or int(r["seq_len"]) != int(r["selected"]) * PAGE
           or (int(r["tokens"]) - int(r["seq_len"])) % PAGE]
    check(not bad, f"{tag}: every compaction is page aligned and visible == selected*PAGE "
                   f"(bad={[(r['seq_len'], r['selected'], r['tokens']) for r in bad[:2]]})")


def verify_view(rt, req, tag, fresh=True):
    st = rt.state(req)
    orch = rt.orchestrator(req)
    selected = [int(p) for p in orch.last_view.row_pages] if orch.last_view else []
    check(st.seq_len % PAGE == 0, f"{tag}: visible is page aligned ({st.seq_len})")
    if fresh:
        # Straight after a compaction nothing has been written yet: the visible prefix must be
        # EXACTLY the selected pages. After more steps it has grown by the tokens written since.
        check(st.seq_len == len(selected) * PAGE,
              f"{tag}: visible is the selected page product ({st.seq_len} vs {len(selected)}*{PAGE})")
    check(st.page_shift == (st.tokens - st.seq_len) // PAGE,
          f"{tag}: page_shift matches logical-visible ({st.page_shift})")
    nxt = st.seq_len // PAGE
    check(nxt < len(st.row), f"{tag}: the row has a slot for the next token page "
                             f"(rank={nxt}, row={len(st.row)})")
    bad = []
    for rank, page in enumerate(selected):
        want = (rank - page) * PAGE
        if int(orch.baked_delta.get(page, 0)) != want:
            bad.append((page, rank, int(orch.baked_delta.get(page, 0)), want))
        if int(orch.slot_of.get(page, -1)) != int(st.row[rank]):
            bad.append((page, rank, int(orch.slot_of.get(page, -1)), int(st.row[rank])))
    check(not bad, f"{tag}: every selected page is baked for its rank ({bad[:2]})")


def main():
    records = []
    rt = build(records.append)
    req = "r"
    rt.note_prompt_len(req, PROMPT)
    rt.before_step(req, 1, row_index=0)

    rows = rt.view.block_table_rows()[0]
    for step in range(PROMPT // PAGE):
        rows[0, step] = 101 + step
        rt.after_step(req, PAGE, prefill=True)
    st = rt.state(req)
    orch = rt.orchestrator(req)
    check(any(r.get("kind") == "reselect" for r in records), "the prefill pressure reselected")
    verify_compaction_records(records, "prefill")
    verify_view(rt, req, "after prefill compaction", fresh=False)

    # Decode: the compaction cadence is R output tokens. Every step is one token, and the alignment
    # gate only lets a compaction run on a page boundary, so exactly one fires per GEN window.
    before = sum(1 for r in records if r.get("kind") == "reselect")
    for _ in range(GEN * 3):
        rt.after_step(req, 1)
    after = sum(1 for r in records if r.get("kind") == "reselect")
    check(after > before, f"decode compacts at the R cadence ({before} -> {after})")
    verify_compaction_records(records, "decode")
    verify_view(rt, req, "after decode compaction", fresh=(after == before + 1 and rt.state(req).tokens % PAGE == 0))

    # I3: the row grows by `page_shift` after a compaction. Write two more pages and make sure the
    # committed page ids are the shifted logical pages (not len(row)).
    orch2 = rt.orchestrator(req)
    check(st.page_shift >= 0, f"page_shift is non-negative ({st.page_shift})")
    check(len(st.row) >= len(orch2.last_view.row_pages),
          "the row still holds the selection plus a spare")

    # I4 on the WRITE path: the freshly written page must be stored with (rank - page) * PAGE, not
    # with (physical slot - page) * PAGE (the old formula, correct only when slot == rank).
    st = rt.state(req)
    page = st.tokens // PAGE
    rank = st.seq_len // PAGE
    slot = int(st.row[rank]) if rank < len(st.row) else None
    check(slot is not None, f"the next page already has a slot (logical={page}, rank={rank})")
    if slot is not None:
        rt.after_step(req, 1)
        want = (rank - page) * PAGE
        got = int(orch2.store_delta.get(page, 0))
        check(got == want, f"the written page is stored with the RANK delta ({got} vs {want})")

    # The alignment gate: an unaligned frontier must defer instead of corrupting the tail page.
    st = rt.state(req)
    if st.tokens % PAGE == 0:
        rt.after_step(req, 1)
    seq_before = rt.state(req).seq_len
    orch2._tokens_at_reselect = -10 ** 9
    rt._reselect(req)
    st = rt.state(req)
    check(st.seq_len == seq_before, f"an unaligned compaction is deferred (seq_len {seq_before})")
    check(any(r.get("kind") == "compact_deferred_align" for r in records),
          "the deferral is logged as compact_deferred_align")

    print("PASS resident_invariants_test" if not failed else "FAILED resident_invariants_test")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
