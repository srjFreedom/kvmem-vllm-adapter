# Resident model: a prompt LONGER than B is prefilled in chunks while the resident book stays
# bounded (the logical book keeps every token). Before this, such a prompt stalled the engine and
# forced a full re-prefill (measured live: 143,216-token prompt, 12 refused compactions, sink pages
# corrupted). The runtime must now compact DURING prefill and report the chunked shape once.
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
PAGES = 13329
BUDGET = 32768
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


class FakeView(RunnerView):
    def __init__(self, rows=4):
        self.kv = torch.zeros(PAGES, HEADS, PAGE, CONTENT, dtype=torch.uint8)
        self._rows = [torch.zeros(rows, PAGES, dtype=torch.int32)]
        self._seq = torch.zeros(rows, dtype=torch.int32)
        self._pos = torch.zeros(65536, dtype=torch.int64)

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
    cfg.gen_budget_tokens = 16384
    cfg.sink_tokens = 8192
    cfg.recent_tokens = 1024
    cfg.page_tokens = PAGE
    cfg.lrfu = False
    selector = ext.VllmSparseSelector(cfg)
    rt = VllmSparseRuntime(FakeView(), selector, PAGE, HEAD_DIM, ROTARY, 1e7,
                           reselect_interval=8, log=log, budget_tokens=BUDGET,
                           gen_reserve=16384, recent_tokens=1024)
    rt.inrequest_compaction = True
    return rt


def prefill(rt, req, row, tokens):
    rows = rt.view.block_table_rows()[0]
    for step in range(tokens // PAGE):
        rows[row, step] = 101 + step
        rt.after_step(req, PAGE, prefill=True)


def main():
    # A 64K prompt (2x B) must be compacted WHILE it is prefilled, with the resident book kept
    # inside the watermark and no page taken from outside the request's own blocks.
    records = []
    rt = build(records.append)
    rt.note_prompt_len("long", 64 * 1024)
    rt.before_step("long", 1, row_index=0)
    prefill(rt, "long", 0, 64 * 1024)
    st = rt.state("long")
    orch = rt.orchestrator("long")
    check(bool(st.row), "a long prefill installs a view")
    check(any(r.get("kind") == "reselect" for r in records), "prefill pressure reselects")
    check(len(orch.resident) <= rt.prefill_high_watermark + 16,
          f"the resident book stays bounded (resident={len(orch.resident)})")
    own = set(range(101, 101 + 64 * 1024 // PAGE))
    check(set(st.row) <= own, "the row uses only the request's own pages")
    check(st.tokens == 64 * 1024, f"the logical book keeps every token (got {st.tokens})")
    logged = [r for r in records if r.get("kind") == "long_prompt_chunked"]
    check(len(logged) == 1 and logged[0].get("prompt_tokens") == 64 * 1024,
          f"the chunked shape is logged once ({len(logged)})")

    print("PASS prefill_compaction_test" if not failed else "FAILED prefill_compaction_test")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
