# Two concurrent long requests must get DISJOINT device slots: without that, both would compact
# into slots 0..|sel|-1 and silently destroy each other's KV. This is the 4-concurrency property.
import sys

import torch

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")
import vllm_kvmem_ext as ext
from kvmem_sparse_rope import _quantize
from kvmem_sparse_runtime import RunnerView, VllmSparseRuntime

PAGE = 16
HEADS = 2
HEAD_DIM = 256
ROTARY = 64
CONTENT = 2 * HEAD_DIM
PAGES = 13329
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
        self._pos = torch.zeros(8192, dtype=torch.int64)

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


def engine_write(view, row, seq_len, counter):
    # The real engine writes at kv[block_table[pos // PAGE]] - the PAGE ID, not the row index.
    page_slot = int(row[seq_len // PAGE])
    g = torch.Generator().manual_seed(counter[0])
    counter[0] += 1
    k = torch.randn(HEADS, PAGE, HEAD_DIM, generator=g) * 0.5
    v = torch.randn(HEADS, PAGE, HEAD_DIM, generator=g) * 0.5
    view.kv[page_slot].copy_(_quantize(torch.cat([k, v], dim=-1).to(torch.float32)))


def main():
    torch.manual_seed(0)
    view = FakeView()
    cfg = ext.SparseSelectorConfig()
    cfg.select_block_tokens = 128
    cfg.budget_tokens = 32768
    cfg.gen_budget_tokens = 16384
    cfg.sink_tokens = 8192
    cfg.recent_tokens = 1024
    cfg.page_tokens = PAGE
    cfg.lrfu = False
    selector = ext.VllmSparseSelector(cfg)
    rt = VllmSparseRuntime(view, selector, PAGE, HEAD_DIM, ROTARY, 1e7, reselect_interval=8)
    # Compaction is opt-in for the vLLM backend; this test exercises it.
    rt.inrequest_compaction = True

    counter = [1]
    req_a, req_b = "req-a", "req-b"
    each_tokens = 64 * 1024                      # 4,096 pages each, both fit the pool

    # Request A prefills, then request B prefills (as the engine would schedule them).
    # The engine allocates a page per logical page and records it in the batch row.
    rows = view.block_table_rows()[0]
    rt.before_step(req_a, 1, row_index=0)
    rt.before_step(req_b, 1, row_index=1)
    for step in range(each_tokens // PAGE):
        rows[0, step] = 101 + step
        rt.after_step(req_a, PAGE, prefill=True)
    for step in range(each_tokens // PAGE):
        rows[1, step] = 101 + 4096 + step
        rt.after_step(req_b, PAGE, prefill=True)

    # Both reselect: each must shrink to the budget with its OWN slots.
    rt.after_step(req_a, 0)
    rt.after_step(req_b, 0)

    a_state, b_state = rt.state(req_a), rt.state(req_b)
    a_slots = set(a_state.row)
    b_slots = set(b_state.row)
    check(len(a_slots & b_slots) == 0,
          f"the two requests use disjoint device slots (shared={sorted(a_slots & b_slots)[:4]})")
    # Resident model: every request compacts INSIDE its own blocks, so each row is that request's
    # own pages (no fresh slots taken from the shared pool) and the two are disjoint by
    # construction.
    check(set(rt.state(req_a).row) <= set(range(101, 101 + each_tokens // PAGE)),
          "req-a compacts only inside its own pages")
    check(set(rt.state(req_b).row) <= set(range(101 + each_tokens // PAGE,
                                             101 + 2 * each_tokens // PAGE)),
          "req-b compacts only inside its own pages")

    # Decode both, interleaved, as a real batch would.
    for step in range(3 * rt.reselect_interval):
        for req, state in ((req_a, a_state), (req_b, b_state)):
            row = list(state.row)
            engine_write(view, row, state.seq_len, counter)
            rt.after_step(req, 1)

    # The slots must STILL be disjoint after several reselects.
    a_slots = set(rt.state(req_a).row)
    b_slots = set(rt.state(req_b).row)
    check(len(a_slots & b_slots) == 0, "slots stay disjoint after reselects")

    # Every resident page of both requests must still be exactly one rotation from its bytes.
    for req in (req_a, req_b):
        orch = rt.orchestrator(req)
        bad = [p for p in sorted(orch.resident) if orch.needle(p) is not True]
        check(not bad, f"{req}: needle exact on {len(orch.resident)} pages (bad={bad[:3]})")
        check(len(orch.resident) <= selector.budget_pages(), f"{req}: resident within budget")

    # Stored bytes must be per-request: request A never sees B's pages.
    check(rt.orchestrator(req_a).store is not rt.orchestrator(req_b).store,
          "each request keeps its own canonical page store")

    print("PASS sparse_concurrency_test" if not failed else "FAILED sparse_concurrency_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
