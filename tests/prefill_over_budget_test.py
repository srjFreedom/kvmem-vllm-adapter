# A prompt ABOVE B cannot be compacted: vLLM requires num_tokens >= prompt_len, so the engine
# refuses every compaction while the worker would keep taking slots and moving pages. Measured
# live (N=2, one 143K-token prefill): 12 refused rounds, sink pages corrupted, gibberish output.
# The runtime must therefore refuse to touch such a request at all - and say so once.
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
    # Control: the engine does not report a prompt length -> prefill pressure keeps working.
    records = []
    rt = build(records.append)
    prefill(rt, "free", 0, 64 * 1024)
    check(bool(rt.state("free").row),
          "a long prefill is compacted when the prompt length is unknown")
    check(any(r.get("kind") == "reselect" for r in records), "the control request reselects")

    # Gated: the engine reports a prompt above B -> nothing is moved, nothing is installed.
    records = []
    rt = build(records.append)
    rt.note_prompt_len("over", 143216)
    prefill(rt, "over", 1, 64 * 1024)
    check(not rt.state("over").row, "an over-budget prompt installs no view")
    check(not rt._owned_slots.get("over"), "an over-budget prompt takes no device slots")
    check(not any(r.get("kind") == "reselect" for r in records), "no reselect is attempted")
    over = [r for r in records if r.get("kind") == "prompt_over_budget"]
    check(len(over) == 1 and over[0].get("prompt_tokens") == 143216,
          f"the refusal is logged exactly once ({len(over)})")

    print("PASS prefill_over_budget_test" if not failed else "FAILED prefill_over_budget_test")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
