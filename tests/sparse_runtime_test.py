# Offline test of the live wiring: a fake runner feeds a 200K-token conversation through
# before_step/after_step and every installed tensor (row, seq_len, positions) must match the
# sparse view, while the orchestrator keeps the needle exact and the resident set within budget.
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
    def __init__(self):
        # two paged attention layers: a logical page spans all of them
        self.kvs = [torch.zeros(PAGES, HEADS, PAGE, CONTENT, dtype=torch.uint8)
                    for _ in range(2)]
        self._rows = [torch.zeros(1, PAGES, dtype=torch.int32)]
        self._seq = torch.zeros(1, dtype=torch.int32)
        self._pos = torch.zeros(4096, dtype=torch.int64)

    def paged_kv(self):
        return self.kvs

    def block_table_rows(self):
        return self._rows

    def seq_lens(self):
        return self._seq

    def positions(self):
        return self._pos

    def num_blocks(self):
        return PAGES


def engine_allocate(view, logical_pages, base):
    """Model the engine's allocator: give each logical page a physical id and record it in the
    block table row (real engines start at 101 because the NULL block spans an aligned unit)."""
    row = view.block_table_rows()[0][0]
    for logical in logical_pages:
        if int(row[logical]) == 0:
            row[logical] = base + logical


def engine_write(view, row, seq_len, page_tokens, n_tokens, counter):
    """Model the engine: write deterministic bytes into the page the installed view reads.

    The real engine writes at `kv[block_table[pos // page_tokens]]` (the page ID), not at the row
    index - with a permuted row those differ, and writing at the index hides exactly the bug this
    test exists to catch.
    """
    for i in range(n_tokens):
        pos = seq_len + i
        page = row[pos // page_tokens]
        slot = int(page)
        g = torch.Generator().manual_seed(counter[0])
        k = torch.randn(HEADS, PAGE, HEAD_DIM, generator=g) * 0.5
        v = torch.randn(HEADS, PAGE, HEAD_DIM, generator=g) * 0.5
        for kv in view.kvs:
            kv[slot].copy_(_quantize(torch.cat([k, v], dim=-1).to(torch.float32)))
        counter[0] += 1


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
    logs = []
    rt = VllmSparseRuntime(view, selector, PAGE, HEAD_DIM, ROTARY, 1e7,
                           reselect_interval=8, log=logs.append)
    # This test exercises compaction, which is opt-in for the vLLM backend (see should_reselect).
    rt.inrequest_compaction = True

    counter = [1]
    # --- prefill: 200K tokens, 16 per step, no sparse view yet -------------------------
    prefilled = 200 * 1024
    for step in range(prefilled // PAGE):
        engine_allocate(view, [step], base=101)     # the engine owns allocation
        rt.after_step(0, PAGE, prefill=True)
    st = rt.state(0)
    check(st.tokens == prefilled, f"prefill tracked {st.tokens} tokens")
    # Prefill now spills under pressure (kvmem's prefill-pressure selection), otherwise a 200K
    # prefill would consume the whole pool and 4 concurrent long requests could never coexist.
    check(len(rt.orchestrator(0).resident) <= rt.prefill_high_watermark + 16,
          f"prefill stays under the watermark "
          f"(resident={len(rt.orchestrator(0).resident)}, hw={rt.prefill_high_watermark})")
    check(st.row != (), "a sparse view is installed once prefill exceeds the watermark")

    # --- first reselect -----------------------------------------------------------------
    st = rt.after_step(0, 0)
    selected = len(rt.orchestrator(0).resident)
    check(selected == 2048, f"working set shrinks to 2048 pages (got {selected})")
    own = set(range(101, 101 + prefilled // PAGE))
    check(set(st.row) == own,
          f"the row is exactly the request's own pages (n={len(st.row)}, budget=2048)")
    check(st.seq_len == min(prefilled, selected * PAGE),
          f"seq_len covers the written tokens (got {st.seq_len})")

    # --- decode: install the view, let the engine write, commit -------------------------
    for step in range(3 * rt.reselect_interval):
        installed = rt.before_step(0, 1)
        check(installed is not None, "before_step installs a view") if step == 0 else None
        row = list(st.row) or list(range(101, 101 + 1))
        engine_allocate(view, [(st.tokens) // PAGE], base=101)
        engine_write(view, row, st.seq_len, PAGE, 1, counter)
        rt.after_step(0, 1)

    st = rt.state(0)
    orch = rt.orchestrator(0)
    check(len(orch.resident) <= selector.budget_pages(), "resident stays within budget")
    bad = [p for p in sorted(orch.resident) if orch.needle(p) is not True]
    check(not bad, f"needle exact on every resident page (bad={bad[:4]})")
    check(len(orch.store) >= 2048, f"store holds the canonical history ({len(orch.store)} pages)")
    # seq_len is the COMPACTED length the kernel reads, not the true token count.
    check(32000 <= st.seq_len <= 33000,
          f"compacted seq_len stays near B (got {st.seq_len} for {st.tokens} tokens)")
    check(st.seq_len <= st.tokens, "the compacted view never exceeds the true history")

    # The installed tensors must agree with the runtime state (install, then compare).
    rt.before_step(0, 1)
    row_t = view.block_table_rows()[0][0]
    check(all(int(row_t[i]) == st.row[i] for i in range(len(st.row))), "row matches the tensor")
    # The installed seq_len covers the current step's query too.
    check(int(view.seq_lens()[0]) == st.seq_len + 1, "seq_len covers the query token")
    reselects = [r for r in logs if r["kind"] == "reselect"]
    check(len(reselects) >= 4, f"reselect happened on schedule ({len(reselects)} times)")

    # --- R cadence baseline: the first decode step anchors at the prompt ----------------
    rt2 = VllmSparseRuntime(view, selector, PAGE, HEAD_DIM, ROTARY, 1e7,
                            reselect_interval=10 ** 6, gen_reserve=16384, log=logs.append)
    rt2.inrequest_compaction = True
    for step in range(64):
        engine_allocate(view, [step], base=101)
        rt2.after_step(1, PAGE, prefill=True)
    rt2.after_step(1, 1)                      # the first decode step (builds the initial view)
    o2 = rt2.orchestrator(1)
    check(64 * PAGE <= o2._tokens_at_reselect <= 64 * PAGE + 1,
          f"baseline anchored at the prompt, not 0 (got {o2._tokens_at_reselect})")
    check(o2.reselect_token_budget == 0, "the adapter-invented timer is off under compaction")
    check(not o2.should_reselect(), "no reselect before R output tokens")

    print("PASS sparse_runtime_test" if not failed else "FAILED sparse_runtime_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
