# A finished request must return its H share, otherwise a long-lived engine leaks the budget and
# eventually refuses new sessions.
import sys
import torch

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")
import vllm_kvmem_ext as ext
from kvmem_session_budget import SessionBudgetPool
from kvmem_sparse_runtime import VllmSparseRuntime

PAGE, BLOCK, PAGES = 16, 128, 13329
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


class View:
    """Minimal live-view stand-in: one paged layer, enough blocks for the pool."""

    def __init__(self):
        self.kv = torch.zeros(PAGES, 2, PAGE, 512, dtype=torch.uint8)
        self.row = tuple(range(101, 101 + 2048))

    def paged_kv(self):
        return [self.kv]

    def num_blocks(self):
        return PAGES

    def block_table_rows(self):
        return {}


def main():
    cfg = ext.SparseSelectorConfig()
    cfg.select_block_tokens, cfg.budget_tokens = BLOCK, 32768
    cfg.gen_budget_tokens, cfg.sink_tokens = 16384, 8192
    cfg.recent_tokens, cfg.page_tokens, cfg.lrfu = 1024, PAGE, False
    rt = VllmSparseRuntime(View(), ext.VllmSparseSelector(cfg), page_tokens=PAGE,
                           head_dim=256, rotary_dim=64, base=1e7,
                           select_block_tokens=BLOCK, budget_tokens=32768,
                           gen_reserve=16384, host_mib=4096, sessions=4,
                           use_kvmem_plan=False, use_kvmem_host=True)
    rt.budget_pool = SessionBudgetPool(host_bytes=4096 * 1024 * 1024, sessions=4)
    check(rt.budget_pool.unassigned_bytes() == 4096 * 1024 * 1024,
          "the pool starts fully unassigned")

    rt.state("req-a"); rt.orchestrator("req-a")
    after_open = rt.budget_pool.unassigned_bytes()
    check(after_open < 4096 * 1024 * 1024, "opening a request assigns its share")

    check(rt.live_requests() == ["req-a"], "the runtime reports its live requests")
    ok = rt.release("req-a")
    check(ok, "release reports that it dropped the request")
    check(rt.budget_pool.unassigned_bytes() == 4096 * 1024 * 1024,
          "the finished request returns its whole share")
    check(not rt.release("req-a"), "releasing twice is harmless")

    # Two concurrent sessions each keep their own share.
    rt.state("req-b"); rt.orchestrator("req-b")
    rt.state("req-c"); rt.orchestrator("req-c")
    check(rt.budget_pool.unassigned_bytes() <= 4096 * 1024 * 1024 - 2 * (4096 * 1024 * 1024 // 4),
          "two live sessions hold two shares")
    rt.release("req-b"); rt.release("req-c")
    check(rt.budget_pool.unassigned_bytes() == 4096 * 1024 * 1024,
          "closing both returns the whole budget")

    print("PASS session_lifecycle_test" if not failed else "FAILED session_lifecycle_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
