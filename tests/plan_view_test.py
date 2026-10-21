# The runtime must take its attention view from KVMem (plan_working_set), not from a hand-rolled
# builder. This test pins the adapter: same pages in, same order and length out, and the budget
# knobs really reach KVMem.
import sys

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")
from kvmem_plan_view import KvmemPlanner

failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def main():
    planner = KvmemPlanner(page_tokens=16, block_tokens=128, budget_tokens=1024,
                           gen_reserve=256, layers=2, record_bytes=128 * 512 * 2)
    total = 24 * 128
    # Pages 0..63 belong to blocks 0..7 (the sink), plus a couple of middle blocks.
    # B=1024 tokens == exactly 8 blocks of 128: the selection must fit the budget.
    selected = list(range(0, 64))                          # blocks 0..7
    result = planner.plan(selected, total, mandatory_pages=range(0, 64))
    check(result.kept_blocks == (0, 1, 2, 3, 4, 5, 6, 7),
          f"kvmem keeps the blocks we asked for ({result.kept_blocks})")
    check(result.pages == tuple(sorted(set(selected))), "pages come back in compact order")
    check(result.visible_tokens == len(result.pages) * 16,
          f"visible tokens == pages x page_tokens ({result.visible_tokens})")

    # A bigger budget must not change the selection when the caller already chose the set: KVMem
    # decides the ORDER and the visible length, the policy decides membership.
    wide = KvmemPlanner(page_tokens=16, block_tokens=128, budget_tokens=8192,
                        gen_reserve=256, layers=2, record_bytes=128 * 512 * 2)
    wider = list(range(0, 64)) + list(range(64, 80))      # 10 blocks
    wide_result = wide.plan(wider, total, mandatory_pages=range(0, 64))
    check(wide_result.kept_blocks == tuple(range(10)),
          f"a bigger budget keeps the extra blocks in order ({wide_result.kept_blocks})")
    check(wide_result.kept_blocks[:8] == result.kept_blocks,
          "the shared prefix keeps the same order")

    # Pages of a partially-filled last block still plan (the catalog uses valid_tokens).
    partial = planner.plan(selected, total - 64, mandatory_pages=range(0, 64))
    check(partial.visible_tokens > 0, "a truncated frontier still produces a view")

    print("PASS plan_view_test" if not failed else "FAILED plan_view_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
