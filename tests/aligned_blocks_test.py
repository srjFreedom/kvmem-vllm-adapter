# The scheduler frees in aligned blocks (1,616 tokens = 101 pages); a block may only be handed
# back when every one of its pages is outside the working set.
import sys

sys.path.insert(0, "src")
from kvmem_aligned_blocks import block_of_page, block_pages, plan_aligned_blocks

PPB = 101          # pages per aligned block (1616 / 16)
PAGE = 16
FIRST = 101        # the NULL block occupies pages 0..100
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def main():
    # Working set covering aligned blocks 0 and 2 completely, block 1 not at all,
    # and block 3 only partially.
    selected = []
    selected += list(block_pages(0, FIRST, PPB))
    selected += list(block_pages(2, FIRST, PPB))
    selected += list(block_pages(3, FIRST, PPB))[:10]
    selected.sort()

    plan = plan_aligned_blocks(selected, FIRST, PPB, PAGE)
    check(plan.tokens_per_block == 1616, "an aligned block is 1,616 tokens")
    check(plan.kept == (0, 2, 3), f"kept blocks are 0,2,3 (got {plan.kept})")
    check(plan.partial == (3,), f"block 3 is only partially selected (got {plan.partial})")
    check(plan.freeable == (1,), f"only block 1 can be handed back (got {plan.freeable})")

    # A block whose pages are all selected must never be freeable.
    check(0 not in plan.freeable and 2 not in plan.freeable,
          "fully selected blocks are not freeable")

    # Page -> block mapping is consistent with the NULL-block offset.
    check(block_of_page(FIRST, FIRST, PPB) == 0, "the first usable page is block 0")
    check(block_of_page(FIRST + PPB, FIRST, PPB) == 1, "the next block starts one PPB later")
    check(list(block_pages(1, FIRST, PPB))[0] == FIRST + PPB, "block_pages starts at the right page")

    # An empty selection leaves everything freeable up to nothing selected.
    empty = plan_aligned_blocks([], FIRST, PPB, PAGE)
    check(empty.kept == () and empty.freeable == (), "no selection -> nothing to keep or free")

    # The real budget: B=32768 tokens == 20 aligned blocks (32768/1616 = 20.3)
    budget_blocks = -(-32768 // 1616)
    check(budget_blocks == 21, f"B/1616 rounds up to 21 aligned blocks (got {budget_blocks})")

    print("PASS aligned_blocks_test" if not failed else "FAILED aligned_blocks_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
