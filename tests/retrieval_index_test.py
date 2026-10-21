# Block ranking must follow CONTENT (pre-RoPE keys vs the current query), not recency - that is the
# whole difference between retrieval mode and the recency fallback.
import sys

import torch

sys.path.insert(0, "src")
from kvmem_retrieval_index import KvmemRetrievalIndex

failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def main():
    idx = KvmemRetrievalIndex(block_tokens=32, page_tokens=16, head_dim=8, layers=2)
    check(idx.block_of(0) == 0 and idx.block_of(1) == 0 and idx.block_of(2) == 1,
          "two pages make one 32-token block")

    torch.manual_seed(0)
    target_block = 5
    for page in range(0, 40):                     # 20 blocks of 2 pages
        block = idx.block_of(page)
        for layer in range(2):
            if block == target_block:
                k = torch.ones(16, 8) * 0.5       # the block that matches the query
            else:
                k = torch.randn(16, 8) * 0.5
            idx.add(page, layer, k)
    idx.set_query(0, torch.ones(4, 8) * 0.5)
    idx.set_query(1, torch.ones(2, 16, 8) * 0.5)  # (heads, tokens, head_dim) is accepted too

    scores = idx.scores()
    best = int(max(range(len(scores)), key=lambda i: scores[i]))
    check(best == target_block, f"the matching block wins ({best} vs {target_block})")
    check(scores[best] > 0.99, f"the match is near-perfect ({scores[best]:.3f})")

    # A later block must be able to beat an earlier one: ranking is content based, not positional.
    check(scores[target_block] > scores[0], "content, not position, decides the order")

    check(idx.bytes_used() > 0, "the index reports its footprint")
    check(idx.covered_blocks() == sorted(idx.covered_blocks()), "covered blocks are ordered")

    # No query -> a clear error rather than silently falling back inside the index.
    bare = KvmemRetrievalIndex(32, 16, 8, 2)
    bare.add(0, 0, torch.ones(16, 8))
    try:
        bare.scores()
        check(False, "scoring without a query must raise")
    except RuntimeError:
        check(True, "scoring without a query raises")

    print("PASS retrieval_index_test" if not failed else "FAILED retrieval_index_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
