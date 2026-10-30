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


def test_neighbor_propagation():
    from kvmem_recall import propagate_neighbor_scores
    scores = [0.0, 0.0, 1.0, 0.0, 0.0, float("-inf")]
    out = propagate_neighbor_scores(scores, 0.2)
    check(abs(out[2] - 1.0) < 1e-9, "the peak keeps its own score (neighbours are 0)")
    check(abs(out[1] - 0.1) < 1e-9 and abs(out[3] - 0.1) < 1e-9,
          f"both neighbours inherit 0.2 x the MEAN of their two neighbours "
          f"({out[1]:.3f}, {out[3]:.3f})")
    check(out[5] == float("-inf"), "a non-finite entry never contributes or changes")
    check(propagate_neighbor_scores(scores, 0.0) == scores, "alpha 0 is a no-op")

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

    # A session checkpoint retains only sealed-prefix key sums; the next turn starts with a fresh Q.
    checkpoint = idx.checkpoint_prefix(10 * 32)
    check(checkpoint.covered_blocks() == list(range(10)),
          "checkpoint copies only blocks before its executed frontier")
    check(checkpoint._query is None, "checkpoint does not inherit the prior turn's query")
    before = checkpoint._sums[0].clone()
    idx.add(20, 0, torch.ones(16, 8) * 9)
    check(torch.equal(checkpoint._sums[0], before),
          "later suffix writes cannot mutate the sealed checkpoint statistics")
    try:
        idx.checkpoint_prefix(10 * 32 + 1)
        check(False, "unaligned retrieval checkpoint frontier must reject")
    except ValueError:
        check(True, "unaligned retrieval checkpoint frontier rejects")

    # No query -> a clear error rather than silently falling back inside the index.
    bare = KvmemRetrievalIndex(32, 16, 8, 2)
    bare.add(0, 0, torch.ones(16, 8))
    try:
        bare.scores()
        check(False, "scoring without a query must raise")
    except RuntimeError:
        check(True, "scoring without a query raises")

    # The capture runs inside a model step (InferenceMode), so _query is an inference tensor.
    # begin_query/add_query used to raise "Inplace update to inference tensor outside InferenceMode"
    # on every layer, leaving the query at zero and making compaction rank with no query.
    live = KvmemRetrievalIndex(32, 16, 8, 2)
    with torch.inference_mode():
        live._query = torch.zeros(2, 8)
        live._query_counts = torch.zeros(2)
    live.begin_query(0)
    live.add_query(0, torch.ones(8))
    live.add_query(0, torch.ones(8))
    check(bool(torch.allclose(live._query[0], torch.full((8,), 2.0))),
          f"query accumulates over inference tensors (got {live._query[0].tolist()[:2]})")
    check(float(live._query_counts[0]) == 2.0, "query count tracks the accumulation")

    test_neighbor_propagation()

    print("PASS retrieval_index_test" if not failed else "FAILED retrieval_index_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
