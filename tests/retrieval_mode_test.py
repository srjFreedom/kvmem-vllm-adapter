# With a retrieval index in place, reselect must keep the CONTENT-relevant blocks rather than the
# newest ones. That is the observable difference between retrieval mode and the recency fallback.
import sys

import torch

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")
import vllm_kvmem_ext as ext
from kvmem_retrieval_index import KvmemRetrievalIndex
from kvmem_sparse_orchestrator import SparseOrchestrator
from kvmem_sparse_rope import _quantize

PAGE, HEADS, HD, ROT, CONTENT, PAGES = 16, 2, 256, 64, 512, 13329
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def canon(p):
    g = torch.Generator().manual_seed(9000 + p)
    k = torch.randn(HEADS, PAGE, HD, generator=g) * 0.5
    v = torch.randn(HEADS, PAGE, HD, generator=g) * 0.5
    return _quantize(torch.cat([k, v], dim=-1).to(torch.float32))


def build(retrieval):
    cfg = ext.SparseSelectorConfig()
    cfg.select_block_tokens, cfg.budget_tokens = 128, 1024
    cfg.gen_budget_tokens, cfg.sink_tokens = 512, 256
    cfg.recent_tokens, cfg.page_tokens, cfg.lrfu = 128, PAGE, False
    orch = SparseOrchestrator(ext.VllmSparseSelector(cfg),
                              [torch.zeros(PAGES, HEADS, PAGE, CONTENT, dtype=torch.uint8)],
                              PAGE, HD, ROT, 1e7, reselect_interval=4096)
    if retrieval:
        orch.retrieval_index = KvmemRetrievalIndex(128, PAGE, HD, 1)
        # Blocks 0..15 exist; make block 2 the content match for the query.
        for page in range(0, 16 * 8):
            k = (torch.ones(PAGE, HD) * 0.5 if page // 8 == 2
                 else torch.randn(PAGE, HD) * 0.5)
            orch.retrieval_index.add(page, 0, k)
        orch.retrieval_index.set_query(0, torch.ones(4, HD) * 0.5)
    else:
        orch.retrieval_index = None
    for p in range(16 * 8):
        orch.write_page(p, canon(p), p)
    orch.note_tokens(16 * 8 * PAGE)
    return orch


def main():
    recency = build(retrieval=False)
    r1 = recency.reselect()
    retrieval = build(retrieval=True)
    r2 = retrieval.reselect()

    check(r1["selected"] > 0 and r2["selected"] > 0, "both modes select something")
    check(set(retrieval.resident) != set(recency.resident),
          "retrieval keeps a different set than recency")
    target_page = 2 * 8                                  # first page of the matching block
    check(target_page in retrieval.resident, "the content-relevant block is kept")
    check(target_page not in recency.resident or len(recency.resident) == len(retrieval.resident),
          "recency does not specifically keep it")

    print("PASS retrieval_mode_test" if not failed else "FAILED retrieval_mode_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
