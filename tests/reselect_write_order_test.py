# A reselect both MOVES resident pages on the device and REBUILDS evicted ones from the host
# store. A rebuild destination can be the source of another page's move, so writing the rebuilt
# bytes first clobbers the bytes the move still has to read.
# Measured live (N=2, long prefill): sink pages 0 and 1 stopped matching their canonical bytes
# after a few reselects and the answer degenerated into noise.
import sys

import torch

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")
import vllm_kvmem_ext as ext
from kvmem_sparse_orchestrator import SparseOrchestrator
from kvmem_sparse_rope import _quantize

PAGE = 16
HEADS = 2
HEAD_DIM = 256
ROTARY = 64
CONTENT = 2 * HEAD_DIM
PAGES = 64
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def canonical(page):
    g = torch.Generator().manual_seed(7000 + page)
    k = torch.randn(HEADS, PAGE, HEAD_DIM, generator=g) * 0.5
    v = torch.randn(HEADS, PAGE, HEAD_DIM, generator=g) * 0.5
    return _quantize(torch.cat([k, v], dim=-1).to(torch.float32)).contiguous()


def main():
    device = "cpu"
    kv = [torch.zeros(PAGES, HEADS, PAGE, CONTENT, dtype=torch.uint8, device=device)
          for _ in range(2)]
    cfg = ext.SparseSelectorConfig()
    cfg.select_block_tokens = 128
    cfg.budget_tokens = 32768
    cfg.gen_budget_tokens = 16384
    cfg.sink_tokens = 8192
    cfg.recent_tokens = 1024
    cfg.page_tokens = PAGE
    cfg.lrfu = False
    selector = ext.VllmSparseSelector(cfg)
    orch = SparseOrchestrator(selector, kv, PAGE, HEAD_DIM, ROTARY, 1e7, reselect_interval=8)
    orch.inrequest_compaction = True
    orch.note_tokens(2 * PAGE)

    src, dst = 5, 7
    canons = {0: canonical(0), 1: canonical(1)}

    # Page 1 is committed at its own slot 1 (baked rotation 0), then moved to `src` by hand: it is
    # RESIDENT at `src` with no rotation, so the reselect can move it on the device (src -> dst).
    orch.write_page(1, canons[1], 1)
    for table in orch.kvs:
        table[src].copy_(table[1])
    orch.slot_of[1] = src
    check(orch.baked_delta[1] == 0, "the resident page carries no rotation")

    # Page 0 has canonical bytes but NO device copy: the reselect must rebuild it - and its
    # destination is `src`, the page-1 slot that still has to be moved away.
    for table in orch.kvs:
        table[0].copy_(canons[0])
    orch.store.put(0, [canons[0] for _ in orch.kvs])
    orch.store_delta[0] = 0
    orch.resident.discard(0)
    orch.slot_of.pop(0, None)

    result = orch.reselect(mandatory_pages=[0, 1], slots=[src, dst])
    check(result["selected"] == 2, f"both pages stay selected (got {result['selected']})")

    check(orch.needle(1) is True, "the MOVED page still matches its canonical bytes")
    check(orch.needle(0) is True, "the REBUILT page matches its canonical bytes")
    check(orch.slot_of[0] == src and orch.slot_of[1] == dst,
          f"the pages land in their slots (0->{orch.slot_of[0]}, 1->{orch.slot_of[1]})")

    print("PASS reselect_write_order_test" if not failed else "FAILED reselect_write_order_test")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
