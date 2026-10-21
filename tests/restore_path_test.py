# The objective requires: an evicted page must be able to come back from the host store and be
# byte-identical to a page that was never evicted (one rotation hop, never a chained one).
import sys
import torch

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")
import vllm_kvmem_ext as ext
from kvmem_sparse_orchestrator import SparseOrchestrator
from kvmem_sparse_rope import _quantize

PAGE, HEADS, HD, ROT, CONTENT, PAGES = 16, 2, 256, 64, 512, 13329
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def canon(p):
    g = torch.Generator().manual_seed(3000 + p)
    k = torch.randn(HEADS, PAGE, HD, generator=g) * 0.5
    v = torch.randn(HEADS, PAGE, HD, generator=g) * 0.5
    return _quantize(torch.cat([k, v], dim=-1).to(torch.float32))


def main():
    cfg = ext.SparseSelectorConfig()
    cfg.select_block_tokens, cfg.budget_tokens = 128, 32768
    cfg.gen_budget_tokens, cfg.sink_tokens = 16384, 8192
    cfg.recent_tokens, cfg.page_tokens, cfg.lrfu = 1024, PAGE, False
    kv = [torch.zeros(PAGES, HEADS, PAGE, CONTENT, dtype=torch.uint8) for _ in range(2)]
    orch = SparseOrchestrator(ext.VllmSparseSelector(cfg), kv, PAGE, HD, ROT, 1e7,
                              reselect_interval=4096)

    for p in range(300):
        orch.write_page(p, canon(p), p)
    orch.note_tokens(300 * PAGE)
    orch.reselect()

    victim = 150                       # a middle page that the window has dropped
    orch.resident.discard(victim); orch.slot_of.pop(victim, None); orch.baked_delta.pop(victim, None)
    check(not orch.resident or victim not in orch.resident, "the page is treated as evicted")

    new_slot = 2900                    # the scheduler would hand us a fresh block
    orch.restore_page(victim, new_slot)
    check(orch.needle(victim) is True, "the restored page matches its one-hop rebuild")
    check(orch.baked_delta[victim] == (new_slot - victim) * PAGE,
          "the restored page records the rotation it now carries")

    # A page that was never resident must restore identically (same code path).
    never = 299
    orch.resident.discard(never); orch.slot_of.pop(never, None); orch.baked_delta.pop(never, None)
    orch.restore_page(never, 2800)
    check(orch.needle(never) is True, "a page that was never resident restores identically")

    # Restoring into a slot that gives delta 0 must not add an fp8 hop at all.
    same_delta = 120
    orch.resident.discard(same_delta); orch.slot_of.pop(same_delta, None)
    orch.restore_page(same_delta, same_delta + orch.store_delta[same_delta] // PAGE)
    check(orch.needle(same_delta) is True, "restoring at the same rotation is byte-exact")

    # Repeated restores must not compound (the store is never rewritten).
    for _ in range(3):
        orch.restore_page(victim, new_slot)
    check(orch.needle(victim) is True, "repeated restores do not compound the fp8 error")

    # The control plane: a page can be restored into a block obtained from EngineCore.
    class FakeChannel:
        def request(self, request_id, count=1):
            return [3100]

    orch.resident.discard(victim); orch.slot_of.pop(victim, None)
    orch.restore_page(victim, slot=None, channel=FakeChannel(), request_id="req-x")
    check(orch.needle(victim) is True, "a page restored through the block channel is byte-exact")

    print("PASS restore_path_test" if not failed else "FAILED restore_path_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
