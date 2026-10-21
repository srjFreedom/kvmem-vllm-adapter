# Offline end-to-end of the sparse orchestrator on a simulated 200K conversation:
#   - the engine writes pages at COMPACTED slots (positions overridden), which we commit,
#   - a reselect evicts the unselected pages and rebuilds the selected ones from canonical,
#   - every device page must equal exactly ONE rotation of its canonical bytes (the needle),
#   - canonical bytes must never be lost or double-rotated.
import sys

import torch

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")
import vllm_kvmem_ext as ext
from kvmem_sparse_orchestrator import SparseOrchestrator
from kvmem_sparse_rope import _dequant, _quantize

PAGE = 16
HEADS = 2
HEAD_DIM = 256
ROTARY = 64
CONTENT = 2 * HEAD_DIM
BASE = 1e7
PAGES = 13329         # the engine's real pool (213,274 tokens / 16)
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def make_canonical(page, device):
    g = torch.Generator(device="cpu").manual_seed(1000 + page)
    k = torch.randn(HEADS, PAGE, HEAD_DIM, generator=g) * 0.5
    v = torch.randn(HEADS, PAGE, HEAD_DIM, generator=g) * 0.5
    packed = torch.cat([k, v], dim=-1)
    return _quantize(packed.to(torch.float32)).contiguous().to(device)


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
    orch = SparseOrchestrator(selector, kv, PAGE, HEAD_DIM, ROTARY, BASE,
                              reselect_interval=8)
    orch.inrequest_compaction = True

    total_pages = 200 * 1024 // PAGE           # a 200K conversation
    canons = {p: make_canonical(p, device) for p in range(total_pages)}

    # A long prefill: the engine writes every page at its own slot (delta 0, no compaction yet).
    for page in range(total_pages):
        orch.write_page(page, canons[page], page)
    orch.note_tokens(total_pages * PAGE)
    check(len(orch.store) == total_pages, f"canonical pool holds all {total_pages} pages")

    # First sparse selection: a 200K conversation shrinks to the B budget.
    first = orch.reselect()
    check(first["selected"] == 2048, f"working set shrinks to B/16 pages (got {first['selected']})")
    check(first["evicted"] == total_pages - 2048,
          f"the rest is evicted (evicted={first['evicted']})")

    # Decode: new tokens keep arriving; the working set follows them.
    next_page = total_pages
    used_slots = set(orch.slot_of.values())
    for step in range(3 * orch.reselect_interval):
        canons[next_page] = make_canonical(next_page, device)
        slot = next(i for i in range(PAGES) if i not in used_slots)
        used_slots.add(slot)
        orch.write_page(next_page, canons[next_page], slot)
        next_page += 1
        orch.note_tokens(next_page * PAGE)
        orch.tick()
        if orch.should_reselect():
            stats = orch.reselect()
            check(stats["resident"] == stats["selected"],
                  f"decode step {step}: resident tracks the selection")

    # Canonical bytes are intact: the orchestrator never mutated the store's copies.
    # A stored page holds one tensor per paged layer; the canonical single-layer pattern is
    # broadcast to every layer by write_page, so compare against it layer by layer.
    same = all(torch.equal(orch.store.get(p)[0], canons[p])
               for p in sorted(orch.store._pages))
    check(same, "canonical bytes are byte-identical to what the engine wrote")

    # The needle: every resident device page is exactly one rotation of canonical.
    bad = [p for p in sorted(orch.resident) if orch.needle(p) is not True]
    check(not bad, f"every resident page is one rotation of canonical (bad={bad[:4]})")

    # Resident set never exceeds the budget.
    check(len(orch.resident) <= selector.budget_pages(),
          f"resident {len(orch.resident)} <= budget {selector.budget_pages()}")

    # The sink prefix survives every reselect.
    check(0 in orch.resident and 1 in orch.resident, "the sink prefix stays resident")

    # A no-op reselect must not change device bytes (idempotent, no extra quantization hop).
    before = [k.clone() for k in kv]
    orch.reselect()
    check(all(torch.equal(b, k) for b, k in zip(before, kv)), "a repeated reselect writes identical bytes (no extra hop)")

    # --- R cadence: baseline anchored at the prompt, R counts OUTPUT tokens -------------
    orc = SparseOrchestrator(selector, kv, PAGE, HEAD_DIM, ROTARY, BASE, reselect_interval=10 ** 6)
    orc.inrequest_compaction = True
    orc.gen_reserve = 1024          # R: the generation reserve
    orc.reselect_token_budget = 0   # no adapter-invented timer in compaction mode
    prompt = 30000
    orc._tokens_at_reselect = prompt          # what after_step sets at the first decode step
    orc.note_tokens(prompt + 1023)
    check(not orc.should_reselect(), "no reselect just before R new output tokens")
    orc.note_tokens(prompt + 1024)
    check(orc.should_reselect(), "reselect fires exactly at R new output tokens (pool full)")
    orc.inrequest_compaction = False
    check(not orc.should_reselect(), "compaction off never reselects")

    print("PASS sparse_orchestrator_test" if not failed else "FAILED sparse_orchestrator_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
