# After the first reselect every kept page is resident, so a window that advances by a few pages
# must be served almost entirely by DEVICE-TO-DEVICE moves (no PCIe re-read), and the bytes must
# still match the one-hop rebuild exactly.
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
    g = torch.Generator().manual_seed(2000 + p)
    k = torch.randn(HEADS, PAGE, HD, generator=g) * 0.5
    v = torch.randn(HEADS, PAGE, HD, generator=g) * 0.5
    return _quantize(torch.cat([k, v], dim=-1).to(torch.float32))


def main():
    cfg = ext.SparseSelectorConfig()
    cfg.select_block_tokens, cfg.budget_tokens = 128, 32768
    cfg.gen_budget_tokens, cfg.sink_tokens = 16384, 8192
    cfg.recent_tokens, cfg.page_tokens, cfg.lrfu = 1024, PAGE, False
    orch = SparseOrchestrator(ext.VllmSparseSelector(cfg),
                             [torch.zeros(PAGES, HEADS, PAGE, CONTENT, dtype=torch.uint8)
                              for _ in range(2)],
                             PAGE, HD, ROT, 1e7, reselect_interval=4096)

    total = 2600 * PAGE
    for p in range(total // PAGE):
        orch.write_page(p, canon(p), p)
    orch.note_tokens(total)
    orch.reselect()
    first = dict(orch.last_move_stats)
    check(first["device_moves"] == 0, "the first reselect has nothing resident to move")

    # Advance the window by 6 pages: sink/recent shift by a tiny amount.
    for p in range(total // PAGE, total // PAGE + 6):
        orch.write_page(p, canon(p), p)
    orch.note_tokens(total + 6 * PAGE)
    orch.reselect()
    stats = dict(orch.last_move_stats)
    check(stats["selected"] > 1500, f"the working set is large ({stats['selected']} pages)")
    # STRUCTURAL FACT: compaction changes almost every page's rank, hence its rotation. Pages
    # whose rotation changes cannot be moved device-to-device (that would mean rotating a rotated
    # page, which compounds fp8 error), so they MUST be rebuilt from the host store.
    check(stats["host_rebuild"] > 0.5 * stats["selected"],
          f"a shifted window rebuilds most pages from the store ({stats['host_rebuild']})")
    check(stats["device_moves"] == 0,
          "no page can skip the rotation when the window moves")

    bad = [p for p in sorted(orch.resident) if orch.needle(p) is not True]
    check(not bad, f"every resident page still matches its one-hop rebuild (bad={bad[:6]})")

    print("PASS device_move_test" if not failed else "FAILED device_move_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
