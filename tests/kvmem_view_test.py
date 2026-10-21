# Convergence step 1: take the attention view from KVMem itself (plan_working_set) instead of
# building it by hand, and prove the two agree. Any divergence would mean the hand-rolled view
# (which has been validated against the live engine) and KVMem's planner disagree about the
# compacted layout, i.e. the convergence is not a no-op.
import sys
import torch

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")
import vllm_kvmem_ext as ext
from kvmem_sparse_view import build_sparse_view

PAGE, BLOCK_TOKENS, BLOCKS, TOTAL = 16, 128, 24, 24 * 128
PAGES_PER_BLOCK = BLOCK_TOKENS // PAGE
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def main():
    cfg = ext.SparseSelectorConfig()
    cfg.select_block_tokens, cfg.budget_tokens = BLOCK_TOKENS, 1024
    cfg.gen_budget_tokens, cfg.sink_tokens = 512, 256
    cfg.recent_tokens, cfg.page_tokens, cfg.lrfu = 128, PAGE, False
    sel = ext.VllmSparseSelector(cfg)
    sel.set_context_tokens(TOTAL)
    selected_blocks = list(sel.select_pages([]))       # kvmem's pick_topk_blocks, 16-token pages
    check(bool(selected_blocks), f"kvmem selected {len(selected_blocks)} pages")

    # The hand-rolled view (validated on the live engine).
    view = build_sparse_view(selected_blocks, [], PAGE, 0)
    check(view.seq_len == len(selected_blocks) * PAGE,
          f"hand-rolled view covers {view.seq_len} tokens")

    # The same selection expressed as KVMem blocks, fed to KVMem's own planner.
    desc = ext.MemoryDescriptor()
    desc.identity.backend = "vllm"
    desc.identity.revision = "sm75-sparse-1"
    desc.identity.model = "test-model"
    desc.identity.state_schema = "kv1"
    desc.layout.format = "fp8-kv"
    desc.layout.version = 1
    desc.layout.block_tokens = BLOCK_TOKENS
    desc.layout.record_bytes = BLOCK_TOKENS * 512    # one 512-byte row per token
    desc.layout.alignment = 16
    half = desc.layout.record_bytes // 2
    planes = []
    for kind, offset in ((ext.PayloadPlaneKind.Key, 0), (ext.PayloadPlaneKind.Value, half)):
        plane = ext.PayloadPlane()
        plane.layer = 0
        plane.kind = kind
        plane.offset = offset
        plane.bytes = half
        plane.alignment = 16
        plane.token_stride = 256
        planes.append(plane)
    desc.layout.planes = planes
    desc.capabilities.page_tokens = PAGE
    desc.capabilities.max_position = TOTAL
    desc.capabilities.native_payload = True
    desc.capabilities.sparse_view = True          # we DO hand back a compacted attention view
    desc.capabilities.original_rope_positions = True
    snap = ext.MemorySnapshot()
    snap.stamp.valid = True
    snap.stamp.session.id = 1
    snap.stamp.session.incarnation = 1
    snap.stamp.history_revision = 1
    snap.stamp.execution_history = 1
    snap.stamp.view_version = 0
    snap.stamp.frontier = TOTAL
    catalog = []
    for b in range(BLOCKS):
        copies = ext.BlockCopies()
        copies.block.session = snap.stamp.session
        copies.block.id = b
        copies.block.original_start = b * BLOCK_TOKENS
        copies.block.valid_tokens = BLOCK_TOKENS
        copies.block.content_version = 1
        copies.host_version = 1        # the host store holds a current copy of every block
        catalog.append(copies)
    snap.blocks = catalog      # a plain list assignment; pybind returns a copy on read

    block_ids = sorted({p // PAGES_PER_BLOCK for p in selected_blocks})
    mandatory = [b for b in block_ids if b * BLOCK_TOKENS < cfg.sink_tokens]
    budget = ext.MemoryBudget(cfg.budget_tokens, cfg.gen_budget_tokens, TOTAL, 1 << 24)
    plan = ext.plan_working_set(desc, snap, block_ids, mandatory, budget)

    kv_blocks = [vb.block.id for vb in plan.next.blocks]
    kv_tokens = plan.next.visible_tokens
    check(kv_blocks == block_ids,
          f"kvmem's plan keeps the same blocks ({len(kv_blocks)} vs {len(block_ids)})")
    check(kv_tokens == len(block_ids) * BLOCK_TOKENS,
          f"kvmem's plan exposes {kv_tokens} tokens == blocks x {BLOCK_TOKENS}")
    check(kv_tokens == view.seq_len,
          f"the two views agree on the visible length ({kv_tokens} == {view.seq_len})")

    compact = [vb.compact_start for vb in plan.next.blocks]
    check(compact == sorted(compact), "kvmem's compact addresses are ascending")
    check(compact[0] == 0, "kvmem's view starts at 0 like the hand-rolled one")

    print("PASS kvmem_view_test" if not failed else "FAILED kvmem_view_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
