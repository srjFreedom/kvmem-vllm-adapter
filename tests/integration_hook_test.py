# Exercises the out-of-tree integration hook end to end with a FAKE vLLM runner (CPU-backed KV
# tensors): bind() -> on_execute_model_done() -> policy -> prepare/advance/publish. Proves the
# factory hook actually drives a spill/restore cycle and updates the resident view, and that the
# policy is a pure function of the conversation state.
import sys

import torch

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")

import vllm_kvmem_ext as ext
from vllm_integration import KvmemAdapter


def check(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)


class FakeBlockTable:
    def __init__(self, rows):
        self.block_table = rows


class FakeInputBatch:
    def __init__(self, rows):
        self.block_table = type("BT", (), {"block_tables": [FakeBlockTable(rows)]})()


class FakeRunner:
    """The minimal GPUModelRunner surface the hook reads."""

    def __init__(self, kv_caches, rows):
        self.kv_caches = kv_caches
        self.input_batch = FakeInputBatch(rows)


def main():
    num_blocks, block_tokens, kv_heads, head_dim, dtype_bytes, layers = 8, 16, 2, 8, 2, 2
    c_bytes = head_dim * dtype_bytes * 2
    kv_caches = []
    for layer in range(layers):
        buf = torch.arange(num_blocks * kv_heads * block_tokens * c_bytes, dtype=torch.uint8)
        buf = (buf + layer * 3 + 1).remainder(255).add(1)
        kv_caches.append(buf.view(num_blocks, kv_heads, block_tokens, c_bytes))
    ref = {i: kv_caches[0][i].clone() for i in range(num_blocks)}

    runner = FakeRunner(kv_caches, [[5, 6, 7]])
    record_bytes = block_tokens * kv_heads * head_dim * dtype_bytes * 2 * layers
    adapter = KvmemAdapter(
        runner=runner,
        pool_capacity_bytes=num_blocks * record_bytes * 2,
        session_id=1,
        block_tokens=block_tokens,
        num_full_attn_layers=layers,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
        resident_budget_tokens=3 * block_tokens,   # the GPU holds 3 blocks
        async_transfer=False,
    ).bind()

    # The engine has committed all 8 blocks and they are all resident to begin with.
    cat = adapter._handle._catalog
    for i in range(num_blocks):
        cb = ext.CatalogBlock()
        si = ext.SessionIdentity(); si.id = 1; si.incarnation = 1
        cb.session = si; cb.id = i; cb.original_start = i * block_tokens
        cb.valid_tokens = block_tokens; cb.content_version = 1; cb.statistics_version = 1
        cat.append(cb)
    adapter._handle._resident = list(range(num_blocks))
    # The engine's committed logical frontier is the end of the last committed block.
    adapter._handle._stamp.frontier = num_blocks * block_tokens

    # First cycle: the hook must spill the older blocks and publish a 3-block view.
    view = adapter.on_execute_model_done(None)
    check(view is not None, "hook published a view")
    resident = sorted(b.block.id for b in view.blocks)
    check(len(resident) == 3, f"resident shrinks to the 3-block budget (got {resident})")
    check(resident == [5, 6, 7], f"the 3 newest blocks stay resident (got {resident})")
    check(adapter._handle._resident == [5, 6, 7], "handle resident set follows the view")

    # Second cycle with the same conversation: identical decision (pure function, lockstep).
    view2 = adapter.on_execute_model_done(None)
    check(view2 is not None, "second cycle published")
    check(sorted(b.block.id for b in view2.blocks) == resident, "same conversation -> same working set")

    # The evicted blocks' bytes must still be intact on the device (spill does not reclaim).
    bad = [i for i in range(num_blocks) if not torch.equal(kv_caches[0][i], ref[i])]
    check(not bad, f"payload intact after the cycle (bad={bad[:4]})")

    # Restoring: make block 0 mandatory, so the hook brings it back from the host pool.
    runner.input_batch = FakeInputBatch([[0]])
    view3 = adapter.on_execute_model_done(None)
    check(view3 is not None, "restore cycle published")
    check(0 in [b.block.id for b in view3.blocks], "mandatory block 0 restored into the view")
    check(torch.equal(kv_caches[0][0], ref[0]), "block 0 restored byte-for-byte")

    print("PASS integration_hook_test")


if __name__ == "__main__":
    main()
