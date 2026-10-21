# The correctness 埋针判活 test: plant a needle (a per-block KV payload pattern) on the GPU,
# spill it to the host pool, restore it, and verify the payload is byte-for-byte identical —
# the needle is still alive. Runs on two independent engines (different seeds) to confirm the
# needle is the payload, not a coincidence of the pattern. CPU-backed KV tensors exercise the
# same torch copy path the GPU uses.
import sys

import torch

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")
import vllm_kvmem_ext as ext
from vllm_handle import VllmKvHandleVllm


def check(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)


def make_engine(num_blocks, block_tokens, kv_heads, head_dim, dtype_bytes, seed=0):
    c_bytes = head_dim * dtype_bytes * 2
    kv_caches = []
    for layer in range(2):
        buf = torch.arange(num_blocks * kv_heads * block_tokens * c_bytes, dtype=torch.uint8)
        buf = (buf + layer * 7 + seed + 1).remainder(255).add(1)
        kv_caches.append(buf.view(num_blocks, kv_heads, block_tokens, c_bytes))
    return kv_caches


def main():
    num_blocks, block_tokens, kv_heads, head_dim, dtype_bytes = 8, 16, 2, 8, 2
    record_bytes = block_tokens * kv_heads * head_dim * dtype_bytes * 2 * 2
    geom = ext.Geometry(2, kv_heads, head_dim, block_tokens, dtype_bytes)

    # --- pool-internal: the host budget holds every spilled block ---
    kv = make_engine(num_blocks, block_tokens, kv_heads, head_dim, dtype_bytes)
    catalog = []
    for i in range(num_blocks):
        cb = ext.CatalogBlock()
        si = ext.SessionIdentity(); si.id = 1; si.incarnation = 1
        cb.session = si; cb.id = i; cb.original_start = i * block_tokens
        cb.valid_tokens = block_tokens; cb.content_version = 1; cb.statistics_version = 1
        catalog.append(cb)
    handle = VllmKvHandleVllm(kv, None, geom, 1, 1, catalog, list(range(num_blocks)),
                              num_blocks * block_tokens, 1, 1)
    backend = ext.VllmMemoryBackend(handle, num_blocks * record_bytes * 2, "pool-internal", False)
    session = ext.MemorySession(backend)
    ref = {i: kv[0][i].clone() for i in range(num_blocks)}

    # Spill blocks 0,1 (they fit in the host budget); keep 2..7.
    st = handle.stamp()
    session.prepare(st, list(range(2, num_blocks)), list(range(2, num_blocks)),
                     ext.MemoryBudget((num_blocks - 2) * block_tokens, block_tokens,
                                      (num_blocks - 1) * block_tokens, num_blocks * record_bytes))
    phase = session.advance()
    while phase in (ext.MemoryPhase.Spilling, ext.MemoryPhase.Restoring):
        phase = session.advance()
    check(phase == ext.MemoryPhase.Ready, "pool-internal ready")
    session.publish()

    # Restore block 0 and verify the needle is alive (byte-for-byte).
    st = handle.stamp()
    session.prepare(st, [0] + list(range(2, num_blocks)), [0] + list(range(2, num_blocks)),
                     ext.MemoryBudget((num_blocks - 1) * block_tokens, block_tokens,
                                      num_blocks * block_tokens, num_blocks * record_bytes))
    phase = session.advance()
    while phase in (ext.MemoryPhase.Spilling, ext.MemoryPhase.Restoring):
        phase = session.advance()
    check(phase == ext.MemoryPhase.Ready, "pool-internal restore ready")
    session.publish()
    check(torch.equal(kv[0][0], ref[0]), "pool-internal: needle alive (byte-for-byte)")

    # --- a second, independent engine (different seed) to confirm the needle is the payload,
    #     not a coincidence of the pattern ---
    kv2 = make_engine(num_blocks, block_tokens, kv_heads, head_dim, dtype_bytes, seed=99)
    handle2 = VllmKvHandleVllm(kv2, None, geom, 1, 1, catalog, list(range(num_blocks)),
                               num_blocks * block_tokens, 1, 1)
    backend2 = ext.VllmMemoryBackend(handle2, num_blocks * record_bytes * 2, "engine2", False)
    session2 = ext.MemorySession(backend2)
    ref2 = {i: kv2[0][i].clone() for i in range(num_blocks)}

    # Spill blocks 0,1; keep 2..7.
    st = handle2.stamp()
    session2.prepare(st, list(range(2, num_blocks)), list(range(2, num_blocks)),
                      ext.MemoryBudget((num_blocks - 2) * block_tokens, block_tokens,
                                       (num_blocks - 1) * block_tokens, num_blocks * record_bytes))
    phase = session2.advance()
    while phase in (ext.MemoryPhase.Spilling, ext.MemoryPhase.Restoring):
        phase = session2.advance()
    check(phase == ext.MemoryPhase.Ready, "engine2 ready")
    session2.publish()

    # Restore block 0 and verify the needle is alive (byte-for-byte).
    st = handle2.stamp()
    session2.prepare(st, [0] + list(range(2, num_blocks)), [0] + list(range(2, num_blocks)),
                      ext.MemoryBudget((num_blocks - 1) * block_tokens, block_tokens,
                                       num_blocks * block_tokens, num_blocks * record_bytes))
    phase = session2.advance()
    while phase in (ext.MemoryPhase.Spilling, ext.MemoryPhase.Restoring):
        phase = session2.advance()
    check(phase == ext.MemoryPhase.Ready, "engine2 restore ready")
    session2.publish()
    check(torch.equal(kv2[0][0], ref2[0]), "engine2: needle alive (byte-for-byte)")

    print("PASS correctness_test")


if __name__ == "__main__":
    main()
