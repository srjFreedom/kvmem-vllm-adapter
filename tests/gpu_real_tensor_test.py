# The GPU-mode real-tensor test: exercises both arms on real GPU-backed KV tensors. Verifies a
# spill -> restore cycle leaves every block byte-for-byte identical (the needle stays alive) and
# times repeated cycles (with a warm-up, so allocator/stream setup does not skew the numbers).
#
# GPU required: run only when the serving vLLM is stopped.
import sys
import time

import torch


def check(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)


def make_kv(num_blocks, block_tokens, kv_heads, head_dim, dtype_bytes, layers, device="cuda"):
    c_bytes = head_dim * dtype_bytes * 2
    kv = []
    for layer in range(layers):
        buf = torch.arange(num_blocks * kv_heads * block_tokens * c_bytes, dtype=torch.uint8,
                           device=device)
        buf = (buf + layer * 7 + 1).remainder(255).add(1)
        kv.append(buf.view(num_blocks, kv_heads, block_tokens, c_bytes))
    return kv


def main():
    sys.path.insert(0, "build/python")
    sys.path.insert(0, "src")
    import vllm_kvmem_ext as ext
    from vllm_handle import VllmKvHandleVllm

    check(torch.cuda.is_available(), "CUDA required")
    num_blocks, block_tokens, kv_heads, head_dim, dtype_bytes = 32, 16, 2, 8, 2
    layers = 4
    evict = 8          # blocks spilled each cycle
    cycles, warmup = 20, 3
    record_bytes = block_tokens * kv_heads * head_dim * dtype_bytes * 2 * layers
    geom = ext.Geometry(layers, kv_heads, head_dim, block_tokens, dtype_bytes)

    catalog = []
    for i in range(num_blocks):
        cb = ext.CatalogBlock()
        si = ext.SessionIdentity(); si.id = 1; si.incarnation = 1
        cb.session = si; cb.id = i; cb.original_start = i * block_tokens
        cb.valid_tokens = block_tokens; cb.content_version = 1; cb.statistics_version = 1
        catalog.append(cb)

    def build(name, flag):
        kv = make_kv(num_blocks, block_tokens, kv_heads, head_dim, dtype_bytes, layers)
        handle = VllmKvHandleVllm(kv, None, geom, 1, 1, catalog, list(range(num_blocks)),
                                  num_blocks * block_tokens, 1, 1)
        backend = ext.VllmMemoryBackend(handle, num_blocks * record_bytes * 2, name, flag)
        return kv, handle, ext.MemorySession(backend)

    def cycle(session, handle, num_blocks, evict, block_tokens, record_bytes):
        # Spill the oldest `evict` blocks, then restore them.
        st = handle.stamp()
        session.prepare(st, list(range(evict, num_blocks)), list(range(evict, num_blocks)),
                        ext.MemoryBudget((num_blocks - evict) * block_tokens, 0,
                                         (num_blocks - evict) * block_tokens, evict * record_bytes))
        phase = session.advance()
        while phase in (ext.MemoryPhase.Spilling, ext.MemoryPhase.Restoring):
            phase = session.advance()
        check(phase == ext.MemoryPhase.Ready, "spill ready")
        session.publish()

        st = handle.stamp()
        session.prepare(st, list(range(num_blocks)), list(range(num_blocks)),
                        ext.MemoryBudget(num_blocks * block_tokens, 0,
                                         num_blocks * block_tokens, evict * record_bytes))
        phase = session.advance()
        while phase in (ext.MemoryPhase.Spilling, ext.MemoryPhase.Restoring):
            phase = session.advance()
        check(phase == ext.MemoryPhase.Ready, "restore ready")
        session.publish()

    results = {}
    for flag, name in ((False, "sync"), (True, "async")):
        kv, handle, session = build(name, flag)
        ref = {i: kv[0][i].clone() for i in range(num_blocks)}
        for _ in range(warmup):
            cycle(session, handle, num_blocks, evict, block_tokens, record_bytes)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(cycles):
            cycle(session, handle, num_blocks, evict, block_tokens, record_bytes)
        torch.cuda.synchronize()
        per_cycle = (time.perf_counter() - t0) / cycles * 1000
        # The needle must be alive on every block after the final restore.
        bad = [i for i in range(num_blocks) if not torch.equal(kv[0][i], ref[i])]
        check(not bad, f"{name}: blocks identical after cycles (bad={bad[:4]})")
        results[name] = per_cycle
        print(f"{name:5s} arm: {per_cycle:7.3f} ms/cycle  ({evict}/{num_blocks} blocks "
              f"spill+restore, {layers} layers)  needle alive")

    print(f"speedup: {results['sync'] / results['async']:.2f}x")
    print("PASS gpu_real_tensor_test")

    # Release the pinned staging while the CUDA context is alive.
    for flag, name in ((False, "sync"), (True, "async")):
        pass
    import gc
    gc.collect()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
