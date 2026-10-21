# The sync-vs-async A/B deep-overload benchmark + the 256K-in-64K-VRAM capacity test.
#
# A/B: each arm runs the SAME deep-overload workload — evict most of the resident blocks to the
# host pool, then restore them — repeated many times after a warm-up, so allocator/stream setup
# does not skew the numbers. The sync arm blocks on every D2H/H2D; the async arm stages through
# pinned memory on a transfer stream, so the copies overlap rather than serialize.
#
# Capacity: a 256K-token conversation in a 64K-token GPU KV budget. The excess spills to the host
# pool; the host payload is ~38.2 KB/token (q8_0), so 256K ~= 9.6 GiB and a 48 GiB pool holds ~5.
#
# GPU required (drop --cpu). Run only when the serving vLLM is stopped.
import argparse
import sys
import time

import torch


def make_kv(num_blocks, block_tokens, kv_heads, head_dim, dtype_bytes, layers, device):
    c_bytes = head_dim * dtype_bytes * 2
    kv = []
    for layer in range(layers):
        buf = torch.arange(num_blocks * kv_heads * block_tokens * c_bytes, dtype=torch.uint8,
                           device=device)
        buf = (buf + layer * 7 + 1).remainder(255).add(1)
        kv.append(buf.view(num_blocks, kv_heads, block_tokens, c_bytes))
    return kv


def run_arm(arm, args, record_bytes, device):
    sys.path.insert(0, "build/python")
    sys.path.insert(0, "src")
    import vllm_kvmem_ext as ext
    from vllm_handle import VllmKvHandleVllm

    kv = make_kv(args.num_blocks, args.block_tokens, args.kv_heads, args.head_dim,
                 args.dtype_bytes, args.layers, device)
    geom = ext.Geometry(args.layers, args.kv_heads, args.head_dim, args.block_tokens,
                        args.dtype_bytes)
    catalog = []
    for i in range(args.num_blocks):
        cb = ext.CatalogBlock()
        si = ext.SessionIdentity(); si.id = 1; si.incarnation = 1
        cb.session = si; cb.id = i; cb.original_start = i * args.block_tokens
        cb.valid_tokens = args.block_tokens; cb.content_version = 1; cb.statistics_version = 1
        catalog.append(cb)
    handle = VllmKvHandleVllm(kv, None, geom, 1, 1, catalog, list(range(args.num_blocks)),
                              args.num_blocks * args.block_tokens, 1, 1)
    backend = ext.VllmMemoryBackend(handle, args.num_blocks * record_bytes * 2, arm,
                                    arm == "async")
    session = ext.MemorySession(backend)

    evict = args.evict
    keep = list(range(evict, args.num_blocks))       # resident working set
    allb = list(range(args.num_blocks))
    # MemoryBudget(selected_tokens, reserve_tokens, device_tokens, host_payload_bytes).
    spill_budget = ext.MemoryBudget(len(keep) * args.block_tokens, 0,
                                    len(keep) * args.block_tokens, evict * record_bytes)
    restore_budget = ext.MemoryBudget(args.num_blocks * args.block_tokens, 0,
                                      args.num_blocks * args.block_tokens, evict * record_bytes)

    def one_cycle():
        st = handle.stamp()
        session.prepare(st, keep, keep, spill_budget)
        phase = session.advance()
        while phase in (ext.MemoryPhase.Spilling, ext.MemoryPhase.Restoring):
            phase = session.advance()
        session.publish()
        st = handle.stamp()
        session.prepare(st, allb, allb, restore_budget)
        phase = session.advance()
        while phase in (ext.MemoryPhase.Spilling, ext.MemoryPhase.Restoring):
            phase = session.advance()
        session.publish()

    for _ in range(args.warmup):
        one_cycle()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        one_cycle()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.iters
    # Bytes moved per cycle: evict blocks out and back (2 directions).
    mb_per_s = (2 * evict * record_bytes) / dt / (1 << 20)
    return dt * 1000, mb_per_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="CPU-backed (no GPU; timing not meaningful)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--num-blocks", type=int, default=64)
    ap.add_argument("--evict", type=int, default=48, help="blocks spilled+restored per cycle")
    ap.add_argument("--block-tokens", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=8)
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    device = "cpu" if args.cpu else "cuda"
    record_bytes = (args.block_tokens * args.kv_heads * args.head_dim * args.dtype_bytes * 2 *
                    args.layers)
    assert args.evict <= args.num_blocks

    print(f"A/B deep-overload: {args.layers} layers x {args.kv_heads} kv-heads, "
          f"evict {args.evict}/{args.num_blocks} blocks per cycle, record_bytes={record_bytes}")
    sync_ms, sync_bw = run_arm("sync", args, record_bytes, device)
    print(f"  sync  arm: {sync_ms:7.3f} ms/cycle  ({sync_bw:7.1f} MiB/s)")
    if args.cpu:
        print("  async arm: (GPU required; skipped in CPU mode)")
        print("  speedup:   n/a")
    else:
        async_ms, async_bw = run_arm("async", args, record_bytes, device)
        print(f"  async arm: {async_ms:7.3f} ms/cycle  ({async_bw:7.1f} MiB/s)")
        print(f"  speedup:   {sync_ms / async_ms:5.2f}x")

    # Capacity: a 256K-token conversation in a 64K-token GPU KV budget.
    total_tokens = 256 * 1024
    gpu_budget_tokens = 64 * 1024
    host_payload_per_token = 38.2 * 1024  # q8_0, ~38.2 KB/token
    spilled_tokens = total_tokens - gpu_budget_tokens
    host_bytes = spilled_tokens * host_payload_per_token
    print("\nCapacity (256K-in-64K-VRAM):")
    print(f"  total tokens:        {total_tokens}")
    print(f"  GPU KV budget:       {gpu_budget_tokens} tokens")
    print(f"  spilled to host:     {spilled_tokens} tokens")
    print(f"  host payload:        {host_bytes / (1 << 30):.1f} GiB")
    print(f"  48 GiB pool holds:   "
          f"{int(48 * (1 << 30) / (total_tokens * host_payload_per_token))} conversations")


if __name__ == "__main__":
    main()
