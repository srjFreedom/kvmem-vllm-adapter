# Validates the REAL VllmKvHandleVllm copy logic (block indexing, the 4D (B,H,N,C) paged-KV
# layout, the D2H/H2D byte round-trip) against a fake vLLM engine with CPU-backed KV tensors.
# This exercises the same torch code path the GPU uses (.cpu() / .to(device) are no-ops on CPU),
# so the copy logic is validated now; the actual GPU<->host transfer is exercised later.
import ctypes
import sys

import numpy as np
import torch

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")
import vllm_kvmem_ext as ext
from vllm_handle import VllmKvHandleVllm


def check(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)


def main():
    num_layers = 2
    kv_heads = 2
    head_dim = 8
    block_tokens = 16
    dtype_bytes = 2
    num_blocks = 8
    record_bytes = block_tokens * kv_heads * head_dim * dtype_bytes * 2 * num_layers
    check(record_bytes == 2048, "record_bytes")

    # Fake vLLM engine: per-layer KV tensors on CPU, shaped (B, H, N, C) in bytes.
    # C = state_content_size_bytes = head_dim * dtype_bytes * 2 (K+V), N = block_tokens.
    c_bytes = head_dim * dtype_bytes * 2
    n_states = block_tokens
    kv_caches = []
    for layer in range(num_layers):
        # Fill each block with a per-(layer, block) pattern so a layout/index error is caught.
        buf = torch.arange(num_blocks * kv_heads * n_states * c_bytes, dtype=torch.uint8)
        buf = (buf + layer * 7 + 1).remainder(255).add(1)  # nonzero pattern, varies per layer
        kv_caches.append(buf.view(num_blocks, kv_heads, n_states, c_bytes))

    catalog = []
    for i in range(num_blocks):
        cb = ext.CatalogBlock()
        si = ext.SessionIdentity()
        si.id = 1
        si.incarnation = 1
        cb.session = si
        cb.id = i
        cb.original_start = i * block_tokens
        cb.valid_tokens = block_tokens
        cb.content_version = 1
        cb.statistics_version = 1
        catalog.append(cb)

    handle = VllmKvHandleVllm(
        kv_caches=kv_caches,
        block_tables=None,
        geometry=ext.Geometry(num_layers, kv_heads, head_dim, block_tokens, dtype_bytes),
        session_id=1,
        incarnation=1,
        catalog=catalog,
        resident=list(range(num_blocks)),
        frontier=num_blocks * block_tokens,
        execution_history=1,
        history_revision=1,
    )

    backend = ext.VllmMemoryBackend(handle, 1 << 30, "vllm-rank0-real")
    session = ext.MemorySession(backend)

    # Reference: the original GPU (CPU here) blocks, untouched.
    ref = {i: kv_caches[0][i].clone() for i in range(num_blocks)}

    # Cycle 1: evict+spill blocks 0,1; keep 2..7.
    st = handle.stamp()
    plan = session.prepare(st, list(range(2, num_blocks)), list(range(2, num_blocks)),
                           ext.MemoryBudget((num_blocks - 2) * block_tokens, block_tokens,
                                            (num_blocks - 1) * block_tokens, num_blocks * record_bytes))
    check(len(plan.spill) == 2, "cycle1 spill count")
    phase = session.advance()
    while phase in (ext.MemoryPhase.Spilling, ext.MemoryPhase.Restoring):
        phase = session.advance()
    check(phase == ext.MemoryPhase.Ready, "cycle1 ready")
    view = session.publish()
    check([b.block.id for b in view.blocks] == list(range(2, num_blocks)), "cycle1 view")
    check(handle.resident() == list(range(2, num_blocks)), "cycle1 resident")

    # Cycle 2: restore block 0 (it has a host copy); keep 2..7.
    st = handle.stamp()
    plan = session.prepare(st, [0] + list(range(2, num_blocks)), [0] + list(range(2, num_blocks)),
                           ext.MemoryBudget((num_blocks - 1) * block_tokens, block_tokens,
                                            num_blocks * block_tokens, num_blocks * record_bytes))
    check(len(plan.restore) == 1 and plan.restore[0].id == 0, "cycle2 restore block 0")
    phase = session.advance()
    while phase in (ext.MemoryPhase.Spilling, ext.MemoryPhase.Restoring):
        phase = session.advance()
    check(phase == ext.MemoryPhase.Ready, "cycle2 ready")
    view = session.publish()
    check([b.block.id for b in view.blocks] == [0] + list(range(2, num_blocks)), "cycle2 view")
    check(handle.resident() == [0] + list(range(2, num_blocks)), "cycle2 resident")

    # Byte-for-byte: the restored block 0 must match the untouched reference.
    check(torch.equal(kv_caches[0][0], ref[0]), "cycle2 block 0 byte-for-byte")

    print("PASS vllm_real_handle_test")


if __name__ == "__main__":
    main()
