#pragma once

// The engine-facing handle the vLLM side implements. The adapter (VllmMemoryBackend /
// VllmMemoryTransfer) talks only to this interface, so it stays decoupled from vLLM's
// Python/worker internals: the hook supplies the KV geometry, the logical block catalog,
// the resident set, and the byte-level spill/restore/publish operations.
//
// One handle per TP rank. Each rank holds a shard of the model's KV heads; the handle
// describes only that rank's shard. Ranks are independent processes (vLLM multiproc
// executor) and never share this handle.
#include <cstdint>
#include <vector>

#include "kvmem/memory_contract.hpp"

namespace vllm_kvmem {

// A logical block as the engine sees it. `id` is the engine's logical block id (shared
// across ranks); the handle maps it to this rank's physical KV storage.
struct CatalogBlock {
    kvmem::SessionIdentity session;
    uint64_t id = 0;
    uint64_t original_start = 0; // first token position of the block
    uint32_t valid_tokens = 0;   // tokens actually written (<= block_tokens)
    uint64_t content_version = 0;
    uint64_t statistics_version = 0;
};

class VllmKvHandle {
public:
    virtual ~VllmKvHandle() = default;

    // The KV geometry for this rank. block_tokens is the vLLM paged-block size (16).
    // num_kv_heads is the TP-sharded head count for THIS rank. bytes_per_token_kv is the
    // K+V payload bytes for one token on this rank (num_kv_heads * head_dim * 2 * dtype_bytes).
    struct Geometry {
        uint32_t num_layers = 0;     // full-attention layers carrying a paged KV
        uint32_t num_kv_heads = 0;   // KV heads resident on this rank
        uint32_t head_dim = 0;
        uint32_t block_tokens = 0;   // vLLM block size
        uint32_t dtype_bytes = 2;    // 2 = FP16/BF16
    };
    virtual Geometry geometry() const = 0;

    // Complete logical catalog in chronological order (every block ever committed).
    virtual std::vector<CatalogBlock> catalog() const = 0;
    // Complete CURRENT versions owned by the device view (the resident set).
    virtual std::vector<uint64_t> resident() const = 0;
    virtual kvmem::MemoryStamp stamp() const noexcept = 0;

    // D2H: copy this rank's KV payload for `logical_block` into `dst` (record_bytes). Read-only
    // on the device; the source GPU block is NOT reclaimed by this call (contract: start_spill
    // cannot reclaim source pages).
    virtual void spill_block(uint64_t logical_block, void * dst) const = 0;
    // H2D: copy this rank's KV payload from `src` (record_bytes) into `logical_block`,
    // allocating the device block if it was evicted. Mutates the device; may be called only
    // after the spill of the evicted copy completed.
    virtual void restore_block(uint64_t logical_block, const void * src) = 0;
    // Atomically expose the prepared view at an engine-safe boundary: the attention block
    // table is updated so only `view.blocks` are readable. Prior bytes remain intact until
    // the evicted pages are reclaimed.
    virtual void publish_view(const kvmem::ExecutionView & view) = 0;

    // --- Optional async extension (the KVMEM_ASYNC_TRANSFER performance arm) ---
    // A handle that can overlap the copies returns a PINNED host scratch buffer of at least
    // `bytes` (stable for the handle's lifetime). The async transfer stages each block through
    // it and enqueues the copy on a CUDA stream. Returning nullptr keeps the blocking copy path
    // (the async arm then behaves like the sync arm but still harvests at step boundaries).
    virtual void * pinned_scratch(size_t /*bytes*/) { return nullptr; }
    // The CUDA stream the async copies run on. A handle that owns a torch stream returns its
    // cudaStream_t so the transfer never creates/destroys a stream behind torch's back
    // (destroying a stream torch still references corrupts its allocator bookkeeping).
    virtual void * transfer_stream() { return nullptr; }
    // D2H into the pinned scratch slot `dst`, enqueued (non-blocking) on CUDA `stream`.
    // The default falls back to the blocking copy so a handle need not opt in.
    virtual void spill_block_async(uint64_t logical_block, void * dst, void * /*stream*/) const {
        spill_block(logical_block, dst);
    }
    // H2D from the pinned scratch slot `src`, enqueued (non-blocking) on CUDA `stream`.
    virtual void restore_block_async(uint64_t logical_block, const void * src, void * /*stream*/) {
        restore_block(logical_block, src);
    }
};

} // namespace vllm_kvmem
