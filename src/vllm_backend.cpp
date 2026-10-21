#include "vllm_kvmem/vllm_backend.hpp"

#include <stdexcept>

namespace vllm_kvmem {
using namespace kvmem;

// ---------------------------------------------------------------------------
// VllmMemoryBackend
// ---------------------------------------------------------------------------

VllmMemoryBackend::VllmMemoryBackend(std::shared_ptr<VllmKvHandle> handle,
    uint64_t pool_capacity_bytes, std::string backend_name, bool async_transfer)
    : handle_(std::move(handle)),
      pool_(std::make_unique<HostKvStorage>(pool_capacity_bytes)),
      backend_name_(std::move(backend_name)),
      async_transfer_(async_transfer) {}

MemoryDescriptor VllmMemoryBackend::descriptor() const {
    const auto g = handle_->geometry();
    if (g.num_layers == 0 || g.num_kv_heads == 0 || g.head_dim == 0 || g.block_tokens == 0) {
        throw MemoryError(MemoryErrorCode::Unsupported,
            "vllm rank geometry is empty; no paged KV to describe");
    }
    const uint64_t per_token_layer_bytes =
        static_cast<uint64_t>(g.num_kv_heads) * g.head_dim * g.dtype_bytes;
    const uint64_t layer_bytes = static_cast<uint64_t>(g.block_tokens) * per_token_layer_bytes * 2;
    const uint64_t record_bytes = layer_bytes * g.num_layers;

    MemoryDescriptor d;
    d.identity.backend = backend_name_;
    d.identity.revision = "1";
    d.identity.model = "vllm-paged-kv";
    d.identity.state_schema = "vllm-paged-kv-v1";
    d.layout.format = "vllm-paged-kv-v1";
    d.layout.version = 1;
    d.layout.block_tokens = g.block_tokens;
    d.layout.record_bytes = record_bytes;
    d.layout.alignment = 256;
    d.layout.planes.reserve(g.num_layers * 2);
    for (uint32_t layer = 0; layer < g.num_layers; ++layer) {
        const uint64_t base = static_cast<uint64_t>(layer) * layer_bytes;
        const uint64_t plane_bytes = static_cast<uint64_t>(g.block_tokens) * per_token_layer_bytes;
        d.layout.planes.push_back(
            {layer, PayloadPlaneKind::Key, base, plane_bytes, 256, per_token_layer_bytes});
        d.layout.planes.push_back({layer, PayloadPlaneKind::Value,
            base + plane_bytes, plane_bytes, 256, per_token_layer_bytes});
    }
    d.capabilities.page_tokens = g.block_tokens;
    d.capabilities.max_position = 1u << 20; // well above any 256K conversation frontier
    d.capabilities.native_payload = true; // native (vLLM) paged-KV payload
    d.capabilities.sparse_view = true; // the view is sparse: only resident blocks
    d.capabilities.original_rope_positions = true; // positions preserved as stored
    d.capabilities.pre_rope_statistics = false;
    d.capabilities.state_checkpoint = false;
    d.capabilities.asynchronous_transfer = async_transfer_; // the KVMEM_ASYNC_TRANSFER switch
    d.validate();
    return d;
}

MemorySnapshot VllmMemoryBackend::snapshot() const {
    MemorySnapshot s;
    s.stamp = handle_->stamp();
    const auto catalog = handle_->catalog();
    s.blocks.reserve(catalog.size());
    for (const auto & c : catalog) {
        BlockCopies bc;
        bc.block.session = c.session;
        bc.block.id = c.id;
        bc.block.original_start = c.original_start;
        bc.block.valid_tokens = c.valid_tokens;
        bc.block.content_version = c.content_version;
        bc.block.statistics_version = c.statistics_version;
        // A retained host copy means a completed current host copy. Absent = none. (Stale-copy
        // detection and LRU host eviction arrive with the policy-store phase.)
        bc.host_version = host_copies_.count(c.id) ? c.content_version : 0;
        s.blocks.push_back(std::move(bc));
    }
    s.resident = handle_->resident();
    return s;
}

MemoryStamp VllmMemoryBackend::stamp() const noexcept { return handle_->stamp(); }

std::unique_ptr<MemoryTransfer> VllmMemoryBackend::prepare(const WorkingSetPlan & plan) {
    if (invalidated_) {
        throw MemoryError(MemoryErrorCode::InvalidSession, "backend invalidated");
    }
    const uint64_t record_bytes = descriptor().layout.record_bytes;
    // The KVMEM_ASYNC_TRANSFER switch: the async arm overlaps the copies on a CUDA transfer
    // stream; the sync arm does blocking D2H/H2D. Both pre-wire the spill records the same way.
    std::unique_ptr<MemoryTransfer> transfer;
    if (async_transfer_) {
        transfer = std::make_unique<VllmAsyncTransfer>(*this, plan);
    } else {
        transfer = std::make_unique<VllmMemoryTransfer>(*this, plan);
    }
    // Reserve ALL spill destinations up front (contract: prepare reserves everything;
    // start_spill cannot reclaim source pages).
    if (async_transfer_) {
        auto & recs = static_cast<VllmAsyncTransfer &>(*transfer).spill_records_;
        recs.reserve(plan.spill.size());
        for (const auto & block : plan.spill) { recs.push_back(pool_->allocate(record_bytes)); }
    } else {
        auto & recs = static_cast<VllmMemoryTransfer &>(*transfer).spill_records_;
        recs.reserve(plan.spill.size());
        for (const auto & block : plan.spill) { recs.push_back(pool_->allocate(record_bytes)); }
    }
    return transfer;
}

void VllmMemoryBackend::invalidate() noexcept { invalidated_ = true; }

// ---------------------------------------------------------------------------
// VllmMemoryTransfer (sync arm)
// ---------------------------------------------------------------------------

VllmMemoryTransfer::VllmMemoryTransfer(
    VllmMemoryBackend & backend, const WorkingSetPlan & plan)
    : backend_(backend), plan_(plan) {}

void VllmMemoryTransfer::start_spill() {
    if (spill_started_) { return; }
    spill_started_ = true;
    if (spill_records_.size() != plan_.spill.size()) {
        error_ = "spill record count mismatch";
        return;
    }
    try {
        for (size_t i = 0; i < plan_.spill.size(); ++i) {
            const uint64_t block_id = plan_.spill[i].id;
            HostKvRecord & rec = spill_records_[i];
            backend_.handle().spill_block(block_id, rec.writable_data()); // D2H (blocking)
            rec.seal(); // immutable; now eligible for host eviction
            backend_.host_copies()[block_id] = std::move(rec); // retain
        }
        spill_done_ = true;
    } catch (const std::exception & e) {
        error_ = e.what();
    }
}

TransferPoll VllmMemoryTransfer::poll() {
    if (!error_.empty()) { return {TransferStatus::Failed, error_}; }
    if (spill_started_ && spill_done_) {
        // Sync arm: the D2H (and later the H2D) completed inline.
        return {TransferStatus::Complete, {}};
    }
    return {TransferStatus::Pending, {}};
}

void VllmMemoryTransfer::start_restore() {
    if (restore_done_) { return; }
    if (!spill_done_) {
        error_ = "start_restore before spill completed";
        return;
    }
    try {
        for (const auto & block : plan_.restore) {
            auto it = backend_.host_copies().find(block.id);
            if (it == backend_.host_copies().end()) {
                throw MemoryError(MemoryErrorCode::Stale,
                    "restore block has no retained host copy");
            }
            backend_.handle().restore_block(block.id, it->second.data()); // H2D (blocking)
        }
        restore_done_ = true;
    } catch (const std::exception & e) {
        error_ = e.what();
    }
}

void VllmMemoryTransfer::publish(const ExecutionView & view) {
    if (published_) { return; }
    if (error_.empty() && (!spill_done_ || !restore_done_)) {
        // Sync arm publishes only once both transfers completed.
        error_ = "publish before transfers completed";
        throw MemoryError(MemoryErrorCode::Pending, error_);
    }
    if (!error_.empty()) {
        throw MemoryError(MemoryErrorCode::TransferFailed, error_);
    }
    backend_.handle().publish_view(view);
    published_ = true;
}

AbortResult VllmMemoryTransfer::abort() noexcept {
    // Sync arm: no outstanding async work. The previous view (and its bytes) remain usable.
    // Retained host copies are left in place (valid, reusable or evictable later).
    return AbortResult::PreviousViewPreserved;
}

} // namespace vllm_kvmem
