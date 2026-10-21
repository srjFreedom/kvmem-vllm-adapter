// The paicat1-style async page pipeline: the KVMEM_ASYNC_TRANSFER performance arm. A
// MemoryTransfer that overlaps the D2H/H2D block copies on a dedicated CUDA transfer stream so
// they run concurrently with the compute stream. Each block is staged through a PINNED host
// scratch buffer: the async copy writes pinned scratch, and the step-boundary harvest (poll)
// moves it into the retained HostKvRecord with a cheap pageable memcpy (the retained copy stays
// in kvmem's own storage policy). The KV tensor is NOT in the TP allreduce, so a KV-only
// transfer stream cannot deadlock the collective. The backend's prepare() pre-wires the spill
// records exactly as for the sync arm.
#include "vllm_kvmem/vllm_backend.hpp"

#include <cstring>

#include <cuda_runtime.h>

namespace vllm_kvmem {
using namespace kvmem;

VllmAsyncTransfer::VllmAsyncTransfer(VllmMemoryBackend & backend, const WorkingSetPlan & plan)
    : backend_(backend), plan_(plan) {
    // Prefer the stream the handle owns (torch keeps it alive); only create one when the handle
    // has none, so the transfer never destroys a stream torch still references.
    stream_ = backend_.handle().transfer_stream();
    if (stream_ == nullptr) {
        // A dedicated, low-priority transfer stream: KV copies overlap compute without
        // contending for the compute stream's resources.
        cudaStreamCreateWithPriority(reinterpret_cast<cudaStream_t *>(&stream_),
            cudaStreamNonBlocking, /*priority=*/-1);
        owns_stream_ = true;
    }
    record_bytes_ = backend_.descriptor().layout.record_bytes;
    spill_count_ = plan_.spill.size();
    restore_count_ = plan_.restore.size();
    // One pinned staging slot per block in the plan (spill slots first, then restore slots).
    const size_t slots = spill_count_ + restore_count_;
    if (slots != 0) {
        scratch_ = backend_.handle().pinned_scratch(slots * record_bytes_);
    }
}

VllmAsyncTransfer::~VllmAsyncTransfer() {
    if (stream_ != nullptr) {
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
        if (owns_stream_) { cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_)); }
        stream_ = nullptr;
    }
}

char * VllmAsyncTransfer::slot(size_t index) const {
    return static_cast<char *>(scratch_) + index * record_bytes_;
}

void VllmAsyncTransfer::start_spill() {
    if (spill_enqueued_) { return; }
    if (spill_records_.size() != plan_.spill.size()) {
        error_ = "spill record count mismatch";
        return;
    }
    if (spill_count_ != 0 && scratch_ == nullptr) {
        error_ = "async spill requires a pinned scratch buffer from the handle";
        return;
    }
    spill_enqueued_ = true;
    try {
        if (scratch_ != nullptr) {
            // Enqueue every D2H into the pinned scratch on the transfer stream (truly async).
            for (size_t i = 0; i < spill_count_; ++i) {
                backend_.handle().spill_block_async(plan_.spill[i].id, slot(i), stream_);
            }
        } else {
            // No blocks to spill: nothing to enqueue; poll() reports Complete immediately.
            spill_enqueued_ = false;
            spill_done_ = true;
        }
    } catch (const std::exception & e) {
        error_ = e.what();
    }
}

TransferPoll VllmAsyncTransfer::poll() {
    if (!error_.empty()) { return {TransferStatus::Failed, error_}; }
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);

    // Restore phase first: once start_restore() enqueued the H2D copies, their completion is
    // what poll() reports. (The spill flag is cleared at the spill harvest, so the restore
    // phase must be distinguished explicitly — otherwise poll() would report Pending forever.)
    if (restore_enqueued_ && !restore_done_) {
        cudaError_t r = cudaStreamQuery(stream);
        if (r == cudaSuccess) {
            restore_done_ = true;
            return {TransferStatus::Complete, {}};
        }
        if (r == cudaErrorNotReady) { return {TransferStatus::Pending, {}}; }
        error_ = cudaGetErrorString(r);
        return {TransferStatus::Failed, {error_}};
    }

    // Spill phase: the step-boundary harvest. Query the transfer stream without blocking compute.
    if (spill_enqueued_ && !spill_done_) {
        cudaError_t r = cudaStreamQuery(stream);
        if (r == cudaSuccess) {
            // The D2H is in: move each pinned slot into its retained record and seal it.
            for (size_t i = 0; i < spill_count_; ++i) {
                HostKvRecord & rec = spill_records_[i];
                if (scratch_ != nullptr) {
                    std::memcpy(rec.writable_data(), slot(i), record_bytes_);
                }
                rec.seal();
                backend_.host_copies()[plan_.spill[i].id] = std::move(rec);
            }
            spill_enqueued_ = false;
            spill_done_ = true;
            return {TransferStatus::Complete, {}};
        }
        if (r == cudaErrorNotReady) { return {TransferStatus::Pending, {}}; }
        error_ = cudaGetErrorString(r);
        return {TransferStatus::Failed, {error_}};
    }

    // Spill finished, restore not yet requested: the session is about to enter the restore phase.
    if (spill_done_) { return {TransferStatus::Complete, {}}; }
    return {TransferStatus::Pending, {}};
}

void VllmAsyncTransfer::start_restore() {
    if (restore_enqueued_) { return; }
    if (!spill_done_) {
        error_ = "start_restore before spill completed";
        return;
    }
    if (restore_count_ != 0 && scratch_ == nullptr) {
        error_ = "async restore requires a pinned scratch buffer from the handle";
        return;
    }
    restore_enqueued_ = true;
    try {
        for (size_t j = 0; j < restore_count_; ++j) {
            const auto & block = plan_.restore[j];
            auto it = backend_.host_copies().find(block.id);
            if (it == backend_.host_copies().end()) {
                throw MemoryError(MemoryErrorCode::Stale,
                    "restore block has no retained host copy");
            }
            // Stage the retained bytes into pinned scratch, then enqueue the H2D (truly async).
            char * dst = slot(spill_count_ + j);
            std::memcpy(dst, it->second.data(), record_bytes_);
            backend_.handle().restore_block_async(block.id, dst, stream_);
        }
    } catch (const std::exception & e) {
        error_ = e.what();
    }
}

void VllmAsyncTransfer::publish(const ExecutionView & view) {
    if (published_) { return; }
    if (error_.empty() && (!spill_done_ || !restore_enqueued_)) {
        error_ = "publish before transfers completed";
        throw MemoryError(MemoryErrorCode::Pending, error_);
    }
    if (!error_.empty()) {
        throw MemoryError(MemoryErrorCode::TransferFailed, error_);
    }
    // Drain the transfer stream before exposing the view (the bytes must be in place).
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
    backend_.handle().publish_view(view);
    published_ = true;
}

AbortResult VllmAsyncTransfer::abort() noexcept {
    // Drain/cancel outstanding work BEFORE releasing any buffers/reservations.
    if (stream_ != nullptr) { cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)); }
    spill_enqueued_ = false;
    restore_enqueued_ = false;
    restore_done_ = false;
    return AbortResult::PreviousViewPreserved;
}

} // namespace vllm_kvmem
