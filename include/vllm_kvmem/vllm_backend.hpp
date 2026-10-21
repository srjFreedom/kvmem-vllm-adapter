#pragma once

// VllmMemoryBackend / VllmMemoryTransfer: the out-of-tree KVMem adapter for one vLLM TP
// rank. The backend describes this rank's paged-KV shard through the kvmem contract and
// retains spilled host copies; the transfer executes the spill/restore/publish at engine-
// safe boundaries. The sync arm (this file) does blocking D2H/H2D; the async arm layers a
// PendingDemote transfer-stream pipeline over the same skeleton (see the KVMEM_ASYNC_TRANSFER
// switch, added in the async phase).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "kvmem/host_kv_storage.hpp"
#include "kvmem/memory_contract.hpp"
#include "vllm_kvmem/vllm_kv_handle.hpp"

namespace vllm_kvmem {

class VllmMemoryTransfer;

class VllmMemoryBackend final : public kvmem::MemoryBackend {
public:
    // handle: the engine's per-rank KV handle. pool_capacity: this rank's host-pool byte
    // budget (0 = unlimited). backend_name: identity string (e.g. "vllm-rank0").
    // async_transfer: the KVMEM_ASYNC_TRANSFER switch (sync arm = false, async arm = true).
    VllmMemoryBackend(std::shared_ptr<VllmKvHandle> handle,
        uint64_t pool_capacity_bytes, std::string backend_name, bool async_transfer = false);

    kvmem::MemoryDescriptor descriptor() const override;
    kvmem::MemorySnapshot snapshot() const override;
    kvmem::MemoryStamp stamp() const noexcept override;
    std::unique_ptr<kvmem::MemoryTransfer> prepare(const kvmem::WorkingSetPlan &) override;
    void invalidate() noexcept override;

    // Transfer-facing access (the transfer is always owned by this backend's session).
    VllmKvHandle & handle() const { return *handle_; }
    kvmem::HostKvStorage & pool() const { return *pool_; }
    std::unordered_map<uint64_t, kvmem::HostKvRecord> & host_copies() { return host_copies_; }
    const std::unordered_map<uint64_t, kvmem::HostKvRecord> & host_copies() const { return host_copies_; }

private:
    std::shared_ptr<VllmKvHandle> handle_;
    std::unique_ptr<kvmem::HostKvStorage> pool_;
    std::string backend_name_;
    bool async_transfer_ = false; // the KVMEM_ASYNC_TRANSFER switch
    // Retained host copies, keyed by logical block id. A block present here has a completed
    // current host copy (snapshot host_version = its content_version); absent = no host copy.
    std::unordered_map<uint64_t, kvmem::HostKvRecord> host_copies_;
    bool invalidated_ = false;
    friend class VllmMemoryTransfer;
    friend class VllmAsyncTransfer;
};

// Sync arm: start_spill blocks on the D2H, poll reports Complete, start_restore blocks on
// the H2D. publish atomically exposes the view at the engine-safe boundary. abort preserves
// the previous view (bytes AND view remain usable).
class VllmMemoryTransfer final : public kvmem::MemoryTransfer {
public:
    VllmMemoryTransfer(VllmMemoryBackend & backend, const kvmem::WorkingSetPlan & plan);

    void start_spill() override;
    kvmem::TransferPoll poll() override;
    void start_restore() override;
    void publish(const kvmem::ExecutionView &) override;
    kvmem::AbortResult abort() noexcept override;

private:
    VllmMemoryBackend & backend_;
    kvmem::WorkingSetPlan plan_;
    // Host records allocated in prepare() for each spill block, in plan.spill order. The
    // D2H writes into writable_data(); seal() then moves the record into the backend's
    // retained host_copies_.
    std::vector<kvmem::HostKvRecord> spill_records_;
    bool spill_started_ = false;
    bool spill_done_ = false;
    bool restore_done_ = false;
    bool published_ = false;
    std::string error_;
    // The backend's prepare() allocates the spill records up front (contract: prepare reserves
    // ALL destinations), so it needs write access to this private member.
    friend class VllmMemoryBackend;
};

// Async arm (the KVMEM_ASYNC_TRANSFER performance path): overlaps the D2H/H2D block copies on a
// dedicated CUDA transfer stream so they run concurrently with the compute stream. start_spill /
// start_restore enqueue the copies (async); poll() is the step-boundary harvest (it also seals
// the host records once the D2H lands); abort() drains the stream before releasing any
// reservation. The backend's prepare() pre-wires the spill records exactly as for the sync arm.
class VllmAsyncTransfer final : public kvmem::MemoryTransfer {
public:
    VllmAsyncTransfer(VllmMemoryBackend & backend, const kvmem::WorkingSetPlan & plan);
    ~VllmAsyncTransfer() override;

    void start_spill() override;
    kvmem::TransferPoll poll() override;
    void start_restore() override;
    void publish(const kvmem::ExecutionView &) override;
    kvmem::AbortResult abort() noexcept override;

private:
    // Pinned-staging slot `index` (spill slots first, then restore slots).
    char * slot(size_t index) const;
    VllmMemoryBackend & backend_;
    kvmem::WorkingSetPlan plan_;
    std::vector<kvmem::HostKvRecord> spill_records_; // pre-wired by the backend's prepare()
    bool spill_enqueued_ = false;
    bool spill_done_ = false;
    bool restore_enqueued_ = false;
    bool restore_done_ = false;
    bool published_ = false;
    std::string error_;
    // Pinned host staging: each block's async D2H lands here first, then a cheap pageable memcpy
    // moves it into the retained HostKvRecord (so the retained copy keeps kvmem's own storage
    // policy). nullptr = the handle cannot stage; the arm falls back to blocking copies.
    void * scratch_ = nullptr;
    uint64_t record_bytes_ = 0;
    size_t spill_count_ = 0;
    size_t restore_count_ = 0;
    // The dedicated CUDA transfer stream, held as void* so this header stays CUDA-free (the
    // sync arm and the contract tests do not link CUDA). The CUDA translation unit casts it.
    void * stream_ = nullptr;
    bool owns_stream_ = false; // only destroy a stream this transfer created
    friend class VllmMemoryBackend;
};

} // namespace vllm_kvmem
