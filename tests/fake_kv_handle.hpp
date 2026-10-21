#pragma once

// In-memory stand-in for one vLLM TP rank's paged-KV shard. The "GPU" is a map of block id
// -> record_bytes, filled with a per-block pattern so spill/restore byte-correctness is
// checkable. No CUDA, no vLLM — just the contract surface the adapter drives.
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "vllm_kvmem/vllm_kv_handle.hpp"

namespace vllm_kvmem {
using kvmem::ExecutionView;
using kvmem::MemoryStamp;

class FakeVllmHandle final : public VllmKvHandle {
public:
    FakeVllmHandle(uint32_t layers, uint32_t kv_heads, uint32_t head_dim, uint32_t block_tokens,
        uint32_t dtype_bytes = 2)
        : geom_{layers, kv_heads, head_dim, block_tokens, dtype_bytes},
          record_bytes_(static_cast<uint64_t>(block_tokens) * kv_heads * head_dim * dtype_bytes * 2 * layers) {
        // session.id/incarnation, history_revision and execution_history must be non-zero for
        // the contract; frontier is advanced by add_block.
        stamp_ = MemoryStamp{true, {1, 1}, 1, 0, 0, 1};
    }
    Geometry geometry() const override { return geom_; }
    std::vector<CatalogBlock> catalog() const override { return catalog_; }
    std::vector<uint64_t> resident() const override { return resident_; }
    MemoryStamp stamp() const noexcept override { return stamp_; }

    void spill_block(uint64_t id, void * dst) const override {
        const auto it = gpu_.find(id);
        if (it == gpu_.end()) { throw std::runtime_error("spill_block: not on device"); }
        std::memcpy(dst, it->second.data(), record_bytes_);
    }
    void restore_block(uint64_t id, const void * src) override {
        auto & buf = gpu_.try_emplace(id, record_bytes_, uint8_t{0}).first->second;
        std::memcpy(buf.data(), src, record_bytes_);
    }
    void publish_view(const ExecutionView & view) override {
        resident_.clear();
        for (const auto & vb : view.blocks) { resident_.push_back(vb.block.id); }
        stamp_ = view.stamp; // the session asserts backend stamp == published view stamp
    }

    // --- test helpers ---
    void add_block(uint64_t id, uint64_t original_start, uint32_t valid_tokens) {
        CatalogBlock c;
        c.session = {1, 1}; // must equal the stamp's session identity
        c.id = id;
        c.original_start = original_start;
        c.valid_tokens = valid_tokens;
        c.content_version = 1;
        c.statistics_version = 1;
        catalog_.push_back(c);
        std::vector<uint8_t> buf(record_bytes_);
        for (uint64_t i = 0; i < record_bytes_; ++i) { buf[i] = uint8_t((id * 31 + i) & 0xff); }
        gpu_.emplace(id, std::move(buf));
        resident_.push_back(id);
        stamp_.frontier = original_start + valid_tokens;
    }
    bool on_device(uint64_t id) const { return gpu_.count(id) > 0; }
    std::vector<uint8_t> device_bytes(uint64_t id) const { return gpu_.at(id); }
    uint64_t record_bytes() const { return record_bytes_; }

private:
    Geometry geom_;
    std::vector<CatalogBlock> catalog_;
    std::unordered_map<uint64_t, std::vector<uint8_t>> gpu_;
    std::vector<uint64_t> resident_;
    MemoryStamp stamp_;
    uint64_t record_bytes_;
};

} // namespace vllm_kvmem
