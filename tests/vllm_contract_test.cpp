// Drives the real kvmem::MemorySession coordinator against VllmMemoryBackend + FakeVllmHandle.
// Validates: (1) the contract is satisfied (descriptor/snapshot/prepare/publish), (2) the sync
// spill path retains host copies and publishes the reduced view, (3) the restore path brings a
// spilled block back byte-for-byte.
#include <cstdio>
#include <cstdlib>

#include "kvmem/memory_contract.hpp"
#include "vllm_kvmem/vllm_backend.hpp"
#include "fake_kv_handle.hpp"

using namespace kvmem;
using namespace vllm_kvmem;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

static MemoryPhase advance_to_ready(MemorySession & s) {
    MemoryPhase phase;
    do { phase = s.advance(); }
    while (phase == MemoryPhase::Spilling || phase == MemoryPhase::Restoring);
    return phase;
}

int main() {
    const uint32_t bt = 16;
    // `ref` is an untouched reference rank (same geometry/bytes) for byte-correctness checks.
    FakeVllmHandle ref(2, 2, 8, bt);
    for (uint64_t i = 0; i < 4; ++i) { ref.add_block(i, i * bt, bt); }
    const uint64_t record_bytes = ref.record_bytes();
    CHECK(record_bytes == 2048);

    auto live = std::make_shared<FakeVllmHandle>(2, 2, 8, bt);
    for (uint64_t i = 0; i < 4; ++i) { live->add_block(i, i * bt, bt); }
    auto backend = std::make_shared<VllmMemoryBackend>(live, 1u << 30, "vllm-rank0-test");
    MemorySession session(backend);

    const auto desc = backend->descriptor();
    CHECK(desc.layout.format == "vllm-paged-kv-v1");
    CHECK(desc.layout.block_tokens == bt);
    CHECK(desc.layout.record_bytes == record_bytes);
    CHECK(desc.layout.planes.size() == 4); // 2 layers x {Key, Value}
    CHECK(desc.capabilities.asynchronous_transfer == false);
    CHECK(desc.capabilities.page_tokens == bt);

    // --- Cycle 1: evict+spill blocks 0,1; keep 2,3. ---
    {
        const auto & plan = session.prepare(backend->stamp(), {2, 3}, {2, 3},
            MemoryBudget{2 * bt, bt, 3 * bt, 4 * record_bytes});
        CHECK(plan.spill.size() == 2);
        CHECK(plan.evict.size() == 2);
        CHECK(plan.restore.empty());
        CHECK(plan.next.blocks.size() == 2);
        CHECK(advance_to_ready(session) == MemoryPhase::Ready);
        const auto view = session.publish();
        CHECK(view.blocks.size() == 2);
        CHECK(view.blocks[0].block.id == 2 && view.blocks[1].block.id == 3);
        CHECK(live->resident() == (std::vector<uint64_t>{2, 3}));
        CHECK(backend->host_copies().count(0) == 1);
        CHECK(backend->host_copies().count(1) == 1);
        CHECK(backend->host_copies().count(2) == 0);
    }

    // --- Cycle 2: restore block 0 (it has a host copy); keep 2,3. ---
    {
        const auto & plan = session.prepare(backend->stamp(), {0, 2, 3}, {0, 2, 3},
            MemoryBudget{3 * bt, bt, 4 * bt, 4 * record_bytes});
        CHECK(plan.restore.size() == 1 && plan.restore[0].id == 0);
        CHECK(plan.spill.empty() && plan.evict.empty());
        CHECK(advance_to_ready(session) == MemoryPhase::Ready);
        const auto view = session.publish();
        CHECK(view.blocks.size() == 3);
        CHECK(live->resident() == (std::vector<uint64_t>{0, 2, 3}));
        CHECK(live->device_bytes(0) == ref.device_bytes(0)); // byte-for-byte restore
    }

    std::fprintf(stderr, "PASS vllm_contract_test\n");
    return 0;
}
