#pragma once

// The sparse-selection layer for the vLLM graft (KVMem "A mode").
//
// KVMem selects history at --kvmem-block-tokens granularity (128 tokens by default) and always
// keeps a sink prefix and a recent suffix; vLLM addresses KV at 16-token physical pages. This
// class owns the KvMemStore, feeds it the conversation length and the per-block relevance
// scores, and expands the selected selection blocks into the page ids the vLLM block table uses.
//
// Selection only decides WHICH pages belong in the working set; the caller spills the rest to
// the host pool, compacts the survivors to the front of the request's page range (re-applying
// RoPE), and rewrites the block table row.
#include <cstdint>
#include <vector>

#include "kvmem/kvmem_store.hpp"
#include "kvmem/mean_key_retrieval.hpp"

namespace vllm_kvmem {

class VllmSparseSelector {
public:
    struct Config {
        uint32_t select_block_tokens = 128;  // --kvmem-block-tokens (selection granularity)
        uint32_t budget_tokens = 32768;      // --kvmem-budget   (B, history selected per request)
        uint32_t gen_budget_tokens = 16384;  // --kvmem-gen-reserve (R, excluded from B)
        uint32_t sink_tokens = 8192;         // --kvmem-sink-tokens   (kept prefix, inside B)
        uint32_t recent_tokens = 1024;       // --kvmem-recent-tokens (kept suffix, inside B)
        uint32_t page_tokens = 16;           // vLLM physical page
        bool lrfu = true;
        double lrfu_lambda = 0.5;
        double lrfu_top_fraction = 0.9;
        kvmem::GqaScoreAggregation aggregation = kvmem::GqaScoreAggregation::Max;
        kvmem::KvMemMethod method = kvmem::KvMemMethod::Retrieval;
        kvmem::KvMemRetrievalMethod retrieval = kvmem::KvMemRetrievalMethod::MeanK;
    };

    explicit VllmSparseSelector(Config config);

    // Grow or shrink the registered conversation. Shrinking is a truncation: blocks past the new
    // frontier are dropped (the caller owns their payload).
    void set_context_tokens(uint32_t tokens);

    // Per-selection-block relevance (size = selection_blocks()). An empty vector leaves whatever
    // the store's own signal (attention heat / recency) provides.
    void set_block_scores(const std::vector<double> &scores);

    // The working set as vLLM page ids, ascending. `mandatory_pages` are pages the engine must
    // keep this step (the newest write target); they consume ordinary budget slots.
    std::vector<uint32_t> select_pages(const std::vector<uint32_t> &mandatory_pages);

    // Prefill-pressure selection: keep the sink prefix and fill the rest with the NEWEST blocks.
    // Used while a long prompt is still streaming in, when relevance is not yet meaningful but
    // the resident set must stay bounded (kvmem's pick_prefill_pressure_blocks semantics).
    std::vector<uint32_t> select_prefill_pressure_pages(
        const std::vector<uint32_t> &mandatory_pages = {});

    // Pages that fit the selection budget (B / page_tokens), for callers that need the cap.
    uint32_t budget_pages() const;
    uint32_t selection_blocks() const;
    uint32_t tokens_per_selection_block() const { return config_.select_block_tokens; }
    uint32_t pages_per_selection_block() const {
        return config_.select_block_tokens / config_.page_tokens;
    }
    const kvmem::KvMemStore &store() const { return store_; }

private:
    uint32_t block_of_page(uint32_t page) const { return page / pages_per_selection_block(); }

    Config config_;
    kvmem::KvMemStore store_;
    // Selection block -> io tier/slot are unused here (the host pool is the caller's), but the
    // store needs a stable block count, which register_append maintains.
};

} // namespace vllm_kvmem
