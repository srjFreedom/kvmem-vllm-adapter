#include "vllm_kvmem/vllm_sparse_selector.hpp"

#include <algorithm>
#include <stdexcept>

namespace vllm_kvmem {
using namespace kvmem;

namespace {

KvMemStoreConfig build_store_config(const VllmSparseSelector::Config &c) {
    if (c.page_tokens == 0 || c.select_block_tokens == 0 ||
        c.select_block_tokens % c.page_tokens != 0) {
        throw std::invalid_argument("selection block must be a positive multiple of the page");
    }
    KvMemStoreConfig cfg;
    cfg.block_tokens = c.select_block_tokens;
    cfg.select_budget = c.budget_tokens;
    cfg.gen_budget = c.gen_budget_tokens;
    cfg.lrfu = c.lrfu;
    cfg.lrfu_lambda = c.lrfu_lambda;
    cfg.lrfu_top_fraction = c.lrfu_top_fraction;
    cfg.select_method = c.method;
    cfg.retrieval_method = c.retrieval;
    cfg.select_policy = KvMemSelectPolicy::TopK;
    // Token-based sink/recent bands are resolved into block counts once the budget and the
    // selection block size are known (the CLI resolves them the same way).
    const auto keep = resolve_kvmem_keep_allocation(cfg.block_tokens, cfg.select_budget,
        /*sink_blocks=*/-1, /*recent_blocks=*/-1, static_cast<int64_t>(c.sink_tokens),
        static_cast<int64_t>(c.recent_tokens));
    cfg.sink_blocks = keep.sink_blocks;
    cfg.recent_blocks = keep.recent_blocks;
    return cfg;
}

} // namespace

VllmSparseSelector::VllmSparseSelector(Config config)
    : config_(config), store_(build_store_config(config)) {}

void VllmSparseSelector::set_context_tokens(uint32_t tokens) {
    const uint32_t current = store_.total_tokens();
    if (tokens > current) {
        store_.register_append(tokens - current);
    } else if (tokens < current) {
        store_.truncate_to(tokens);
    }
}

void VllmSparseSelector::set_block_scores(const std::vector<double> &scores) {
    if (scores.empty()) { return; }
    if (scores.size() != store_.block_count()) {
        throw std::invalid_argument("one score per selection block is required");
    }
    store_.set_retrieval_scores(scores);
}

std::vector<uint32_t> VllmSparseSelector::select_pages(
    const std::vector<uint32_t> &mandatory_pages) {
    const uint32_t ppb = pages_per_selection_block();
    std::vector<uint32_t> mandatory_blocks;
    mandatory_blocks.reserve(mandatory_pages.size());
    for (uint32_t page : mandatory_pages) {
        const uint32_t block = block_of_page(page);
        if (std::find(mandatory_blocks.begin(), mandatory_blocks.end(), block) ==
            mandatory_blocks.end()) {
            mandatory_blocks.push_back(block);
        }
    }
    const std::vector<uint32_t> blocks = store_.pick_topk_blocks(mandatory_blocks);

    std::vector<uint32_t> pages;
    pages.reserve(blocks.size() * ppb);
    for (uint32_t block : blocks) {
        for (uint32_t i = 0; i < ppb; ++i) {
            pages.push_back(block * ppb + i);
        }
    }
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    return pages;
}

std::vector<uint32_t> VllmSparseSelector::select_prefill_pressure_pages(
    const std::vector<uint32_t> &mandatory_pages) {
    const uint32_t ppb = pages_per_selection_block();
    std::vector<uint32_t> mandatory_blocks;
    for (uint32_t page : mandatory_pages) {
        const uint32_t block = block_of_page(page);
        if (std::find(mandatory_blocks.begin(), mandatory_blocks.end(), block) ==
            mandatory_blocks.end()) {
            mandatory_blocks.push_back(block);
        }
    }
    const std::vector<uint32_t> blocks = store_.pick_prefill_pressure_blocks(mandatory_blocks);
    std::vector<uint32_t> pages;
    pages.reserve(blocks.size() * ppb);
    for (uint32_t block : blocks) {
        for (uint32_t i = 0; i < ppb; ++i) {
            pages.push_back(block * ppb + i);
        }
    }
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    return pages;
}

uint32_t VllmSparseSelector::budget_pages() const {
    return store_.select_budget_tokens() / config_.page_tokens;
}

uint32_t VllmSparseSelector::selection_blocks() const { return store_.block_count(); }

} // namespace vllm_kvmem
