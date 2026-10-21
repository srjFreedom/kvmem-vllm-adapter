// Validates the sparse-selection layer (KVMem "A mode" plumbing) on CPU: 200K tokens of history
// must select roughly B tokens of pages, always keep the sink prefix and the recent suffix, and
// rank the middle by the supplied scores.
#include "vllm_kvmem/vllm_sparse_selector.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char *message) {
    std::printf("%s %s\n", condition ? "ok  " : "FAIL", message);
    if (!condition) { ++failures; }
}

} // namespace

int main() {
    using namespace vllm_kvmem;

    VllmSparseSelector::Config config;
    config.select_block_tokens = 128;
    config.budget_tokens = 32768;
    config.gen_budget_tokens = 16384;
    config.sink_tokens = 8192;
    config.recent_tokens = 1024;
    config.page_tokens = 16;
    config.lrfu = false;   // deterministic: scores decide, no history weighting
    config.method = kvmem::KvMemMethod::Retrieval;

    VllmSparseSelector selector(config);
    const uint32_t context_tokens = 200 * 1024;
    selector.set_context_tokens(context_tokens);

    check(selector.selection_blocks() == context_tokens / config.select_block_tokens,
          "200K tokens become selection blocks of 128");
    check(selector.pages_per_selection_block() == 8, "one selection block is eight 16-token pages");

    // Score the middle blocks: make block 300 the most relevant, then decreasing downward.
    const uint32_t blocks = selector.selection_blocks();
    std::vector<double> scores(blocks, 0.0);
    scores[300] = 100.0;
    scores[400] = 50.0;
    scores[500] = 25.0;
    selector.set_block_scores(scores);

    const auto pages = selector.select_pages({});
    const uint32_t budget_pages = selector.budget_pages();

    check(!pages.empty(), "the selector returns a working set");
    check(pages.size() <= budget_pages, "the working set fits the B budget in pages");
    // B=32768 tokens = 256 selection blocks of 128; allow the trailing partial block.
    check(pages.size() + 8 >= budget_pages, "the working set uses most of the budget");

    const auto contains_block = [&](uint32_t block) {
        return std::find(pages.begin(), pages.end(), block * 8) != pages.end();
    };
    check(contains_block(0), "the sink prefix is always kept");
    check(contains_block(blocks - 1), "the recent suffix is always kept");
    check(contains_block(300), "the highest-scoring middle block is selected");
    check(contains_block(400) && contains_block(500),
          "the next scoring middle blocks are selected too");

    // A zero-score middle block far from the sink should lose to the scored ones.
    check(!contains_block(1000), "an unscored middle block is dropped");

    // Shrinking the context must drop the trailing blocks from the selection space.
    selector.set_context_tokens(64 * 1024);
    const auto small = selector.select_pages({});
    check(small.size() <= 64 * 1024 / config.page_tokens,
          "a shortened context selects no more than it holds");

    std::printf(failures == 0 ? "PASS sparse_selector_test\n" : "FAILED sparse_selector_test\n");
    return failures == 0 ? 0 : 1;
}
