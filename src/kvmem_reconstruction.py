"""Pure contracts for the experimental full-working-set reconstruction path."""


def select_tokens(prompt_token_ids, logical_pages, page_tokens):
    """Select original prompt tokens in page order, including a partial final page."""
    ids = list(prompt_token_ids or ())
    pages = [int(p) for p in logical_pages]
    width = int(page_tokens)
    if width <= 0 or not pages or pages != sorted(set(pages)):
        raise ValueError("reconstruction needs ascending unique logical pages")
    if any(p < 0 or p * width >= len(ids) for p in pages):
        raise ValueError("reconstruction page lies outside the original prompt")
    return [token for p in pages for token in ids[p * width:(p + 1) * width]]


def adopt_attention_blocks(kv_cache_manager, request_id, physical_pages, block_pages):
    """Keep packed attention blocks only; recurrent groups retain their own state slots.

    The worker packs into its existing native blocks. A partial last block stays allocated;
    all page IDs must describe the same contiguous expansion the engine will publish later.
    Validate before freeing anything, so a bad row cannot partially mutate the allocator.
    """
    pages = [int(p) for p in physical_pages]
    width = int(block_pages)
    if not pages or width <= 0 or len(pages) != len(set(pages)):
        raise ValueError("invalid physical page row")
    blocks = list(dict.fromkeys(p // width for p in pages))
    expanded = [b * width + offset for b in blocks for offset in range(width)]
    if pages != expanded[:len(pages)]:
        raise ValueError("compaction row must pack complete native blocks before its tail")
    managers = [m for m in kv_cache_manager.coordinator.single_type_managers
                if "FullAttention" in type(m).__name__]
    if len(managers) != 1:
        raise ValueError("compaction currently requires one full-attention group")
    manager = managers[0]
    old = list(manager.req_to_blocks.get(request_id, ()))
    by_id = {int(b.block_id): b for b in old}
    if any(b not in by_id for b in blocks):
        raise ValueError("compaction cannot adopt blocks owned by another request")
    keep = set(blocks)
    freed = [b for b in old if int(b.block_id) not in keep]
    manager.req_to_blocks[request_id] = [by_id[b] for b in blocks]
    if freed:
        kv_cache_manager.block_pool.free_blocks(freed)
    return {"blocks": len(blocks), "freed": len(freed),
            "free_blocks": kv_cache_manager.block_pool.get_num_free_blocks()}
