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


def plan_attention_blocks(kv_cache_manager, request_id, physical_pages, block_pages):
    """Validate a packed row against this request's blocks without changing ownership."""
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
    return manager, [by_id[b] for b in blocks], freed


def adopt_attention_blocks(kv_cache_manager, request_id, physical_pages, block_pages):
    """Keep packed attention blocks only; recurrent groups retain their own state slots."""
    manager, kept, freed = plan_attention_blocks(
        kv_cache_manager, request_id, physical_pages, block_pages)
    manager.req_to_blocks[request_id] = kept
    if freed:
        kv_cache_manager.block_pool.free_blocks(freed)
    return {"blocks": len(kept), "freed": len(freed),
            "free_blocks": kv_cache_manager.block_pool.get_num_free_blocks()}


def validate_quiescent_compaction(request, records, tp_size, epoch, kv_cache_manager,
                                  d_mode=False):
    """Validate TP proposals on Core's idle boundary without touching GPU or allocator.

    Acceptance is not a commit: staged GPU moves and all-rank acknowledgment must follow before
    EngineCore changes ownership or its resident book.
    """
    if (int(getattr(request, "num_in_flight_tokens", -1)) != 0
            or int(getattr(request, "num_output_placeholders", 0)) != 0):
        raise ValueError("compaction requires a quiescent draft-free request")
    if not isinstance(records, (tuple, list)) or len(records) != int(tp_size) or tp_size <= 0:
        raise ValueError("compaction needs one proposal from every TP rank")
    try:
        req_id = request.request_id
        logical = int(request.num_logical_tokens)
        resident = int(request.num_computed_tokens)
        if resident <= 0 or logical < resident:
            raise ValueError("EngineCore served/resident books disagree")
        # `num_kvmem_offset` is derived at apply time, not required to match before the rewrite.
        ranks = set()
        proposal = None
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("missing TP compaction proposal")
            rank = int(record["rank"])
            ranks.add(rank)
            page = int(record["page_tokens"])
            block = int(record["block_pages"])
            selected = tuple(int(p) for p in record["selected_pages"])
            physical = tuple(int(p) for p in record["physical_pages"])
            visible = int(record["visible_tokens"])
            if (record["request"] != req_id or int(record["epoch"]) != int(epoch)
                    or int(record["logical_frontier"]) != logical
                    or (not d_mode
                        and int(record["resident_frontier"]) != resident)):
                # Plan D: the worker's resident book is the VIEW length and
                # Core's computed book is the TRUE length - the equality this
                # check enforces has no D meaning (the entry-side twin is
                # exempted the same way; the logical/frontier equality above,
                # stamped from the sender, is the surviving time-agreement
                # guarantee).
                raise ValueError("TP compaction frontier/epoch differs from EngineCore")
            # Partial tail pages are legal as long as the selection KEEPS the last page (the
            # recent window always does): `visible` is then the sum of per-page token counts,
            # not a multiple of the page. Requiring page-aligned logical/resident here refused
            # every prompt-end reselect for a prompt whose length is not page-multiple
            # (measured: 169,998 -> 52 refusals while the worker had already ranked the needle
            # block #1 and both halves into the selection).
            expected_visible = sum(min(page, max(0, logical - p * page)) for p in selected)
            if (page <= 0 or block <= 0 or not selected
                    or selected[0] < 0 or selected[-1] * page >= logical
                    or selected != tuple(sorted(set(selected)))
                    or (logical - 1) // page not in selected
                    # Logical pages and physical slots are distinct namespaces: only require
                    # that the adopted row covers at least the selected pages.
                    or len(physical) < len(selected)
                    or visible != expected_visible
                    or (not d_mode and visible >= resident)):
                # D: `resident` is Core's TRUE book; view-based visible is
                # smaller by construction only while the window holds room.
                # The shrinking law is enforced worker-side instead (the
                # knife must free a real margin before proposing).
                raise ValueError("invalid or non-shrinking TP compaction proposal")
            candidate = (selected, physical, visible, page, block)
            if proposal is not None and candidate != proposal:
                raise ValueError("TP ranks proposed different working sets or physical rows")
            proposal = candidate
        if ranks != set(range(int(tp_size))):
            raise ValueError("duplicate or missing TP rank in compaction proposal")
        if proposal is None:
            raise ValueError("missing TP compaction proposal")
        plan_attention_blocks(kv_cache_manager, req_id, proposal[1], proposal[4])
        return {"request": req_id, "epoch": int(epoch), "logical_frontier": logical,
                "resident_frontier": resident, "selected_pages": proposal[0],
                "physical_pages": proposal[1], "visible_tokens": proposal[2],
                "page_tokens": proposal[3], "block_pages": proposal[4]}
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("incomplete TP compaction proposal") from exc
