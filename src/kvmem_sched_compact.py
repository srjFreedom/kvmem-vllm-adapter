"""EngineCore-side quiescent compaction: decide, validate, apply.

The worker-originated compaction moves pages while the next step is already scheduled with the old
books, so the step after the compaction writes at a stale offset (measured: row 1,274 pages,
40,768 tokens, while the next chunk needed 40,960). This module is the Core half of the fix: on the
idle boundary after a step's output (and before the next schedule), call every TP worker, require
identical proposals, then adopt blocks and rewrite EngineCore's bounds exactly once.

Nothing here runs by default; `KVMEM_SCHED_COMPACT=1` opts in.
"""


def pressure_due(free_blocks, reserve_blocks):
    """True when the attention pool has less headroom than the configured reserve."""
    return int(free_blocks) < max(1, int(reserve_blocks))


def pick_candidate(running, reserve_blocks):
    """The resident request that may be compacted at this boundary, or None."""
    best = None
    best_tokens = 0
    for request in list(running or ()):
        if int(getattr(request, "num_in_flight_tokens", 0) or 0) != 0:
            continue
        if int(getattr(request, "num_output_placeholders", 0) or 0) != 0:
            continue
        if (int(getattr(request, "num_prompt_tokens", 0) or 0) > 0
                and int(getattr(request, "num_logical_tokens", 0) or 0)
                >= int(getattr(request, "num_prompt_tokens", 0) or 0)):
            # THE PROMPT IS COMPLETE: pressure compaction is forbidden here, the FINAL
            # reselect / rebuild owns this request now (goal-7833cb61 round-10).
            # Measured: (c) firing on the last chunks left the logical book rewound
            # (tokens 86016 < prompt 86670) with a fragmented resident row; the next
            # chunk need (prompt+1-resident = 4751) then exceeded the 4096 window and
            # the FINAL chunk could NEVER be scheduled - the request starved 654 tokens
            # short of its own prompt end, and finalize rebuilt over a truncated ask.
            continue
        tokens = int(getattr(request, "num_computed_tokens", 0) or 0)
        if tokens <= 1:
            continue
        if tokens > best_tokens:
            best, best_tokens = request, tokens
    return best


def apply_compaction(request, result, adopt_attention_blocks):
    """Adopt the agreed blocks and rewrite the two bounds exactly once.

    `result` is the validated proposal from `validate_quiescent_compaction`. The logical book and
    `_all_token_ids` are never touched: only the resident KV count and the derived offset move.
    """
    visible = int(result["visible_tokens"])
    logical = int(request.num_logical_tokens)
    if visible <= 0 or visible >= logical:
        raise ValueError("compaction must shorten the resident book")
    adopted = adopt_attention_blocks(visible, result["physical_pages"], result["block_pages"])
    request.num_computed_tokens = visible
    request.num_kvmem_offset = logical - visible
    return {"request": result["request"], "visible": visible,
            "offset": int(request.num_kvmem_offset), "adopted": adopted}
