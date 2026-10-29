"""Resolve the scheduler's served/resident counters for the current worker step."""


def scheduled_position_offset(output, request_id):
    """Return the true-position minus resident-write-position delta, if published.

    V2's CPU num_computed_tokens_np is intentionally a *logical* sampler upper
    bound, not the resident KV counter. CachedRequestData carries both books.
    """
    cached = getattr(output, "scheduled_cached_reqs", None)
    if cached is not None:
        ids = getattr(cached, "req_ids", ()) or ()
        if request_id in ids:
            index = ids.index(request_id)
            computed = cached.num_computed_tokens[index]
            logical = cached.num_logical_tokens[index]
            return max(0, int(logical) - int(computed))
    for fresh in getattr(output, "scheduled_new_reqs", ()) or ():
        if getattr(fresh, "req_id", None) == request_id:
            computed = int(getattr(fresh, "num_computed_tokens", 0) or 0)
            logical = int(getattr(fresh, "num_logical_tokens", computed) or 0)
            return max(0, logical - computed)
    return None
