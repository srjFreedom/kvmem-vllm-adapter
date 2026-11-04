"""All-TP publication barrier for an opt-in, exact-frontier session checkpoint."""

from kvmem_session_store import (terminal_checkpoint_requests,
                                 verify_tp_checkpoint_records)


def retire_finished_worker_requests(runtime, scheduler_output):
    """Return old H shares before admission opens a host store for a new request."""
    finished = set(getattr(scheduler_output, "finished_req_ids", ()) or ())
    released = []
    for req in runtime.live_requests():
        if req in finished:
            # ARRIVAL evidence for the false-finished case (design 28): a
            # request retired while the SAME scheduler_output still schedules
            # it (or while its prefill is behind the prompt) is Core marking
            # completion that is not - the exact event that de-registers the
            # runtime ledger mid-flight and freezes statistics at the knife.
            sched = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
            st = runtime._state.get(req)
            plen = int(getattr(runtime, "_prompt_len", {}).get(req, 0) or 0)
            tok = int(getattr(st, "tokens", -1)) if st else -1
            if runtime.log is not None:
                runtime.log({"kind": "d_retire_arrival", "req": str(req)[-8:],
                             "still_scheduled": bool(req in sched),
                             "scheduled_n": int(sched.get(req, -1) or -1),
                             "tokens": tok, "prompt_len": plen,
                             "unfinished": bool(plen and tok < plen)})
            if runtime.release(req):
                released.append(req)
    return released


def retire_terminal_sessions(output_map, allowed_reasons, executor, tp_size, store,
                             report=lambda event: None):
    request_ids = terminal_checkpoint_requests(output_map, allowed_reasons)
    if not request_ids or executor is None or getattr(executor, "is_failed", False):
        return []
    try:
        prepared = executor.collective_rpc(
            "prepare_kvmem_session_checkpoints", timeout=120, args=(request_ids,))
        agreed = {}
        for req in request_ids:
            records = ([rank.get(req) for rank in prepared]
                       if isinstance(prepared, (tuple, list)) and
                       all(isinstance(rank, dict) for rank in prepared) else [])
            agreed[req] = verify_tp_checkpoint_records(records, tp_size, req)
        # The scheduler has not returned this terminal output yet: no new request can look up a
        # half-published checkpoint. Validate every rank/request before touching shared metadata.
        for req, record in agreed.items():
            store.park(
                [bytes.fromhex(h) for h in record["hashes"]],
                covered_tokens=int(record["covered"]),
                page_tokens=int(record["page_tokens"]),
                pages_per_block=int(record["pages_per_block"]),
                capacity_pages=int(record["capacity_pages"]),
                has_state=bool(record["has_state"]),
                prefix_digest=str(record["prefix_digest"]),
                owner_session_id=tuple(record["session_id"]),
                source_request=str(req))
            report({"kind": "session_checkpoint_retired", "req": req,
                    "frontier": int(record["covered"]),
                    "ranks": list(range(tp_size)), "blocks": len(record["hashes"])})
        return request_ids
    except Exception as exc:
        store.discard_requests(request_ids)
        try:
            # The first RPC may have timed out on a worker. Queue cleanup behind it, but never
            # wait indefinitely for that rank before returning the terminal HTTP response.
            executor.collective_rpc(
                "abort_kvmem_session_checkpoints", timeout=5,
                args=(request_ids,), non_block=True)
        except Exception:
            pass
        report({"kind": "session_checkpoint_retire_aborted",
                "requests": request_ids, "error": f"{type(exc).__name__}: {exc}"})
        return []
