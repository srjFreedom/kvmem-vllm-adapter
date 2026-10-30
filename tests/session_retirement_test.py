"""Offline all-TP terminal checkpoint publication and failure gating."""

import sys
from types import SimpleNamespace as NS

sys.path.insert(0, "src")
from kvmem_session_retirement import (retire_finished_worker_requests,
                                       retire_terminal_sessions)
from kvmem_session_store import MetaStore


def record(req, rank):
    return {"request": req, "rank": rank, "expected_ranks": 2,
            "hashes": ["ab", "cd"], "covered": 32, "kv_block_size": 16,
            "page_tokens": 16, "pages_per_block": 1, "capacity_pages": 2,
            "has_state": True, "session_id": ["tenant", "chat"],
            "prefix_digest": "a" * 64}


class Executor:
    is_failed = False

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def collective_rpc(self, method, *, timeout, args, non_block=False):
        self.calls.append((method, timeout, non_block))
        if method == "prepare_kvmem_session_checkpoints":
            if isinstance(self.responses, Exception):
                raise self.responses
            return self.responses
        return None


def main():
    order = []
    runtime = NS(live_requests=lambda: ["finished", "paused"],
                 release=lambda req: order.append(req) or True)
    batch = NS(finished_req_ids={"finished"}, num_scheduled_tokens={"new": 4096})
    assert retire_finished_worker_requests(runtime, batch) == ["finished"]
    assert order == ["finished"], "only completed requests are retired before new admission"

    output = {0: NS(outputs=[NS(request_id="r", finish_reason=0),
                              NS(request_id="abort", finish_reason=2)])}
    store = MetaStore()
    reports = []
    good = Executor([{"r": record("r", 0)}, {"r": record("r", 1)}])
    assert retire_terminal_sessions(output, (0, 1), good, 2, store, reports.append) == ["r"]
    assert len(store) == 1
    assert reports[-1]["kind"] == "session_checkpoint_retired"
    assert good.calls == [("prepare_kvmem_session_checkpoints", 120, False)]

    for responses in ([{"r": record("r", 0)}, {"r": None}],
                      [{"r": record("r", 0)},
                       {"r": {**record("r", 1), "prefix_digest": "b" * 64}}],
                      TimeoutError("stalled TP rank")):
        store = MetaStore()
        executor = Executor(responses)
        reports = []
        assert retire_terminal_sessions(output, (0, 1), executor, 2, store,
                                        reports.append) == []
        assert len(store) == 0
        assert reports[-1]["kind"] == "session_checkpoint_retire_aborted"
        assert executor.calls[-1] == ("abort_kvmem_session_checkpoints", 5, True)

    print("session_retirement_test: OK")


if __name__ == "__main__":
    main()
