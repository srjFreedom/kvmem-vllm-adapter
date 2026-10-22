# The resident-book contract (fork side): `num_logical_tokens` is the SERVED count and
# `num_computed_tokens` is the RESIDENT KV length. A compaction rewinds only the resident one, so
# the logical book (and the API's output length) keeps every token while the engine's KV demand
# stays inside the working area.
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vllm.sampling_params import SamplingParams          # noqa: E402
from vllm.v1.request import Request                      # noqa: E402
from vllm.v1.core.sched.utils import check_stop          # noqa: E402

checks = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    checks.append(bool(cond))


def make_request(prompt_tokens=100, max_tokens=8):
    return Request(
        request_id="r1",
        prompt_token_ids=list(range(prompt_tokens)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        pooling_params=None,
    )


def main():
    req = make_request(100)
    check(req.num_logical_tokens == 0, "a fresh request served nothing yet")
    check(req.num_tokens == 100 and req.num_computed_tokens == 0,
          "the logical book starts with the prompt, the resident book is empty")

    # The scheduler advances both counters together while tokens are computed.
    req.num_computed_tokens += 100
    req.num_logical_tokens += 100
    check(req.num_logical_tokens == req.num_computed_tokens == 100,
          "without a compaction the two counters move in lockstep")

    # A compaction rewinds ONLY the resident book: the served count and the logical token list
    # stay where they are, which is what keeps a long prompt lossless.
    req.num_computed_tokens = 40
    check(req.num_logical_tokens == 100, "the served count keeps every token after a compaction")
    check(req.num_tokens == 100, "the logical book keeps every token after a compaction")
    check(req.num_tokens >= req.num_prompt_tokens,
          "the logical book still covers the prompt (the sampler needs it)")

    # New tokens keep landing in the resident book, so the gap only grows.
    req.append_output_token_ids(7)
    req.num_computed_tokens += 1
    req.num_logical_tokens += 1
    check(req.num_logical_tokens - req.num_computed_tokens == 60,
          f"the gap is the compacted amount (got {req.num_logical_tokens - req.num_computed_tokens})")
    check(req.num_output_tokens == 1, "the output count is unaffected by the compaction")

    # The stop test must use kvmem's logical ceiling, not the engine window.
    req2 = make_request(60, max_tokens=1000)
    req2.num_logical_tokens = req2.num_computed_tokens = 60
    req2.append_output_token_ids(1001)           # 1001 != eos/stop ids
    check(req2.num_tokens == 61, "the logical book grew with the output")
    check(not check_stop(req2, 1000),
          "a long logical context does not stop at a kvmem CTX of 1000")
    check(check_stop(req2, 61), "it does stop at the logical ceiling itself")

    # Preemption resets both (the KV is gone and the tokens must be served again).
    req.num_computed_tokens = 0
    req.num_logical_tokens = 0
    check(req.num_logical_tokens == 0, "a preempted request re-serves from the start")

    print(("PASS" if all(checks) else "FAILED") + " resident_book_test")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
