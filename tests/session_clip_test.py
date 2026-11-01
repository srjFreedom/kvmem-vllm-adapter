"""Round-11 regression: session checkpoint chunk alignment under speculation.

Failure being defended (measured on GPU, spec=4 + reuse=1, 2026-11-01): the scheduler's
checkpoint-clip branch carried a `self.num_spec_tokens == 0` term, so under MTP the prefill
chunk ladder never stopped at the planned frontier. The executed GPU book therefore never
equaled the frontier, the boundary capture never fired on any rank, and the terminal park
answered `no sealed executed-frontier checkpoint` -> TP barrier abort -> turn2 silently
re-prefilled the whole conversation. The clip is now unconditional under speculation, which
is safe because the boundary gate keeps EVERY prefill step draft-free (`logical <= prompt`
purges the draft pool) - aligning a chunk is exactly the spec0 operation.

Contracts checked here (all offline):
  1. clip_session_checkpoint_chunk full behavior table (cross / no-cross / past frontier /
     no-plan / boundary equalities).
  2. verify_tp_checkpoint_records fail-closed semantics (False/None/mixed -> refuse, full
     dict agreement -> the record).
  3. The scheduler call-site must NOT gate the clip on the spec token count (source guard;
     the GPU run owns the end-to-end proof).
"""
import inspect
import re
import sys

sys.path.insert(0, "src")
from kvmem_session_store import (clip_session_checkpoint_chunk,
                                 verify_tp_checkpoint_records)

failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


# frontier for these parameters: lcm(block=1616, select=64)=6464? -> use the function itself
# to name the target, then test around it. prompt 88000, block 16, budget 36864, select 64
# mirrors the machine config (kv_block 16 tokens here; use a synthetic-but-parallel shape).
BLOCK, BUDGET, SELECT = 32, 4096, 64
from kvmem_session_store import session_checkpoint_frontier
TARGET = session_checkpoint_frontier(2000, BLOCK, BUDGET, SELECT)
check(TARGET == 1984, f"planned frontier 1984 = min(prompt//q*q, (budget-block)//q*q) (got {TARGET})")

# 1) behavior table
check(clip_session_checkpoint_chunk(TARGET - 512, 1024, 2000, BLOCK, BUDGET, SELECT) == 512,
      "chunk crossing the frontier is clipped to land exactly on it")
check(clip_session_checkpoint_chunk(TARGET, 512, 2000, BLOCK, BUDGET, SELECT) == 512,
      "a step already AT the frontier is not clipped (decode/next chunk unaffected)")
check(clip_session_checkpoint_chunk(TARGET + 640, 512, 2000, BLOCK, BUDGET, SELECT) == 512,
      "a step past the frontier is untouched")
check(clip_session_checkpoint_chunk(0, TARGET // 2, 2000, BLOCK, BUDGET, SELECT) == TARGET // 2,
      "a chunk that ends before the frontier passes through")
check(clip_session_checkpoint_chunk(TARGET - 512, 512, 16, BLOCK, BUDGET, SELECT) == 512,
      "a prompt not above one block has no plan (frontier 0) -> passthrough")
check(clip_session_checkpoint_chunk(TARGET - 512, 512, 2000, BLOCK, BUDGET, SELECT) == 512,
      "a chunk ending exactly on the frontier is not clipped (strict <..> bounds)")

# 2) TP record verification (the barrier that aborted the park)
def refuses(records, tp, req):
    try:
        verify_tp_checkpoint_records(records, tp, req)
        return False
    except ValueError:
        return True

# geometry per verify_tp_checkpoint_records: block=page*ppb, covered=len(hashes)*block,
# capacity*page==covered, capacity%ppb==0, digest 64 hex chars.
def _rec(rank):
    return {"hashes": [f"h{i}" for i in range(120)], "covered": 1920,
            "prefix_digest": "0" * 64, "kv_block_size": 16,
            "page_tokens": 16, "pages_per_block": 1, "capacity_pages": 120,
            "has_state": True, "session_id": ["", "mtp4-synth"],
            "expected_ranks": 2, "request": "req", "rank": rank}
GOOD = [_rec(0), _rec(1)]
check(verify_tp_checkpoint_records(GOOD, 2, "req") is not None,
      "two agreeing full records across TP2 publish")
check(refuses([{**GOOD[0]}, False], 2, "req"),
      "a rank answering False (nothing sealed) refuses publication")
check(refuses([{**GOOD[0]}, None], 2, "req"),
      "a rank returning None (error path) refuses publication")
check(refuses([dict(GOOD[0])], 2, "req"),
      "a short record list (rank never answered) refuses publication")

# 3) source guard on the scheduler call site
try:
    import vllm  # noqa: F401  (engine importable only inside the venv)
    src = inspect.getsource(vllm.v1.core.sched.scheduler)
except Exception:
    src = open("engines/vllm/vllm/v1/core/sched/scheduler.py", encoding="utf-8").read()
region = src[src.index("clip_session_checkpoint_chunk") - 900: src.index("clip_session_checkpoint_chunk")]
code = "\n".join(ln for ln in region.splitlines() if not ln.strip().startswith("#"))
check("num_spec_tokens == 0" not in code,
      "the clip branch is unconditional under speculation (no num_spec_tokens==0 code term)")
check("kvmem_session_reuse" in code and "request.session_id" in code,
      "the clip still requires session reuse + a native session id")

print("FAIL session_clip_test" if failed else "PASS session_clip_test")
sys.exit(1 if failed else 0)
