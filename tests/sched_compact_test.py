"""Pure contracts for the EngineCore-side quiescent compaction (opt-in, default off)."""
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, "src")
from kvmem_reconstruction import validate_quiescent_compaction
from kvmem_sched_compact import apply_compaction, pick_candidate, pressure_due

failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


class FullAttention:
    def __init__(self, blocks):
        self.req_to_blocks = {"r": blocks}


def main():
    check(pressure_due(8, 24), "low headroom reports pressure")
    check(not pressure_due(64, 24), "ample headroom does not")

    busy = NS(request_id="busy", num_in_flight_tokens=1, num_output_placeholders=0,
              num_computed_tokens=90000)
    tiny = NS(request_id="tiny", num_in_flight_tokens=0, num_output_placeholders=0,
              num_computed_tokens=1)
    resident = NS(request_id="r", num_in_flight_tokens=0, num_output_placeholders=0,
                  num_computed_tokens=147456, num_logical_tokens=147456)
    picked = pick_candidate([busy, tiny, resident], 24)
    check(picked is resident, "only an idle, resident request is compactable")
    check(pick_candidate([busy], 24) is None, "a request with tokens in flight is skipped")

    blocks = [NS(block_id=b) for b in range(0, 40)]
    attention = FullAttention(blocks)
    cache = NS(coordinator=NS(single_type_managers=[attention]),
               block_pool=NS(free_blocks=lambda b: None))
    pages = list(range(1152))          # a packed prefix of native blocks 0..23
    records = [{"rank": rank, "request": "r", "epoch": 3,
                "logical_frontier": 147456, "resident_frontier": 147456,
                "selected_pages": pages, "physical_pages": pages,
                "visible_tokens": 36864, "page_tokens": 32, "block_pages": 49}
               for rank in (0, 1)]
    result = validate_quiescent_compaction(resident, records, 2, 3, cache)
    adopted = {}

    def adopt(visible, physical, block_pages):
        adopted.update(visible=visible, pages=len(physical), block=block_pages)
        return {"blocks": len(physical) // block_pages}

    out = apply_compaction(resident, result, adopt)
    check(resident.num_computed_tokens == 36864, "resident book moves to the agreed visible length")
    check(resident.num_logical_tokens == 147456, "logical book is untouched")
    check(resident.num_kvmem_offset == 147456 - 36864, "offset is derived once")
    check(out["visible"] == 36864 and adopted["pages"] == 1152,
          f"the agreed blocks are adopted ({adopted})")

    resident.num_computed_tokens = resident.num_logical_tokens
    with_raise = False
    try:
        apply_compaction(resident, {**result, "visible_tokens": 147456}, adopt)
    except ValueError:
        with_raise = True
    check(with_raise, "a non-shrinking compaction is refused")

    print("PASS sched_compact_test" if not failed else "FAILED sched_compact_test")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
