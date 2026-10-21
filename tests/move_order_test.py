# Device-side page moves must never overwrite bytes that a later move still has to read.
# Moves whose destination is itself a pending source form cycles; those are NOT shuffled in
# place, they are rebuilt from the host store instead (one fp8 hop, exact bytes).
import sys
sys.path.insert(0, "src")
from kvmem_sparse_orchestrator import SparseOrchestrator

failed = False
def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond

def simulate(moves, slots=12):
    mem = list(range(100, 100 + slots))     # each slot holds its own id as "data"
    ordered, cyclic = SparseOrchestrator.order_moves(moves)
    written = set()
    for src, dst in ordered:
        check(dst not in written, f"destination {dst} is written only once")
        check(mem[src] >= 0, f"source {src} still holds data when copied")
        mem[dst] = mem[src]
        written.add(dst)
    return ordered, cyclic, mem

# Shift into free slots: everything can be done on the device.
ordered, cyclic, mem = simulate([(0, 5), (1, 6), (2, 7)])
check(len(ordered) == 3 and not cyclic, "a shift into free destinations is fully ordered")
check(mem[5] == 100 and mem[7] == 102, "values land in their destinations")

# 2-cycle: neither move is safe without a spare page -> both defer to the host rebuild.
ordered, cyclic, _ = simulate([(0, 1), (1, 0)])
check(not ordered and len(cyclic) == 2, "a 2-cycle defers both moves (no in-place shuffle)")

# 3-cycle plus an independent move: only the independent one is safe.
ordered, cyclic, _ = simulate([(0, 1), (1, 2), (2, 0), (5, 9)])
check(ordered == [(5, 9)], f"only the independent move is ordered (got {ordered})")
check(len(cyclic) == 3, "the whole 3-cycle defers")

# A chain 3->2, 2->1 where 1 is free: the tail can be moved first, then the head.
ordered, cyclic, mem = simulate([(3, 2), (2, 1)])
check(ordered == [(2, 1), (3, 2)], f"a chain is ordered tail-first (got {ordered})")
check(mem[1] == 102 and mem[2] == 103, "chain values are correct after the moves")

print("PASS move_order_test" if not failed else "FAILED move_order_test")
sys.exit(0 if not failed else 1)
