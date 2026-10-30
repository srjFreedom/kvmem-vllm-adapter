# H is a SHARED budget (KVMem contract): C sessions get H//C each and may borrow only the
# unassigned remainder. Over-subscription must fail loudly.
import sys

sys.path.insert(0, "src")
from kvmem_session_budget import BudgetError, SessionBudgetPool

MiB = 1024 * 1024
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def main():
    pool = SessionBudgetPool(host_bytes=4 * 1024 * MiB, sessions=4)
    g0 = pool.open(0)
    check(g0.capacity_bytes == 1024 * MiB, "each session gets H / sessions")

    pool.charge(0, 800 * MiB)
    check(pool.grant(0).used_bytes == 800 * MiB, "the charge is recorded")
    check(pool.used_bytes() == 800 * MiB, "the pool tracks total use")

    # Once all four sessions are open the whole H is assigned, so there is nothing to borrow.
    for s in range(1, 4):
        pool.open(s)
    check(pool.unassigned_bytes() == 0, "opening every session assigns the whole H")
    try:
        pool.charge(0, 600 * MiB)
        check(False, "over-subscription must raise")
    except BudgetError as exc:
        check("unassigned" in str(exc), "over-subscription reports why it failed")

    # With only two sessions open, the other half is unassigned and can be borrowed.
    pool2 = SessionBudgetPool(host_bytes=4 * 1024 * MiB, sessions=2)
    pool2.open(1)
    pool2.charge(1, 2 * 1024 * MiB + 512 * MiB)
    check(pool2.grant(1).capacity_bytes > 2 * 1024 * MiB,
          "a session may borrow the unassigned remainder")

    # Closing a session returns its share.
    before = pool2.assigned_bytes()
    pool2.close(1)
    check(pool2.assigned_bytes() < before, "closing a session frees its share")

    # A retained session keeps its H share; a different ID cannot alias or over-assign it.
    one = SessionBudgetPool(host_bytes=1024 * MiB, sessions=1)
    held = one.open(1)
    check(one.open(1) is held, "the same host session ID refers to one grant")
    try:
        one.open(2)
        check(False, "a new H share must not be handed out beyond SESSIONS")
    except BudgetError:
        check(True, "an occupied H share cannot be assigned twice")
    one.close(1)
    check(one.open(2).capacity_bytes == 1024 * MiB,
          "an evicted/closed H share becomes available to a fresh ID")

    # The pool never exceeds H.
    pool3 = SessionBudgetPool(host_bytes=1024 * MiB, sessions=4)
    for s in range(4):
        pool3.open(s)
        try:
            pool3.charge(s, 400 * MiB)
        except BudgetError:
            pass
    check(pool3.used_bytes() <= pool3.host_bytes, "total use never exceeds H")

    print("PASS session_budget_test" if not failed else "FAILED session_budget_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
