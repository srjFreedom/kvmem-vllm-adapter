# Validates the deterministic policy store: a pure function of the conversation state. The same
# committed-block set + budget + mandatory set always yields the identical decision, which is the
# lockstep guarantee that keeps every TP rank in sync (no host-pool occupancy reads).
import sys

sys.path.insert(0, "src")
from policy_store import DeterministicPolicyStore, PolicyDecision


def check(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)


def main():
    ps = DeterministicPolicyStore(block_tokens=16)
    # 8 committed blocks, positions 0..112 (16 tokens each). Budget holds 3 blocks (48 tokens).
    committed = [(i, i * 16) for i in range(8)]
    budget = 48  # 3 blocks
    mandatory = (7,)  # the newest block must stay resident

    d1 = ps.next_working_set(committed, budget, mandatory)
    d2 = ps.next_working_set(committed, budget, mandatory)
    check(d1 == d2, "determinism: same input -> identical decision")
    check(isinstance(d1, PolicyDecision), "decision type")

    # The newest block (7) is mandatory and stays resident. The budget holds 3 blocks, so the
    # three newest (5,6,7) are resident; the oldest five (0..4) are evicted by the session.
    check(d1.selected == (5, 6, 7), "selected = newest 3 blocks")
    check(d1.mandatory == (7,), "mandatory preserved")

    # A mandatory block outside the budget is still kept resident (it cannot be evicted).
    d3 = ps.next_working_set(committed, budget, (0,))
    check(0 in d3.selected, "mandatory block 0 is kept resident")
    check(7 in d3.selected, "budget blocks still resident")

    # Pure function: shuffling the input order must not change the decision.
    shuffled = list(reversed(committed))
    d4 = ps.next_working_set(shuffled, budget, mandatory)
    check(d4 == d1, "pure function: input order irrelevant")

    print("PASS policy_store_test")


if __name__ == "__main__":
    main()
