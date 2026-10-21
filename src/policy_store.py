# The deterministic policy store: decides the next working set (which blocks stay resident on
# the GPU, which are evicted to host, which are restored) as a PURE FUNCTION of the conversation
# state. It never reads the host-pool occupancy, so every TP rank — given the same conversation —
# makes the identical decision (lockstep). The eviction order is the block's original_start
# (position in the conversation): the oldest blocks are evicted first, the newest stay resident.
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PolicyDecision:
    """The policy's output: the working set (blocks to keep resident) and the mandatory set.

    The evict/restore sets are decided by the kvmem session (the core), not the policy: the
    session spills the committed blocks beyond the budget and restores the mandatory blocks that
    are not currently resident. The policy only decides which blocks to keep resident.
    """
    selected: tuple[int, ...]    # blocks to keep resident (the working set)
    mandatory: tuple[int, ...]   # blocks that cannot be evicted (the active frontier)


class DeterministicPolicyStore:
    """A pure function of the conversation state. No host-pool occupancy reads.

    The conversation state is the set of (block_id, original_start, content_version) tuples the
    engine has committed for the active request. The policy keeps the newest `resident_budget`
    blocks resident and evicts the oldest; it restores the blocks the engine marked mandatory.
    """

    def __init__(self, block_tokens: int):
        self._block_tokens = block_tokens

    def next_working_set(self, committed_blocks, resident_budget_tokens, mandatory_block_ids):
        # committed_blocks: an iterable of (block_id, original_start) for the active request.
        # resident_budget_tokens: how many tokens the GPU KV cache can hold (the budget).
        # mandatory_block_ids: the blocks the engine requires to be resident (the frontier).
        blocks = sorted(committed_blocks, key=lambda b: b[1])  # oldest first by position
        resident_budget_blocks = max(1, resident_budget_tokens // self._block_tokens)
        mandatory = set(mandatory_block_ids)

        # Keep the newest resident_budget_blocks resident; the mandatory blocks always stay.
        # The evict/restore sets are decided by the kvmem session, not the policy.
        selected = []
        for block_id, _ in reversed(blocks):  # newest first
            if len(selected) < resident_budget_blocks or block_id in mandatory:
                selected.append(block_id)
        selected.reverse()  # oldest-first order for the view

        return PolicyDecision(
            selected=tuple(selected),
            mandatory=tuple(mandatory),
        )
