# The TP=2 lockstep test: two ranks, the SAME conversation, must make identical decisions
# (identical evict/restore sets) at every step. This is the lockstep guarantee that keeps the two
# ranks' block pools in sync without any cross-rank IPC. The policy store is a pure function of
# the conversation state (no host-pool occupancy reads), so the ranks cannot diverge.
import sys

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")
import vllm_kvmem_ext as ext
from policy_store import DeterministicPolicyStore


def check(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)


def make_fake_handle(rank, num_blocks, block_tokens):
    """A CPU-backed fake handle for one rank (the same conversation on both ranks)."""
    catalog = []
    for i in range(num_blocks):
        cb = ext.CatalogBlock()
        si = ext.SessionIdentity()
        si.id = rank + 1
        si.incarnation = 1
        cb.session = si
        cb.id = i
        cb.original_start = i * block_tokens
        cb.valid_tokens = block_tokens
        cb.content_version = 1
        cb.statistics_version = 1
        catalog.append(cb)

    class RankHandle:
        def __init__(self):
            self._catalog = catalog
            self._resident = list(range(num_blocks))
            self._stamp = ext.MemoryStamp()
            self._stamp.valid = True
            self._stamp.session = si
            self._stamp.history_revision = 1
            self._stamp.view_version = 0
            self._stamp.frontier = num_blocks * block_tokens
            self._stamp.execution_history = 1

        def geometry(self):
            return ext.Geometry(1, 2, 8, block_tokens, 2)

        def catalog(self):
            return self._catalog

        def resident(self):
            return self._resident

        def stamp(self):
            return self._stamp

        def spill_block(self, b, d, n):
            pass

        def restore_block(self, b, s, n):
            pass

        def publish_view(self, view):
            self._resident = [vb.block.id for vb in view.blocks]
            self._stamp = view.stamp

    return RankHandle()


def main():
    num_blocks = 8
    block_tokens = 16
    budget = 48  # 3 blocks

    rank0_handle = make_fake_handle(0, num_blocks, block_tokens)
    rank1_handle = make_fake_handle(1, num_blocks, block_tokens)
    rank0_backend = ext.VllmMemoryBackend(rank0_handle, 1 << 30, "vllm-rank0", False)
    rank1_backend = ext.VllmMemoryBackend(rank1_handle, 1 << 30, "vllm-rank1", False)
    rank0_session = ext.MemorySession(rank0_backend)
    rank1_session = ext.MemorySession(rank1_backend)

    policy = DeterministicPolicyStore(block_tokens)
    committed = [(i, i * block_tokens) for i in range(num_blocks)]
    mandatory = (7,)

    # Both ranks decide the same working set (the lockstep guarantee).
    d0 = policy.next_working_set(committed, budget, mandatory)
    d1 = policy.next_working_set(committed, budget, mandatory)
    check(d0 == d1, "lockstep: identical policy decision on both ranks")

    # Both ranks run the same prepare/advance/publish and end with the same resident set.
    for session, backend in ((rank0_session, rank0_backend), (rank1_session, rank1_backend)):
        st = backend.stamp()
        session.prepare(st, list(d0.selected), list(d0.mandatory),
                         ext.MemoryBudget(budget, block_tokens, num_blocks * block_tokens,
                                          num_blocks * 2048))
        phase = session.advance()
        while phase in (ext.MemoryPhase.Spilling, ext.MemoryPhase.Restoring):
            phase = session.advance()
        check(phase == ext.MemoryPhase.Ready, "rank ready")
        session.publish()

    check(rank0_handle.resident() == rank1_handle.resident(),
          "lockstep: identical resident sets on both ranks")
    check(rank0_handle.resident() == list(d0.selected), "resident = selected working set")

    print("PASS tp2_lockstep_test")


if __name__ == "__main__":
    main()
