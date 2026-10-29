"""CPU regressions for logical token selection and shared-pool ownership."""
import sys
import unittest
from types import SimpleNamespace as NS
sys.path.insert(0, "src")
from kvmem_reconstruction import select_tokens, adopt_attention_blocks


class FullAttentionManager:
    def __init__(self, blocks):
        self.req_to_blocks = {"r": blocks}


class MambaManager(FullAttentionManager):
    pass


class ReconstructionTest(unittest.TestCase):
    def test_original_pages_and_partial_tail(self):
        original = list(range(35))
        self.assertEqual(select_tokens(original, [0, 4, 8], 4),
                         [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34])
        self.assertEqual(original, list(range(35)))
        for pages in ([8, 0], [0, 0], [-1], [9]):
            with self.assertRaises(ValueError):
                select_tokens(original, pages, 4)

    def test_each_rank_retires_old_view_without_dropping_archive(self):
        from kvmem_sparse_runtime import VllmSparseRuntime, _ReqState
        for rank in (0, 1):
            runtime = VllmSparseRuntime.__new__(VllmSparseRuntime)
            st = _ReqState()
            st.row = (40, 41)
            st.logical = (0, 8)
            st.row_index = 2
            st.tokens, st.seq_len = 35, 7
            runtime._state = {"r": st}
            archive = object()
            runtime._orch = {"r": archive}
            runtime._seq_override = {2: 7, 3: 11}
            runtime._owned_slots = {"r": {40, 41}}
            runtime._slot_owner = {40: "r", 41: "r", 50: "other"}
            for name in ("_prev_n", "_pending_new", "_pending_frontier", "_inflight",
                         "_engine_computed", "_engine_logical", "_page_map", "_absorb_upto"):
                setattr(runtime, name, {"r": 35})
            runtime.tp_rank = rank
            events = []
            runtime.log = events.append
            runtime.accept_reconstruction("r")
            self.assertTrue(runtime.is_rebuilt("r"))
            self.assertEqual((st.row, st.logical, st.tokens, st.seq_len), ((), (), 0, 0))
            self.assertEqual(runtime._seq_override, {3: 11})
            self.assertEqual(runtime._slot_owner, {50: "other"})
            self.assertIs(runtime._orch["r"], archive)
            self.assertEqual(events[0]["rank"], rank)

    def test_native_blocks_and_recurrent_ownership(self):
        blocks = [NS(block_id=i) for i in (10, 12, 15)]
        state = [NS(block_id=2)]
        attention = FullAttentionManager(blocks)
        mamba = MambaManager(state)
        freed = []
        manager = NS(coordinator=NS(single_type_managers=[mamba, attention]),
                     block_pool=NS(free_blocks=freed.extend,
                                   get_num_free_blocks=lambda: len(freed)))
        # Device slots differ completely from the original token page indices.
        result = adopt_attention_blocks(manager, "r", [40, 41, 42, 43, 48], 4)
        self.assertEqual(attention.req_to_blocks["r"], blocks[:2])
        self.assertEqual(mamba.req_to_blocks["r"], state)
        self.assertEqual(freed, blocks[2:])
        self.assertEqual(result["freed"], 1)
        for pages in ([41, 42], [40, 40], [80, 81]):
            with self.assertRaises(ValueError):
                adopt_attention_blocks(manager, "r", pages, 4)
        self.assertEqual(freed, blocks[2:])


if __name__ == "__main__":
    unittest.main()
