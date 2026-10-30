"""CPU contract: V2's sampler mirror is not the resident KV book."""
import sys
import unittest
from types import SimpleNamespace as NS

sys.path.insert(0, "src")
from kvmem_position_books import scheduled_position_offset


class PositionBooksTest(unittest.TestCase):
    def test_reselect_uses_engine_books_not_sampler_mirror(self):
        cached = NS(req_ids=["other", "long"], num_computed_tokens=[12, 36864],
                    num_logical_tokens=[12, 102400])
        output = NS(scheduled_cached_reqs=cached, scheduled_new_reqs=[])
        self.assertEqual(scheduled_position_offset(output, "long"), 65536)
        self.assertEqual(scheduled_position_offset(output, "other"), 0)
        self.assertIsNone(scheduled_position_offset(output, "missing"))

    def test_new_and_rebuilt_requests_start_at_zero(self):
        output = NS(scheduled_cached_reqs=None,
                    scheduled_new_reqs=[NS(req_id="fresh", num_computed_tokens=0,
                                           num_logical_tokens=0)])
        self.assertEqual(scheduled_position_offset(output, "fresh"), 0)
        self.assertIsNone(scheduled_position_offset(output, "missing"))

    def test_compacted_prefill_refuses_row_shorter_than_scheduled_step(self):
        import torch
        from kvmem_sparse_runtime import (SparseViewCapacityError, VllmSparseRuntime,
                                           _ReqState)

        rt = VllmSparseRuntime.__new__(VllmSparseRuntime)
        st = _ReqState()
        st.row_index = 0
        st.tokens = 145088      # logical history
        st.seq_len = 36864      # resident after the first pressure compaction
        st.row = tuple(range(101, 101 + 1176))
        table = torch.arange(101, 101 + 1274, dtype=torch.int32).reshape(1, 1274)
        seq = torch.zeros(1, dtype=torch.int32)
        rt.view = NS(block_table_rows=lambda: [table], seq_lens=lambda: seq,
                     positions=lambda: torch.zeros(4096, dtype=torch.int64))
        rt._orch = {"r": NS()}
        rt._state = {"r": st}
        rt._engine_computed = {"r": 36864}
        rt._seq_override = {}
        rt.page_tokens = 32
        rt.inrequest_compaction = True
        rt.install = True
        rt.install_row = False
        rt.install_seq = True
        rt.install_pos = False
        rt.log = lambda record: None
        rt.sync_row_from_engine("r", 1274, n_tokens=4096)
        self.assertEqual(st.seq_len, 40768)
        with self.assertRaisesRegex(SparseViewCapacityError, "44864.*40768"):
            rt.before_step("r", row_index=0, position_slots=range(4096))
        self.assertEqual(rt._seq_override, {})
        self.assertEqual(int(seq[0]), 0)
        with self.assertRaises(SparseViewCapacityError):
            rt.before_step("r", row_index=0, position_slots=range(4096), rows_only=True)

        # A completed, resident 36,864-token prefix plus a 4,096-token chunk needs exactly
        # 1,280 pages; the same check must accept that correctly allocated row.
        st.seq_len = 36864
        st.row = tuple(range(101, 101 + 1280))
        rt.before_step("r", row_index=0, position_slots=range(4096))
        self.assertEqual(int(seq[0]), 40960)
        self.assertEqual(rt._seq_override, {0: 40960})


if __name__ == "__main__":
    unittest.main()
