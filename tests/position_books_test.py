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
        st.applied_view_coord = True   # the compaction rewound the coordinates
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
        self.assertEqual(st.seq_len, 36864,
                         "row mirroring must not count this step's query a second time")
        with self.assertRaisesRegex(SparseViewCapacityError, "40960.*40768"):
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


class UncompactedKeepsEngineBooks(unittest.TestCase):
    """The measured spec4 regression (2026-11-01): a NON-compacted request must not get a
    view seq/positions override at all. Under speculation `step_end = st.seq_len + rows`
    double-counts the in-flight draft rows and the override shrank the attention book below
    the engine's real rows - greedy MTP then diverged from spec0 from the first draft step
    (sha f700b1ee vs c2f36446; the no-view builds stayed byte-equal). The row itself must
    still be published."""
    def test_no_override_without_applied_coord(self):
        import torch
        from kvmem_sparse_runtime import VllmSparseRuntime, _ReqState
        rt = VllmSparseRuntime.__new__(VllmSparseRuntime)
        st = _ReqState()
        st.row_index = 0
        st.tokens = 4000
        st.seq_len = 4000      # resident == true: nothing was ever compacted
        st.row = tuple(range(101, 101 + 128))
        table = torch.zeros(1, 128, dtype=torch.int32)
        seq = torch.zeros(1, dtype=torch.int32)
        rt.view = NS(block_table_rows=lambda: [table], seq_lens=lambda: seq,
                     positions=lambda: torch.zeros(64, dtype=torch.int64))
        rt._orch = {"r": NS()}
        rt._state = {"r": st}
        rt._engine_computed = {"r": 4000}
        rt._seq_override = {}
        rt.page_tokens = 32
        rt.inrequest_compaction = True
        rt.install = True
        rt.install_row = True
        rt.install_seq = True
        rt.install_pos = True
        rt.sched_compact = False
        rt.log = lambda record: None
        rt._absorb_upto = {"r": 0}
        rt.before_step("r", row_index=0, position_slots=range(64))
        self.assertEqual(rt._seq_override, {},
                         "the engine book must not be overridden for an un-compacted request")
        self.assertEqual(int(seq[0]), 0, "seq_lens must keep the engine value")
        self.assertTrue(bool(table[0, :128].ne(0).any()),
                        "the block-table row is still published")


class PositionsStayTrueAfterCompaction(unittest.TestCase):
    def _runtime(self, tokens, seq_len):
        import torch
        from kvmem_sparse_runtime import VllmSparseRuntime, _ReqState
        rt = VllmSparseRuntime.__new__(VllmSparseRuntime)
        st = _ReqState()
        st.row_index = 0
        st.tokens, st.seq_len = tokens, seq_len
        st.row = tuple(range(101, 101 + 1176))
        seq = torch.zeros(1, dtype=torch.int32)
        pos = torch.arange(65536, dtype=torch.int64)
        rt.view = NS(block_table_rows=lambda: [torch.zeros(1, 4096, dtype=torch.int32)],
                     seq_lens=lambda: seq, positions=lambda: pos)
        rt._state = {"r": st}
        rt._orch = {"r": NS()}
        rt._engine_computed = {}
        rt._seq_override = {}
        rt._absorb_upto = {}
        rt._page_map = {"r": {}}
        rt.page_tokens = 32
        rt.inrequest_compaction = True
        rt.install = True
        rt.install_row = True
        rt.install_seq = True
        rt.install_pos = True
        rt.log = lambda record: None
        return rt, st, pos

    def test_compacted_view_keeps_true_positions(self):
        rt, st, pos = self._runtime(tokens=147456, seq_len=36864)
        before = pos.clone()
        rt.before_step("r", row_index=0, position_slots=range(4, 8))
        self.assertTrue(bool(rt._positions_must_stay_true))
        self.assertTrue(bool(pos[4:8].equal(before[4:8])),
                        "positions must stay TRUE so prepare_attn subtracts the offset once")

    def test_dense_prefix_still_installs_compacted_positions(self):
        rt, st, pos = self._runtime(tokens=0, seq_len=0)
        rt.before_step("r", row_index=0, position_slots=range(0, 4))
        self.assertFalse(bool(rt._positions_must_stay_true))
        self.assertEqual([int(v) for v in pos[:4]], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
