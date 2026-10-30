"""CPU regressions for logical token selection and shared-pool ownership."""
import sys
import unittest
from types import SimpleNamespace as NS
sys.path.insert(0, "src")
from kvmem_reconstruction import (select_tokens, adopt_attention_blocks,
                                  plan_attention_blocks, validate_quiescent_compaction)


class FullAttentionManager:
    def __init__(self, blocks):
        self.req_to_blocks = {"r": blocks}


class MambaManager(FullAttentionManager):
    pass


class ReconstructionTest(unittest.TestCase):
    def test_initial_prefill_pressure_tracks_physical_ceiling_not_b(self):
        from kvmem_sparse_runtime import pool_prefill_high_watermark, prefill_pressure_due
        pages = pool_prefill_high_watermark(153664, 16)
        self.assertEqual(pages * 16, 147520)
        self.assertGreater(pages * 16, 36864)  # B does not trigger early compaction
        self.assertLess(pages * 16, 153664)      # still leaves an allocator safety margin
        # Live failure at a 161,832-token pool: first compact at 145,088; a second at 149,184
        # was only one 4K chunk later and faulted. A new pressure point is eligible after R=16K.
        self.assertTrue(prefill_pressure_due(145088, 4704, 4610, 0, 16384))
        self.assertFalse(prefill_pressure_due(149184, 1700, 1664, 145088, 16384))
        self.assertTrue(prefill_pressure_due(161472, 1700, 1664, 145088, 16384))

    def test_rebuild_requires_an_actual_prefill_compaction(self):
        from kvmem_sparse_runtime import VllmSparseRuntime, _ReqState
        runtime = VllmSparseRuntime.__new__(VllmSparseRuntime)
        st = _ReqState()
        runtime.state = lambda req: st
        runtime.compact_rebuild = True
        runtime._rebuilt_req = set()
        # A prompt longer than B may still fit the physical KV pool unchanged.
        st.tokens = 50000
        self.assertFalse(runtime.needs_reconstruction("r"))
        st.compacted_in_prefill = True
        self.assertTrue(runtime.needs_reconstruction("r"))
        st.rebuilt = True
        self.assertFalse(runtime.needs_reconstruction("r"))

    def test_recurrent_session_park_fails_closed_without_exact_frontier(self):
        from kvmem_sparse_runtime import VllmSparseRuntime
        runtime = VllmSparseRuntime.__new__(VllmSparseRuntime)
        runtime.session_reuse = True
        runtime.session_store = object()
        runtime._session_ids = {"r": ("tenant", "conversation")}
        runtime._session_checkpoints = {}
        events = []
        runtime.log = events.append
        self.assertFalse(runtime._park_session("r"))
        self.assertEqual(events[-1]["reason"],
                         "no sealed executed-frontier checkpoint")

    def test_checkpoint_capture_requires_exact_aligned_executed_prefix(self):
        from kvmem_sparse_runtime import VllmSparseRuntime, _ReqState
        runtime = VllmSparseRuntime.__new__(VllmSparseRuntime)
        st = _ReqState()
        st.tokens = 32
        runtime.session_reuse = True
        runtime._session_checkpoints = {}
        runtime._session_ids = {"r": ("tenant", "conversation")}
        runtime._prompt_ids = {"r": tuple(range(52))}
        runtime._prompt_len = {"r": 52}
        runtime._retrieval = {}
        runtime.capture_state = lambda req, frontier=None: {0: "GDN at 32"}
        runtime.has_recurrent = True
        runtime.kv_block_size = 16
        runtime.select_block_tokens = 16
        runtime.budget_tokens = 48
        runtime.stats_core = False
        runtime.retrieval_enabled = False
        runtime.session_checkpoint_chunk_tokens = 32
        runtime.session_checkpoint_speculative_tokens = 0
        runtime.session_checkpoint_executed_frontier = lambda req: 32
        runtime._seal_session_checkpoint_kv = lambda req, frontier: True
        events = []
        runtime.log = events.append
        self.assertTrue(runtime._maybe_capture_session_checkpoint("r", st))
        frontier, tokens, image, index, mean_tail = runtime._session_checkpoints["r"]
        self.assertEqual((frontier, tokens, image, index, mean_tail),
                         (32, tuple(range(32)), {0: "GDN at 32"}, None, None))
        runtime._session_checkpoints.clear()
        runtime._statistics = {"r": NS(frontier=33)}
        runtime.stats_core = True
        self.assertFalse(runtime._maybe_capture_session_checkpoint("r", st))
        runtime.stats_core = False
        runtime._seal_session_checkpoint_kv = lambda req, frontier: False
        self.assertFalse(runtime._maybe_capture_session_checkpoint("r", st))
        runtime._seal_session_checkpoint_kv = lambda req, frontier: True
        runtime.session_checkpoint_executed_frontier = lambda req: 31
        self.assertFalse(runtime._maybe_capture_session_checkpoint("r", st))
        runtime.session_checkpoint_executed_frontier = lambda req: 32
        st.tokens = 33
        self.assertFalse(runtime._maybe_capture_session_checkpoint("r", st))
        runtime.session_checkpoint_speculative_tokens = 4
        st.tokens = 32
        self.assertFalse(runtime._maybe_capture_session_checkpoint("r", st))

    def test_checkpoint_seals_missing_prefix_pages_before_pressure(self):
        from kvmem_sparse_runtime import VllmSparseRuntime
        runtime = VllmSparseRuntime.__new__(VllmSparseRuntime)
        pages = set()
        copied = []
        store = NS(has=lambda page: page in pages)
        def commit(page, slot, baked):
            copied.append((page, slot, baked))
            pages.add(page)
        runtime._orch = {"r": NS(store=store, commit_written_page=commit)}
        runtime._page_map = {"r": {0: 7, 1: 8}}
        runtime._state = {"r": NS(row=(), row_index=0, compacted_in_prefill=False)}
        runtime.view = NS(block_table_rows=lambda: [[NS(item=lambda: 7), NS(item=lambda: 8)]])
        runtime.page_tokens = 16
        runtime._absorb_engine_row = lambda req: None
        runtime.log = lambda record: None
        self.assertTrue(runtime._seal_session_checkpoint_kv("r", 32))
        self.assertEqual(copied, [(0, 7, 0), (1, 8, 0)])
        self.assertTrue(runtime._seal_session_checkpoint_kv("r", 32))
        self.assertEqual(len(copied), 2, "a sealed prefix must not be copied again")
        pages.clear()
        runtime._page_map["r"].pop(1)
        self.assertFalse(runtime._seal_session_checkpoint_kv("r", 32))

    def test_checkpoint_retirement_prepare_proof_and_abort(self):
        from kvmem_sparse_runtime import VllmSparseRuntime
        from kvmem_session_store import SessionStore, verify_tp_checkpoint_records
        runtime = VllmSparseRuntime.__new__(VllmSparseRuntime)
        runtime.session_reuse = True
        runtime.session_store = SessionStore()
        runtime._session_ids = {"r": ("tenant", "conversation")}
        runtime._session_checkpoints = {
            "r": (32, tuple(range(32)), {0: "GDN at 32"}, None, None)}
        runtime._preparked_requests = set()
        runtime._prompt_ids = {"r": tuple(range(52))}
        runtime._orch = {"r": NS(session_id=1, usable_pages=lambda n: [0, 1])}
        runtime._statistics = {}
        runtime.has_recurrent = True
        runtime.retrieval_enabled = False
        runtime.stats_core = False
        runtime.page_tokens = 16
        runtime.kv_block_size = 16
        runtime.pages_per_block = 1
        runtime.selector = NS(budget_pages=lambda: 4)
        runtime.tp_rank = 0
        runtime.tp_size = 2
        runtime.log = lambda event: None
        # A beyond-pool request can reconstruct the selected suffix at prompt end. The saved
        # early prefix still belongs to the ORIGINAL host archive, not the rebuilt token book.
        runtime._rebuilt_req = {"r"}
        runtime._prev_n = {}
        record = runtime.prepare_session_checkpoint("r")
        self.assertEqual(record["covered"], 32)
        self.assertEqual(len(record["hashes"]), 2)
        self.assertTrue(runtime.check_session_checkpoint(
            ("tenant", "conversation"),
            tuple(bytes.fromhex(h) for h in record["hashes"]), 32,
            record["prefix_digest"], 2))
        self.assertFalse(runtime.check_session_checkpoint(
            ("other", "conversation"),
            tuple(bytes.fromhex(h) for h in record["hashes"]), 32,
            record["prefix_digest"], 2))
        self.assertEqual(verify_tp_checkpoint_records(
            [record, {**record, "rank": 1}], 2, "r")["covered"], 32)
        runtime.abort_session_checkpoint("r")
        self.assertEqual(len(runtime.session_store), 0)

    def test_aligned_gdn_checkpoint_uses_executed_block_slot(self):
        import torch
        from unittest.mock import patch
        sys.path.insert(0, "plugin")
        import vllm_kvmem_plugin as plugin

        table = torch.tensor([[3, 7, 9]], dtype=torch.int32)
        cache = torch.arange(12, dtype=torch.float32).reshape(12, 1)
        runner = NS(block_tables=NS(block_tables=[NS(gpu=table)]),
                    model_state=NS(_align_mode=True),
                    vllm_config=NS(cache_config=NS(block_size=16)))
        st = NS(row_index=0, tokens=32, resume=None)
        runtime = NS(_state={"r": st})
        with patch.object(plugin, "_state_groups", return_value={0: [cache]}):
            image = plugin._capture_state_snapshot(runner, runtime, "r")
            self.assertEqual(image[0][0], 7)  # block-table column 1, not stale column 0
            st.tokens = 16  # lagging runtime counter must not override the executed frontier
            self.assertEqual(plugin._capture_state_snapshot(
                runner, runtime, "r", frontier=32)[0][0], 7)
            st.tokens = 32
            with patch.object(plugin, "_capture_state_snapshot", return_value=image) as capture:
                plugin._bind_state_capture(runner, runtime)
                self.assertIs(runtime.capture_state("r", 32), image)
                capture.assert_called_once_with(runner, runtime, "r", 32)
            table[0, 1] = 9  # a new request owns a different physical recurrent slot
            cache[9] = -1
            st.resume = (object(), 32, 32)
            self.assertTrue(plugin._restore_state_snapshot(runner, runtime, "r", image))
            self.assertEqual(float(cache[9]), 7.0)
            self.assertFalse(plugin._restore_state_snapshot(runner, runtime, "r", {}))

    def test_native_mean_checkpoint_restores_partial_frontier(self):
        import numpy as np
        sys.path.insert(0, "build/python")
        from kvmem_statistics import KvmemStatistics
        try:
            stats = KvmemStatistics(1, 1, 4)
        except ModuleNotFoundError:
            self.skipTest("native KVMem extension not built")
        values = {0: np.full((1, 4), 16, dtype=np.float32)}
        self.assertEqual(stats.write_step(0, 32, values), 1)
        checkpoint = stats.mean_checkpoint(32)
        self.assertEqual(stats.write_step(32, 32, values), 1)
        stats.truncate_to(32)
        stats.restore_mean_checkpoint(32, checkpoint)
        self.assertEqual(stats.frontier, 32)
        self.assertEqual(stats.n_tokens(0), 32)
        self.assertFalse(stats.has_block(1))

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
        # Read-only validation must not retire any block before both workers agree.
        planned, kept, removable = plan_attention_blocks(
            manager, "r", [40, 41, 42, 43, 48], 4)
        self.assertIs(planned, attention)
        self.assertEqual((kept, removable), (blocks[:2], blocks[2:]))
        self.assertEqual(attention.req_to_blocks["r"], blocks)
        self.assertEqual(freed, [])
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

    def test_quiescent_tp_compaction_rejects_the_measured_eight_k_gap(self):
        blocks = [NS(block_id=i) for i in (10, 11, 12)]
        attention = FullAttentionManager(blocks)
        freed = []
        cache = NS(coordinator=NS(single_type_managers=[attention]),
                   block_pool=NS(free_blocks=freed.extend))
        request = NS(request_id="r", num_in_flight_tokens=0, num_output_placeholders=0,
                     num_logical_tokens=153280, num_computed_tokens=153280,
                     num_kvmem_offset=0)
        base = {"request": "r", "epoch": 9, "logical_frontier": 153280,
                "resident_frontier": 153280, "selected_pages": [0, 1, 2, 3, 4, 4776],
                "physical_pages": [40, 41, 42, 43, 44, 45],
                "page_tokens": 32, "block_pages": 4, "visible_tokens": 192}
        records = [{"rank": rank, **base} for rank in (0, 1)]
        result = validate_quiescent_compaction(request, records, 2, 9, cache)
        self.assertEqual(result["visible_tokens"], 192)
        self.assertEqual(attention.req_to_blocks["r"], blocks)
        self.assertEqual(freed, [])

        # The observed worker proposals lagged Core by 8,192 tokens on BOTH failed
        # beyond-pool compactions. Neither may free an attention block.
        for core_frontier, worker_frontier in ((153280, 145088), (169664, 161472)):
            request.num_logical_tokens = request.num_computed_tokens = core_frontier
            stale = [{"rank": rank, **base, "logical_frontier": worker_frontier,
                      "resident_frontier": worker_frontier} for rank in (0, 1)]
            with self.subTest(core=core_frontier), self.assertRaises(ValueError):
                validate_quiescent_compaction(request, stale, 2, 9, cache)
        request.num_logical_tokens = request.num_computed_tokens = 153280
        request.num_in_flight_tokens = 4096
        with self.assertRaises(ValueError):
            validate_quiescent_compaction(request, records, 2, 9, cache)
        request.num_in_flight_tokens = 0
        for mismatched in ([records[0]],
                           [records[0], {**records[1], "epoch": 8}],
                           [records[0], {**records[1], "selected_pages": [0, 1]}],
                           [records[0], {**records[1], "physical_pages": [80] * 6}],
                           [{**record, "physical_pages": [80, 81, 82, 83, 84, 85]}
                            for record in records],
                           [records[0], {**records[1], "rank": 0}]):
            with self.assertRaises(ValueError):
                validate_quiescent_compaction(request, mismatched, 2, 9, cache)
        self.assertEqual(attention.req_to_blocks["r"], blocks)
        self.assertEqual(freed, [])


if __name__ == "__main__":
    unittest.main()
