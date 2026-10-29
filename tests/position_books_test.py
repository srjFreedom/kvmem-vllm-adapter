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


if __name__ == "__main__":
    unittest.main()
