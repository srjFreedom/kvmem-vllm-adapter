"""Two-rank CPU tests for the opt-in reconstruction score collective."""
import os
import sys
import tempfile
import unittest
from multiprocessing import get_context
from types import SimpleNamespace

sys.path.insert(0, "src")


def _rank_worker(rank, rendezvous, output):
    import torch
    from vllm.distributed import parallel_state
    from kvmem_tp_scores import average_reconstruction_scores

    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    parallel_state.get_tp_group = lambda: SimpleNamespace(
        device_group=torch.distributed.group.WORLD)
    try:
        local = [2.0, 8.0] if rank == 0 else [6.0, 4.0]
        averaged = average_reconstruction_scores(local, 2, "same-request", 196598,
                                                 torch.device("cpu"))
        failures = []
        for scores, expected, req in ((None if rank else local, 2, "same-request"),
                                      (local[:1] if rank else local, 2, "same-request"),
                                      (local, 2, "other-request" if rank else "same-request")):
            try:
                average_reconstruction_scores(scores, expected, req, 196598,
                                              torch.device("cpu"))
            except RuntimeError:
                failures.append(True)
            else:
                failures.append(False)
        output.put((rank, averaged, failures))
    finally:
        torch.distributed.destroy_process_group()


class TpReconstructionScoreTest(unittest.TestCase):
    def test_two_rank_consensus_and_collective_refusals(self):
        context = get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = os.path.join(directory, "tp-rendezvous")
            output = context.Queue()
            processes = [context.Process(target=_rank_worker, args=(rank, rendezvous, output))
                         for rank in (0, 1)]
            for process in processes:
                process.start()
            try:
                results = [output.get(timeout=30) for _ in processes]
            finally:
                for process in processes:
                    process.join(timeout=10)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
            self.assertEqual(sorted(rank for rank, _, _ in results), [0, 1])
            self.assertTrue(all(scores == [4.0, 6.0] for _, scores, _ in results))
            self.assertTrue(all(failures == [True, True, True]
                                for _, _, failures in results))
            self.assertTrue(all(process.exitcode == 0 for process in processes))


if __name__ == "__main__":
    unittest.main()
