"""Synchronous TP score consensus for the opt-in reconstruction boundary."""
import hashlib
import math

import torch


def average_reconstruction_scores(scores, expected, request_id, frontier, device):
    """Average aligned page scores across TP ranks, or fail all ranks together.

    The fixed-width preflight precedes the variable-width score collective, so missing
    statistics or a divergent frontier cannot hang one rank inside a mismatched reduce.
    This must only be called on a boundary reached by every TP worker.
    """
    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tp_group()
    group = tp.device_group
    if not torch.distributed.is_initialized():
        raise RuntimeError("TP reconstruction requires an initialized process group")
    size = torch.distributed.get_world_size(group=group)
    digest = hashlib.sha256(str(request_id).encode("utf-8")).digest()
    length = len(scores) if scores is not None else -1
    present = scores is not None and all(math.isfinite(float(v)) for v in scores)
    metadata = torch.tensor(
        [int.from_bytes(digest[:4], "big"), int.from_bytes(digest[4:8], "big"),
         int(present), length, int(expected), int(frontier)],
        dtype=torch.int64, device=device,
    )
    minimum = metadata.clone()
    maximum = metadata.clone()
    torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN, group=group)
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=group)
    if (not torch.equal(minimum, maximum) or not present
            or length != int(expected) or length <= 0):
        raise RuntimeError("TP reconstruction score availability, shape or request differs")
    values = torch.as_tensor(scores, dtype=torch.float32, device=device).contiguous()
    if size > 1:
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM, group=group)
        values /= size
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("TP reconstruction scores are nonfinite")
    return values.tolist()
