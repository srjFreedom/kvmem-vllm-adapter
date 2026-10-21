"""KVMem bridge: expose the current step's query to an out-of-tree policy layer.

The hard constraint: under `torch.compile` (fullgraph) + CUDA Graph replay **no Python runs per
decode step**, and the compiled region tolerates nothing that dynamo cannot trace - a graph break,
an allocation, or a `torch` call returning a non-tensor all abort compilation (observed:
"op returned non-Tensor", then "RuntimeError when making fake tensor call").

So the copy lives in a **registered custom op**. Dynamo treats it as opaque, inductor emits the
call into the graph, and the copies it performs are ordinary kernels that replay with the live
buffers. Buffers are allocated once, from outside the graph (`ensure()`), by the adapter.

Keys do not need this bridge: they are already in the KV cache in post-RoPE form, and the policy
layer can read them and un-rotate using the original positions.
"""
from __future__ import annotations

import os

import torch

ARMED = os.environ.get("KVMEM_CAPTURE", "0") == "1"
MAX_TOKENS = int(os.environ.get("KVMEM_CAPTURE_TOKENS", "64") or "64")

_query: dict[int, torch.Tensor] = {}
_count: dict[int, torch.Tensor] = {}
_positions: torch.Tensor | None = None
_pos_count: torch.Tensor | None = None
_ready = False
_op_calls = 0        # how many times the custom op body actually ran (diagnostic)
_op_calls = 0        # how many times the custom op body actually ran (diagnostic)


def ensure(layers: int, heads: int, head_dim: int, device="cuda", max_tokens: int | None = None,
           dtype: torch.dtype = torch.bfloat16) -> bool:
    """Allocate the buffers. Must be called outside any compiled region (adapter does this)."""
    global _positions, _pos_count, _ready
    take = int(max_tokens or MAX_TOKENS)
    # Allocate ONCE per process. The compiled graph and its CUDA-graph replay capture the buffer
    # pointers, so re-allocating later would leave the kernels writing into the old (freed) tensors
    # while `read()` looks at the new ones - which is exactly how the query silently vanished.
    if _ready:
        return True
    try:
        for layer in range(int(layers)):
            _query[layer] = torch.zeros(take, int(heads), int(head_dim), dtype=dtype,
                                        device=device)
            _count[layer] = torch.zeros(1, dtype=torch.int32, device=device)
        _positions = torch.zeros(take, dtype=torch.int64, device=device)
        _pos_count = torch.zeros(1, dtype=torch.int32, device=device)
        _ready = True
        return True
    except Exception:
        _ready = False
        return False


# `mutates_args=("query",)` is what keeps inductor from deleting this node: an op with no declared
# mutation and a None result looks pure, and the graph compiler drops it (measured: the op body ran
# zero times after a clean compile). The op does not actually modify the query.
@torch.library.custom_op("kvmem::capture_q", mutates_args=("query",))
def capture_q(query: torch.Tensor, positions: torch.Tensor, layer: int) -> None:
    """Opaque to dynamo; the body below is plain tensor work executed at capture/replay time."""
    global _op_calls
    _op_calls += 1
    if not ARMED:
        return None
    layer = int(layer)
    buf = _query.get(layer)
    pos_buf = _positions
    if buf is None or pos_buf is None or _pos_count is None:
        return None
    # The model hands us a flat [tokens, heads*head_dim] tensor; the buffer is [tokens, heads, dim].
    if query.dim() == 2 and buf.dim() == 3:
        query = query.view(query.shape[0], buf.shape[1], buf.shape[2])
    n = query.shape[0]
    take = MAX_TOKENS if n > MAX_TOKENS else n
    if n > MAX_TOKENS:
        buf.copy_(query[n - MAX_TOKENS:])
    else:
        buf[:n].copy_(query)
    # M-RoPE hands in [3, tokens] (one row per spatial axis); text is the identity, so the first
    # row is the token position. Flattening it instead made 3*tokens values for n tokens.
    pos = positions if positions.ndim == 1 else positions[0]
    if n > MAX_TOKENS:
        pos_buf.copy_(pos[n - MAX_TOKENS:])
    else:
        pos_buf[:n].copy_(pos)
    _count[layer].fill_(take)
    _pos_count.fill_(take)
    # NOTE: no `.item()` / host sync here. This body also runs while CUDA graphs are being captured,
    # where a device->host synchronisation aborts the capture ("operation failed due to a previous
    # error during capture").
    return None


@capture_q.register_fake
def _(query, positions, layer):
    return None


def read(layer: int):
    """(query_rows, positions) buffered for `layer`, or (None, None). Host-side, outside the graph."""
    buf = _query.get(int(layer))
    count = _count.get(int(layer))
    if buf is None or count is None:
        return None, None
    take = int(count.item())
    if take <= 0:
        return None, None
    return buf[:take], (_positions[:take] if _positions is not None else None)


def stats():
    """Diagnostic: whether the compiled graph really calls the capture op."""
    return {"op_calls": _op_calls, "armed": ARMED, "layers": len(_query),
            "counts": {k: int(v.item()) for k, v in list(_count.items())[:4]}}


def read_all():
    return {layer: read(layer) for layer in list(_query)}
