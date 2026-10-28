"""Pre-RoPE query/key capture for KVMem's mean-K retrieval.

Why inverse rotation instead of intercepting tensors: the vLLM attention layer exposes only the
POST-RoPE q/k (`Qwen3NextAttention._project_qkv_gate` returns post-RoPE values, and one path fuses
norm+rope into a Triton kernel, so pre-RoPE values never materialise). Rotating back is exact up to
floating point: RoPE is an orthogonal rotation per (head, position pair), so applying the inverse
rotation with the token's own position recovers the pre-RoPE space that KVMem's retrieval scoring
expects (`pre_rope_statistics`). This keeps the fork untouched - no graph surgery, no kernel flags.

Reference: kvmem's own backend (`kvmem_capture_register/harvest` in llama) does the same job by
registering the graph tensors; both approaches yield "position-invariant" Q/K for scoring.
"""
from __future__ import annotations

import math

import torch


def rotary_inv_freq(rotary_dim: int, theta: float, device=None, dtype=torch.float64):
    """Standard RoPE frequencies for the first `rotary_dim` channels (NeoX split-half pairing)."""
    half = rotary_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=dtype, device=device) / half))
    return freqs


def _split_half(x):
    half = x.shape[-1] // 2
    return x[..., :half], x[..., half:]


def unrotate(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor,
             rotary_dim: int) -> torch.Tensor:
    """Undo RoPE on the first `rotary_dim` channels of `x` (…, head_dim) for each token.

    x:         (tokens, heads, head_dim) post-RoPE values
    positions: (tokens,) int positions used for this step
    """
    # Clamp instead of raising: a rotary_dim larger than the head width, odd, or zero used to abort
    # the caller (and with it the whole harvest), leaving the request with no query at all.
    rotary_dim = int(rotary_dim)
    if rotary_dim <= 0 or x.shape[-1] < rotary_dim:
        rotary_dim = int(x.shape[-1])
    if rotary_dim % 2:
        rotary_dim -= 1
    if rotary_dim <= 0:
        return x
    out = x.clone()
    half = rotary_dim // 2
    # float32 keeps the temporaries small: a 4K-token chunk x 16 heads x 256 in float64 is >100 MB
    # per layer, which OOMed the engine (capture.errors == wants).
    k = x[..., :rotary_dim].to(torch.float32)
    x1, x2 = _split_half(k)
    # (tokens, 1, ..., half) so the angle broadcasts over every middle dimension (heads, etc.).
    shape = (positions.shape[0],) + (1,) * (x.ndim - 2) + (half,)
    angle = (-positions.to(torch.float32).reshape(shape[0], *([1] * (x.ndim - 2)), 1) *
             inv_freq.to(torch.float32).reshape(1, *([1] * (x.ndim - 2)), half))
    cos, sin = torch.cos(angle), torch.sin(angle)
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    out[..., :rotary_dim] = torch.cat([r1, r2], dim=-1).to(x.dtype)
    return out


def query_sums(q_pre: torch.Tensor) -> torch.Tensor:
    """Sum the pre-RoPE queries over tokens, per head -> (heads, head_dim) FP32.

    KVMem's MeanKeyQueryView takes exactly this shape (query_heads * head_dim FP32 sums).
    """
    return q_pre.to(torch.float32).sum(dim=0)


def block_scores(keys_pre: torch.Tensor, q_sums: torch.Tensor, block_pages: int,
                 page_tokens: int) -> list[float]:
    """Cosine similarity between the mean query and each block's mean key.

    keys_pre:  (pages, kv_heads, page_tokens, head_dim) pre-RoPE keys, device or host
    q_sums:    (heads, head_dim) FLOATS from query_sums()
    Returns one score per block, in block order.
    """
    if keys_pre.ndim != 4:
        raise ValueError("keys_pre must be (pages, kv_heads, page_tokens, head_dim)")
    pages = keys_pre.shape[0]
    blocks = pages // block_pages
    if blocks * block_pages != pages:
        raise ValueError("pages must be a multiple of block_pages")
    k = keys_pre.to(torch.float32).reshape(blocks, block_pages, keys_pre.shape[1],
                                          page_tokens, keys_pre.shape[3])
    k_mean = k.mean(dim=(1, 2, 3))                     # (blocks, head_dim)
    q_mean = q_sums.to(torch.float32).mean(dim=0)      # (head_dim,)
    k_norm = k_mean.norm(dim=-1).clamp_min(1e-6)
    q_norm = q_mean.norm().clamp_min(1e-6)
    cos = (k_mean @ q_mean) / (k_norm * q_norm)
    return [float(v) for v in cos]


def capture_needed(tokens_since_reselect: int, gen_reserve: int, lookahead_frac: float = 0.125) -> bool:
    """Capture only near a reselect, never on every step.

    The reserve R is the amount of new tokens allowed between reselects, so capture starts once we
    are within the lookahead window of exhausting it. KVMem's own backends do the same
    (`kvmem_ubatch_needs_q_capture`) to keep the per-step cost off the decode path.
    """
    reserve = max(1, int(gen_reserve))
    lookahead = max(1, int(reserve * float(lookahead_frac)))
    return int(tokens_since_reselect) >= reserve - lookahead
