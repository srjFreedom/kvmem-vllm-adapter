# Re-RoPE helpers for the sparse-compaction path (KVMem "A mode" on vLLM).
#
# vLLM's paged attention has no per-key positions: a page's logical position is implied by its
# slot in the block table. Compacting a selected page from slot `src` to slot `dst` therefore
# changes the relative offset the attention kernel will compute, so the stored K must be rotated
# by delta = (dst - src) * page_tokens.
#
# Rotation by a delta is position-independent: it multiplies each rotary pair by
# exp(i * delta * inv_freq[j]); no absolute-position table is needed. Same values for every key
# in the page, so one factor per page covers its 16 tokens.
#
# K is stored quantized (fp8/int8) with per-(block, token, head) scales in this fork, so callers
# dequantize, rotate, and re-quantize; the scale stays valid (rotation is magnitude preserving up
# to the fp8 grid) but a rotated value may fall outside the old scale's range, so the caller
# should clamp or recompute the scale.
import torch


def rotary_inv_freq(rotary_dim: int, base: float, device=None) -> torch.Tensor:
    """The standard inverse frequencies for the first `rotary_dim` channels (float64)."""
    if rotary_dim % 2:
        raise ValueError("rotary_dim must be even")
    exponent = torch.arange(0, rotary_dim, 2, dtype=torch.float64, device=device) / rotary_dim
    return torch.pow(torch.tensor(float(base), dtype=torch.float64, device=device), -exponent)


def delta_rotate(k: torch.Tensor, delta_tokens: int | torch.Tensor, inv_freq: torch.Tensor,
                 rotary_dim: int) -> torch.Tensor:
    """Rotate the rotary channels of K by `delta_tokens` positions (NeoX / split-half layout).

    k: [..., head_dim] (any leading dims). Only the first `rotary_dim` channels are rotary.
    """
    head_dim = k.shape[-1]
    if rotary_dim > head_dim:
        raise ValueError("rotary_dim exceeds head_dim")
    half = rotary_dim // 2
    # float64 internally: the engine's own fp32 cos/sin cache already loses ~4e-3 of precision
    # at 200K positions, and a delta rotation must not add to that.
    angle = inv_freq.to(device=k.device, dtype=torch.float64) * float(delta_tokens)
    cos = torch.cos(angle).to(k.dtype)
    sin = torch.sin(angle).to(k.dtype)
    out = k.clone()
    x1 = k[..., :half]
    x2 = k[..., half:rotary_dim]
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:rotary_dim] = x1 * sin + x2 * cos
    return out


def rotate_pages_in_place(kv_tensor: torch.Tensor, page_map, page_tokens: int,
                          inv_freq: torch.Tensor, rotary_dim: int,
                          content_dim: int) -> None:
    """Rotate K in place for pages that moved.

    kv_tensor: [pages, kv_heads, page_tokens, content_dim] uint8/int8 (quantized K+V layout) or
               float for tests.
    page_map: iterable of (src_page, dst_page); K is rotated by (dst - src) * page_tokens.
    """
    for src, dst in page_map:
        if src == dst:
            continue
        delta = (dst - src) * page_tokens
        k = kv_tensor[src, :, :, :rotary_dim].to(torch.float32)
        rotated = delta_rotate(k, delta, inv_freq, rotary_dim)
        kv_tensor[src, :, :, :rotary_dim] = rotated.to(kv_tensor.dtype)


# --- fp8 KV payloads ---------------------------------------------------------------
# The cache stores K (and V) as raw fp8 e4m3 bytes; this fork's packed fp8 layout has no inline
# per-token scales (content dim = K|V), i.e. a per-tensor scale. A rotation acts inside one head
# and is linear, so it commutes with any per-tensor (or per (token, head)) positive scale: the
# stored bytes may be rotated directly without knowing the scale.
#
# The catch is precision: re-quantizing to e4m3 after every rotation would drift. So a page's
# device copy must always be derived from its CANONICAL copy (the host pool keeps it), never by
# rotating an already-rotated device page.
FP8_DTYPE = torch.float8_e4m3fn


def _dequant(bytes_view: torch.Tensor) -> torch.Tensor:
    # `.view(dtype)` reinterprets bits and requires a CONTIGUOUS last dimension. A strided slice
    # like `out[..., :head_dim]` silently reinterprets the wrong bytes, so force a copy first.
    return bytes_view.contiguous().view(FP8_DTYPE).to(torch.float32)


def _quantize(x: torch.Tensor) -> torch.Tensor:
    return x.to(FP8_DTYPE).view(torch.uint8)


def rotate_k_pages_fp8(kv_tensor: torch.Tensor, page_map, page_tokens: int,
                       inv_freq: torch.Tensor, rotary_dim: int,
                       k_slice: slice = slice(0, 256)) -> None:
    """Rotate the K half of the given pages, in place, as fp8.

    kv_tensor: [pages, kv_heads, page_tokens, content] uint8/fp8 bytes.
    page_map:  iterable of (src_page, dst_page); the rotation is (dst - src) * page_tokens.
    k_slice:   the K byte range inside `content` (V follows and is not touched).
    """
    for src, dst in page_map:
        if src == dst:
            continue
        delta = (dst - src) * page_tokens
        k = _dequant(kv_tensor[src, :, :, k_slice])
        kv_tensor[src, :, :, k_slice] = _quantize(delta_rotate(k, delta, inv_freq, rotary_dim))
