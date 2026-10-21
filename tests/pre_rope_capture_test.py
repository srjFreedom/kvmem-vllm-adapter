# mean-K retrieval needs PRE-RoPE Q/K. The vLLM attention layer only exposes post-RoPE values, so
# we invert the rotation using the token positions; this test proves the inversion is exact and
# that the resulting scores point at the block that actually contains the query's key.
import sys

import torch

sys.path.insert(0, "src")
from kvmem_pre_rope_capture import (block_scores, capture_needed, query_sums, rotary_inv_freq,
                                    unrotate)

failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def rotate(x, positions, inv_freq, rotary_dim):
    out = x.clone()
    half = rotary_dim // 2
    k = x[..., :rotary_dim].to(torch.float64)
    x1, x2 = k[..., :half], k[..., half:]
    angle = (positions.to(torch.float64)[:, None, None] *
             inv_freq.to(torch.float64)[None, None, :])
    cos, sin = torch.cos(angle), torch.sin(angle)
    out[..., :rotary_dim] = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos],
                                      dim=-1).to(x.dtype)
    return out


def main():
    torch.manual_seed(0)
    heads, head_dim, rotary_dim, page_tokens = 2, 8, 4, 3
    inv = rotary_inv_freq(rotary_dim, 1e7)

    positions = torch.arange(12)
    x = torch.randn(12, heads, head_dim, dtype=torch.float32)
    post = rotate(x, positions, inv, rotary_dim)
    back = unrotate(post, positions, inv, rotary_dim)
    check(torch.allclose(back, x, atol=1e-5),
          f"inverse rotation recovers the pre-RoPE tensor (max err "
          f"{(back - x).abs().max():.2e})")
    check(not torch.allclose(post, x, atol=1e-3), "the rotation is not the identity")

    # Only the first rotary_dim channels are touched; the tail must survive untouched.
    tail = torch.randn(12, heads, head_dim)
    tail_post = rotate(tail, positions, inv, rotary_dim)
    check(torch.allclose(tail_post[..., rotary_dim:], tail[..., rotary_dim:]),
          "channels beyond rotary_dim are left alone")

    # Scores: build keys where block 1 carries the query's key and everything else is noise.
    pages = 12                                   # 4 blocks of 3 pages
    blocks = pages // 3
    keys = torch.randn(pages, heads, page_tokens, head_dim) * 0.05
    q = torch.randn(1, heads, head_dim)
    q_sums = query_sums(q.repeat(4, 1, 1))
    target_block = 1
    pages_of_target = slice(target_block * 3, (target_block + 1) * 3)
    keys[pages_of_target] = q[0].unsqueeze(1).unsqueeze(0).expand(3, heads, page_tokens, head_dim)
    scores = block_scores(keys, q_sums, block_pages=3, page_tokens=page_tokens)
    check(len(scores) == blocks, f"one score per block ({len(scores)})")
    check(int(max(range(blocks), key=lambda i: scores[i])) == target_block,
          f"the block holding the query key scores highest ({[round(s, 3) for s in scores]})")

    # Capture cadence: only near a reselect.
    check(capture_needed(16000, 16384), "capture starts just before the reserve is exhausted")
    check(not capture_needed(0, 16384), "no capture right after a reselect")

    print("PASS pre_rope_capture_test" if not failed else "FAILED pre_rope_capture_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
