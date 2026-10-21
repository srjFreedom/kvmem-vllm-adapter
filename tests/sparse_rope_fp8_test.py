# Quantifies the fp8 re-RoPE: (a) a single rotation from canonical bytes stays within one fp8
# step, (b) rotating an ALREADY rotated page accumulates error (which is why the design always
# derives the device copy from the canonical host copy), (c) V bytes are untouched.
import sys

import torch

sys.path.insert(0, "src")
from kvmem_sparse_rope import (FP8_DTYPE, _dequant, _quantize, delta_rotate, rotate_k_pages_fp8,
                               rotary_inv_freq)

ROTARY = 64
HEAD_DIM = 256
HEADS = 2
PAGE_TOKENS = 16
INV = rotary_inv_freq(ROTARY, 1e7)
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def main():
    torch.manual_seed(0)
    pages = 4
    # Canonical (float64 truth) K and V.
    k_true = torch.randn(pages, HEADS, PAGE_TOKENS, HEAD_DIM, dtype=torch.float64) * 0.5
    v_true = torch.randn(pages, HEADS, PAGE_TOKENS, HEAD_DIM, dtype=torch.float64) * 0.5
    # Pack as the cache does: content = [K | V] fp8 bytes.
    packed = torch.cat([k_true.to(torch.float32), v_true.to(torch.float32)], dim=-1)
    cache = _quantize(packed).contiguous()  # uint8 [pages, heads, tokens, 512]
    canonical = cache.clone()

    page_map = [(0, 3)]  # page 0 moves to slot 3: delta = +48 tokens
    rotate_k_pages_fp8(cache, page_map, PAGE_TOKENS, INV, ROTARY)

    truth = delta_rotate(k_true[0].to(torch.float32), 48, INV, ROTARY)
    got = _dequant(cache[0, :, :, :HEAD_DIM]).to(torch.float64)
    rel = (got - truth.double()).abs().max().item() / truth.abs().max().item()
    fp8_step = 2.0 ** -3  # e4m3 has 3 mantissa bits
    check(rel < 2 * fp8_step, f"one rotation from canonical stays within fp8 step (rel={rel:.3f})")

    v_untouched = torch.equal(cache[0, :, :, HEAD_DIM:], canonical[0, :, :, HEAD_DIM:])
    check(v_untouched, "V bytes are untouched by the K rotation")

    # Rotating the ALREADY rotated page drifts: this is the case the design must avoid.
    drifted = cache.clone()
    rotate_k_pages_fp8(drifted, [(0, 1)], PAGE_TOKENS, INV, ROTARY)   # +48 then +16
    want_move = delta_rotate(k_true[0].to(torch.float32), 64, INV, ROTARY)
    got_drift = _dequant(drifted[0, :, :, :HEAD_DIM]).to(torch.float64)
    rel_drift = (got_drift - want_move.double()).abs().max().item() / want_move.abs().max().item()
    check(rel_drift > rel, f"chained rotation drifts more than one hop "
                           f"(chained={rel_drift:.3f} vs one={rel:.3f})")

    # Correct way: always rotate from the canonical copy.
    fresh = canonical.clone()
    rotate_k_pages_fp8(fresh, [(0, 4)], PAGE_TOKENS, INV, ROTARY)     # +64 in one hop
    got_fresh = _dequant(fresh[0, :, :, :HEAD_DIM]).to(torch.float64)
    rel_fresh = (got_fresh - want_move.double()).abs().max().item() / want_move.abs().max().item()
    check(rel_fresh <= rel * 1.5, f"rotating from canonical keeps one-hop precision "
                                  f"(rel={rel_fresh:.3f})")

    print("PASS sparse_rope_fp8_test" if not failed else "FAILED sparse_rope_fp8_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
