# The engine-side hook: wrapping _project_qkv_gate must recover pre-RoPE keys from the post-RoPE
# values and feed them into the retrieval index, with the layer index and page grouping correct.
import sys
import types

import torch

sys.path.insert(0, "src")
from kvmem_pre_rope_capture import rotary_inv_freq
from kvmem_retrieval_index import KvmemRetrievalIndex
from kvmem_engine_capture import EngineCapture

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
    PAGE, HD, ROT, HEADS = 16, 8, 4, 1
    inv = rotary_inv_freq(ROT, 1e7)

    # A fake attention layer: layer_idx + rotary_emb are all the hook reads.
    layer = types.SimpleNamespace(layer_idx=3,
                                  rotary_emb=types.SimpleNamespace(rotary_dim=ROT, base=1e7))

    idx = KvmemRetrievalIndex(block_tokens=32, page_tokens=PAGE, head_dim=HD, layers=8)
    cap = EngineCapture(lambda: idx, gen_reserve=4)

    # 2 pages of post-RoPE q/k, positions spanning both pages.
    torch.manual_seed(0)
    positions = torch.arange(2 * PAGE)
    q_true = torch.randn(2 * PAGE, HEADS, HD)
    k_true = torch.randn(2 * PAGE, HEADS, HD)
    q_post = rotate(q_true, positions, inv, ROT)
    k_post = rotate(k_true, positions, inv, ROT)

    cap.reset_tokens()
    check(not cap.wants_capture(), "no capture right after a reselect")
    cap.note_tokens(4)
    check(cap.wants_capture(), "capture starts near a reselect")

    cap.harvest(layer, q_post, k_post, positions)
    check(cap.captured == 1, "the hook harvested one step")
    check(idx.covered_blocks() == [0], f"both pages landed in block 0 ({idx.covered_blocks()})")
    check(3 in idx._query.nonzero()[:, 0].tolist(),
          "the query landed on the layer the attention layer reported")

    # The index's stored statistic must equal the mean over the TRUE (pre-RoPE) keys.
    expected = k_true.mean(dim=0).mean(dim=0)
    stored = (idx._sums[0][3] / idx._counts[0][3].clamp_min(1e-6))
    check(torch.allclose(stored, expected, atol=1e-4),
          f"the block statistic matches the pre-RoPE mean (err "
          f"{(stored - expected).abs().max():.2e})")

    print("PASS engine_capture_test" if not failed else "FAILED engine_capture_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
