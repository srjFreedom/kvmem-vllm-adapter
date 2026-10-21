# Verifies the compaction re-RoPE against vLLM's OWN rotary embedding for this model's config.
# If delta-rotating K by d positions does not reproduce the engine's K at position p+d, the
# sparse-compaction path would silently corrupt attention, so this is the gate for stage 3.
import json
import os
import sys

import torch

sys.path.insert(0, "src")
from kvmem_sparse_rope import delta_rotate, rotary_inv_freq

try:
    import vllm.model_executor.custom_op as _custom_op
    from vllm.model_executor.layers.rotary_embedding import get_rope

    # CustomOps normally need a live VllmConfig to pick a backend. This test only wants the
    # reference math, so pin every CustomOp to its native implementation.
    def _dispatch_native(self, *args, **kwargs):
        self._forward_method = self.forward_native

    _custom_op.CustomOp.dispatch_forward = _dispatch_native
    _custom_op.CustomOp.enabled = lambda self: False
except Exception as exc:  # pragma: no cover
    print("SKIP: vLLM not importable:", exc)
    sys.exit(0)

CONFIG = os.environ.get("KVMEM_TEST_CONFIG", "")
if not CONFIG or not os.path.exists(CONFIG):
    print("SKIP: set KVMEM_TEST_CONFIG to a model config.json to run this test")
    sys.exit(0)


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        check.failed = True


check.failed = False


def main():
    config = json.load(open(CONFIG))
    text = config.get("text_config", config)
    rope_params = dict(text["rope_parameters"])
    head_dim = int(text["head_dim"])
    partial = float(rope_params.get("partial_rotary_factor", 1.0))
    rotary_dim = int(head_dim * partial)
    base = float(rope_params.get("rope_theta", 10000.0))
    print(f"head_dim={head_dim} rotary_dim={rotary_dim} base={base:.0f} "
          f"mrope_section={rope_params.get('mrope_section')} "
          f"interleaved={rope_params.get('mrope_interleaved')}")

    rope = get_rope(head_size=head_dim, max_position=204800, rope_parameters=rope_params,
                    dtype=torch.float32)
    print("rope class:", type(rope).__name__, "is_neox_style=", getattr(rope, "is_neox_style", None))

    torch.manual_seed(0)
    kv_heads = 2
    k = torch.randn(1, kv_heads * head_dim, dtype=torch.float32) * 0.5
    inv_freq = rotary_inv_freq(rotary_dim, base)

    for p, d in ((1234, 16), (1234, -16), (0, 96)):
        # The engine's K at position p and at p+d (text-only: 1D positions).
        pos_p = torch.tensor([p])
        pos_pd = torch.tensor([p + d])
        _, k_p = rope.forward_native(pos_p, k.clone(), k.clone())
        _, k_pd = rope.forward_native(pos_pd, k.clone(), k.clone())
        # K is packed as [tokens, heads * head_dim]; the rotation is per head, so compare and
        # rotate in the head-shaped view (rotary channels are the first rotary_dim of each head).
        p_view = k_p.view(1, kv_heads, head_dim)
        pd_view = k_pd.view(1, kv_heads, head_dim)
        mine = delta_rotate(p_view, d, inv_freq, rotary_dim)
        k_pd = pd_view
        max_err = (mine - k_pd).abs().max().item()
        rel = max_err / max(1e-6, k_pd.abs().max().item())
        check(rel < 2e-4, f"delta rotation matches the engine (p={p}, d={d}, rel_err={rel:.2e})")

    # At 200K positions the ENGINE's own fp32 cos/sin cache carries ~4e-3 error, so an exact
    # match is impossible by construction; require the same order and prove the residual is the
    # engine's cache error, not ours.
    p, d = 200000, -128
    _, k_p = rope.forward_native(torch.tensor([p]), k.clone(), k.clone())
    _, k_pd = rope.forward_native(torch.tensor([p + d]), k.clone(), k.clone())
    mine = delta_rotate(k_p.view(1, kv_heads, head_dim), d, inv_freq, rotary_dim)
    rel = (mine - k_pd.view(1, kv_heads, head_dim)).abs().max().item() / \
        k_pd.abs().max().item()
    cache_err = (rope.cos_sin_cache[p].double()[:32] -
                 torch.cos(inv_freq * p)).abs().max().item()
    check(rel < 10 * cache_err, f"large position: residual {rel:.2e} is the engine's own "
                                f"fp32 cache error ({cache_err:.2e})")

    # Round trip: rotating forward then back must return the original bytes.
    p_view = k_p.view(1, kv_heads, head_dim).clone()
    there = delta_rotate(p_view, 96, inv_freq, rotary_dim)
    back = delta_rotate(there, -96, inv_freq, rotary_dim)
    check((back - p_view).abs().max().item() < 1e-6, "delta rotation is exactly invertible")

    # The pass-through channels (beyond rotary_dim) must be untouched by a delta rotation.
    _, k_p = rope.forward_native(torch.tensor([777]), k.clone(), k.clone())
    p_view = k_p.view(1, kv_heads, head_dim)
    mine = delta_rotate(p_view, 64, inv_freq, rotary_dim)
    same_tail = torch.equal(mine[..., rotary_dim:], p_view[..., rotary_dim:])
    check(same_tail, "non-rotary channels are left untouched")

    print("PASS sparse_rope_test" if not check.failed else "FAILED sparse_rope_test")
    sys.exit(0 if not check.failed else 1)


if __name__ == "__main__":
    main()
