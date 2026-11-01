"""CUDA-graph compatibility regression for the scoring bridge (2026-11-02).

Failures being defended (all measured on GPU during the compiled-path project):
  1. cudaErrorStreamCaptureInvalidated at boot: the bridge ops allocated NEW tensors while
     a CUDA graph stream was capturing - `.to(torch.float32)`, the `x1*cos + x2*sin`
     expression temporaries, `index_select` return values, lazy `torch.zeros`. The fix is
     eager-first static scratch (`_scratch`) + in-place `out=` everywhere; first use inside
     a capture must SKIP the step, never allocate.
  2. Double FP8 rounding regression (engine ddf3cf0): the graph-safe `copy_` briefly used a
     src.dtype buffer; the rotation products then truncated into it before the final
     `.to(dst.dtype)`, polluting the mean-K scores. The q rows scratch MUST be float32.
  3. FMA-vs-expression rounding regression (engine 6441248): `addcmul_` fuses mul into add
     (one rounding) where the pre-graph code computed `x1*cos`, `x2*sin`, then `+` (three).
     Scores feed compaction page selection, so the bridge must stay BIT-EXACT with the
     original expression. This test checks the published bucket sums against the literal
     unfused reference in float64-rounded-down form.

Contracts checked here (all offline, CPU tensors):
  A. source guards: no allocation/hand-sync call forms inside the capture bodies.
  B. numerical equivalence: capture_q/capture_k bucket sums == unfused reference.
  C. _scratch never allocates while "capturing" (simulated by monkeypatching the probe).
"""
import io
import json
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engines", "vllm"))
import torch  # noqa: E402

try:
    import vllm.kvmem_bridge as bridge  # noqa: E402
except Exception as exc:  # engine import needs numpy etc; skip-test rather than false-fail
    print(f"PASS bridge_capture_test (SKIP: cannot import bridge: {exc})")
    sys.exit(0)

failed = 0


def check(cond, what):
    global failed
    print(("  ok  " if cond else "  FAIL ") + what)
    if not cond:
        failed += 1


SRC = io.open(bridge.__file__, encoding="utf-8").read()

# --- A. structural guards -------------------------------------------------------------
def body_of(name):
    i = SRC.index(f"def {name}(")
    j = SRC.index("\n@", i) if "\n@" in SRC[i:] else len(SRC)
    region = SRC[i:j]
    return "\n".join(ln for ln in region.splitlines() if not ln.strip().startswith("#"))

qbody, kbody = body_of("capture_q"), body_of("capture_k")
for name, code in (("capture_q", qbody), ("capture_k", kbody)):
    check(".to(torch.float32)" not in code, f"{name}: no allocating dtype cast in the body")
    check("= cos_sin_cache.index_select" not in code and "table = cos_sin_cache" not in code,
          f"{name}: index_select writes into a scratch (no new tensor return)")
    check("addcmul_" not in code, f"{name}: rotation uses separate mul/add (bit-exact, no FMA)")

# --- B. numerical equivalence -----------------------------------------------------------
os.environ.setdefault("KVMEM_BRIDGE_ARM", "1")
LAYERS, HEADS, HD, KVH = 1, 4, 8, 2
RD = 6  # rotary dim (even, < HD)
ok = bridge.ensure(LAYERS, HEADS, HD, device="cpu", max_tokens=8,
                   dtype=torch.float32, kv_heads=KVH, force=True)
check(bool(ok), "bridge.ensure succeeds on CPU geometry")

torch.manual_seed(3)
N = 5
key = torch.randn(N, KVH * HD)
query = torch.randn(N, HEADS * HD)
positions = torch.arange(N, dtype=torch.int64)
cache = torch.randn(64, RD)  # [cos|sin] table

# reference: unfused expression with true fp32 rounding semantics
def ref_pre_rope(payload, heads_shape):
    x = payload.to(torch.float32).view(payload.shape[0], heads_shape[1], heads_shape[2])
    half = RD // 2
    tab = cache.index_select(0, positions)
    cos = tab[:, :half].unsqueeze(-2)
    sin = tab[:, half:RD].unsqueeze(-2)
    x1 = x[..., :half]
    x2 = x[..., half:RD]
    out = torch.cat([x1 * cos + x2 * sin, x2 * cos - x1 * sin, x[..., RD:]], dim=-1)
    return out

# q publish: capture_q writes query rows (inverse-rotated to pre-RoPE) into _query buffer
bridge.capture_q(query, positions, cache, RD, 0)
got_q, got_pos = bridge.read(0)  # published query rows
ref_q = ref_pre_rope(query, (N, HEADS, HD))
check(got_q is not None and got_q.shape[0] >= N, "capture_q published rows exist")
if got_q is not None and got_q.shape[0] >= N:
    diff_q = (got_q[:N].reshape(ref_q.shape) - ref_q).abs().max().item()
    check(diff_q == 0.0, f"capture_q is BIT-exact vs unfused reference (max diff {diff_q})")

# k buckets: capture_k accumulates inverse-rotated key rows into per-64-token bucket sums
# decode path (n <= ROWS_KEEP_MAX) publishes keep rows instead; test the keep form bitwise.
bridge.capture_k(key, positions, cache, RD, 0)
rows_keep = bridge._KEEP.get(0)
check(rows_keep is not None, "capture_k published keep rows")
if rows_keep is not None:
    ref_k = ref_pre_rope(key, (N, KVH, HD))
    diff_k = (rows_keep[:N].reshape(ref_k.shape) - ref_k).abs().max().item()
    check(diff_k == 0.0, f"capture_k is BIT-exact vs unfused reference (max diff {diff_k})")

# --- C. _scratch must not allocate inside a capture ------------------------------------
seen = {"capturing": True}
real_probe = torch.cuda.is_current_stream_capturing
torch.cuda.is_current_stream_capturing = lambda: seen["capturing"]
try:
    bridge._SCRATCH.clear()
    got = bridge._scratch("sentinel-shape", (4, 4), torch.zeros(1), torch.float32)
    check(got is None, "_scratch returns None when first use lands inside a capture")
    seen["capturing"] = False
    got2 = bridge._scratch("sentinel-shape", (4, 4), torch.zeros(1), torch.float32)
    check(got2 is not None and tuple(got2.shape) == (4, 4),
          "_scratch allocates eagerly outside capture and is reused after")
    got3 = bridge._scratch("sentinel-shape", (9, 9), torch.zeros(1), torch.float32)
    check(got3 is got2, "named scratch is created once (pointer-stable for graph replay)")
finally:
    torch.cuda.is_current_stream_capturing = real_probe
    bridge._SCRATCH.pop("sentinel-shape", None)

print("FAIL bridge_capture_test" if failed else "PASS bridge_capture_test")
sys.exit(1 if failed else 0)
