"""D6 state snapshot/restore invariants (design 42, R3.2a).

CPU-level: fake the runner layout (kv_caches, kv_cache_group_ids,
kv_cache_config groups with MambaSpec/attention specs, block_tables) and
prove:
  1. snapshot returns None cleanly when layout is missing;
  2. snapshot captures exactly the MambaSpec-group tensors (not attention);
  3. restore round-trips: mutate after snapshot, restore, bytes identical;
  4. row/block resolution follows group table column 0.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "engines/vllm")

from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402
from kvmem_sparse_runtime import VllmSparseRuntime as _RT


class _Spec:
    pass


class _FakeGroup:
    def __init__(self, spec):
        self.kv_cache_spec = spec


class _FakeCfg:
    def __init__(self, groups):
        self.kv_cache_groups = groups


class _FakeGpuTable:
    def __init__(self, rows):
        self.gpu = torch.tensor(rows, dtype=torch.int64)


class _FakeBT:
    def __init__(self, rows_list):
        self.block_tables = [_FakeGpuTable(r) for r in rows_list]


class _FakeRunner:
    pass


def make_runner():
    r = _FakeRunner()
    mamba_spec = MambaSpec(
        shapes=((4, 8), (2, 3)),
        dtypes=(torch.float32, torch.float32),
        page_size_padded=None,
        state_block_size=8,
    ) if False else None
    # construct MambaSpec via whatever signature the fork exposes:
    mamba_spec = MambaSpec.__new__(MambaSpec)
    object.__setattr__(mamba_spec, "shapes", ((4, 8), (2, 3)))
    object.__setattr__(mamba_spec, "dtypes", (torch.float32, torch.float32))
    att = _Spec()
    t_mamba0 = torch.zeros(3, 4, 8)
    t_mamba1 = torch.zeros(3, 2, 3)
    t_att = torch.zeros(3, 16)
    r.kv_caches = [t_mamba0, t_att, t_mamba1]
    r.kv_cache_group_ids = [0, 1, 0]
    r.kv_cache_config = _FakeCfg([_FakeGroup(mamba_spec), _FakeGroup(att)])
    # one table per GROUP: gid0 (mamba) rows=[[1,2],[0,0],[2,2]];
    # gid1 (attention) rows irrelevant here.
    r.block_tables = _FakeBT([[[1, 2], [0, 0], [2, 2]],
                              [[0, 0], [1, 1], [2, 2]]])
    return r, t_mamba0, t_mamba1, t_att


def rt(views=1):
    """Build a runtime with a fake view pointing at a fake runner."""
    r, tm0, tm1, ta = make_runner()
    class V:
        pass
    v = V()
    v.runner = r
    v.page_tokens = 16
    v.block_size_bytes = 16
    v.num_blocks = lambda: 100
    rt = _RT.__new__(_RT)
    rt.view = v
    rt.log = None
    rt.page_tokens = 16
    rt.budget_tokens = 36864
    rt.d_mode = True
    rt._state = {}
    rt._prompt_len = {}
    return rt, r, tm0, tm1, ta


def main():
    fails = []

    # 1) missing layout -> None
    class V0:
        runner = None
    rt0 = _RT.__new__(_RT)
    rt0.view = V0()
    rt0.log = None
    assert rt0.d_state_snapshot(0) is None
    assert rt0.d_state_restore(None) == 0
    print("[1] graceful None on missing layout")

    # 2) snapshot captures only mamba tensors (2 of 3)
    rt1, r, tm0, tm1, ta = rt()
    snap = rt1.d_state_snapshot(1)   # row 1 -> block ids: mamba col0=0, att n/a
    assert snap is not None and len(snap) == 2, f"snap len {0 if not snap else len(snap)}"
    ids = {id(c) for c, b, val in snap}
    assert id(tm0) in ids and id(tm1) in ids and id(ta) not in ids
    for c, b, val in snap:
        assert torch.equal(val, c[b])
    print("[2] snapshot selects MambaSpec groups only, block col0 resolved")

    # 3) round-trip identity after mutation
    tm0[0].fill_(7.0)
    tm1[0].fill_(9.0)
    assert not torch.equal(tm0[0], snap[0][2])
    n = rt1.d_state_restore(snap)
    assert n == 2
    # verify restored == original (snap[0] was row1->block0)
    # recompute expected from a fresh build
    rt2, r2, tm0b, tm1b, tab = rt()
    assert torch.equal(tm0[0], tm0b[0]) and torch.equal(tm1[0], tm1b[0])
    print("[3] restore round-trips byte-identically (conv+ssm both)")

    # 4) row resolution follows the GROUP table col 0 per layer group
    snap2 = rt1.d_state_snapshot(2)   # row2 -> mamba blocks [2,2]
    for c, b, val in snap2:
        assert b == 2, f"block {b} != 2"
    print("[4] per-row block resolution correct")

    # 5) restore on partial failure counts successes only
    bad = [(tm0, 0, torch.zeros(99, 99)), (tm1, 0, snap[1][2])]
    n2 = rt1.d_state_restore(bad)
    assert n2 == 1, f"partial restore count {n2}"
    print("[5] partial restore survives and counts")

    print("STATE-SNAPSHOT: ALL OK" if not fails else f"FAILS {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
