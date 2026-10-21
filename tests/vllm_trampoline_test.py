# Drives the real MemorySession through the C++ -> Python trampoline with a numpy-backed
# handle. Validates the pybind11 boundary (geometry/catalog/stamp round-trips, the D2H/H2D
# buffer hand-off via ctypes, publish_view) and the spill/restore contract.
import ctypes
import sys

import numpy as np

sys.path.insert(0, "build/python")
import vllm_kvmem_ext as ext


class PyHandle:
    """A numpy-backed stand-in for one rank's paged-KV shard (the 'GPU' is a dict of buffers)."""

    def __init__(self, layers, kv_heads, head_dim, block_tokens, dtype_bytes=2):
        self.geom = ext.Geometry(layers, kv_heads, head_dim, block_tokens, dtype_bytes)
        self.record_bytes = block_tokens * kv_heads * head_dim * dtype_bytes * 2 * layers
        self._catalog = []
        self.gpu = {}
        self._resident = []
        self._stamp = ext.MemoryStamp()
        self._stamp.valid = True
        s = ext.SessionIdentity()
        s.id = 1
        s.incarnation = 1
        self._stamp.session = s
        self._stamp.history_revision = 1
        self._stamp.view_version = 0
        self._stamp.frontier = 0
        self._stamp.execution_history = 1

    def geometry(self):
        return self.geom

    def catalog(self):
        return self._catalog

    def resident(self):
        return self._resident

    def stamp(self):
        return self._stamp

    def add_block(self, bid, original_start, valid_tokens):
        cb = ext.CatalogBlock()
        s = ext.SessionIdentity()
        s.id = 1
        s.incarnation = 1
        cb.session = s
        cb.id = bid
        cb.original_start = original_start
        cb.valid_tokens = valid_tokens
        cb.content_version = 1
        cb.statistics_version = 1
        self._catalog.append(cb)
        self.gpu[bid] = np.array([(bid * 16 + i) & 0xFF for i in range(self.record_bytes)], dtype=np.uint8)
        self._resident.append(bid)
        self._stamp.frontier = original_start + valid_tokens

    def spill_block(self, bid, dst_addr, nbytes):
        dst = (ctypes.c_uint8 * nbytes).from_address(dst_addr)
        np.frombuffer(dst, dtype=np.uint8)[:] = self.gpu[bid]

    def restore_block(self, bid, src_addr, nbytes):
        src = np.frombuffer((ctypes.c_uint8 * nbytes).from_address(src_addr), dtype=np.uint8)
        self.gpu[bid] = src.copy()

    def publish_view(self, view):
        self._resident = [vb.block.id for vb in view.blocks]
        self._stamp = view.stamp


def check(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)


def advance_to_ready(session):
    phase = session.advance()
    while phase in (ext.MemoryPhase.Spilling, ext.MemoryPhase.Restoring):
        phase = session.advance()
    return phase


def main():
    bt = 16
    h = PyHandle(2, 2, 8, bt)
    check(h.record_bytes == 2048, "record_bytes")
    ref = {}
    for i in range(4):
        h.add_block(i, i * bt, bt)
        ref[i] = h.gpu[i].copy()

    backend = ext.VllmMemoryBackend(h, 1 << 30, "vllm-rank0-py")
    session = ext.MemorySession(backend)

    # Cycle 1: evict+spill 0,1; keep 2,3.
    plan = session.prepare(h.stamp(), [2, 3], [2, 3], ext.MemoryBudget(2 * bt, bt, 3 * bt, 4 * h.record_bytes))
    check(len(plan.spill) == 2 and len(plan.evict) == 2 and len(plan.restore) == 0, "cycle1 plan")
    check(advance_to_ready(session) == ext.MemoryPhase.Ready, "cycle1 ready")
    view = session.publish()
    check([b.block.id for b in view.blocks] == [2, 3], "cycle1 view")
    check(h.resident() == [2, 3], "cycle1 resident")
    snap = backend.snapshot()
    check(snap.blocks[0].host_version == 1 and snap.blocks[1].host_version == 1, "cycle1 host copies")

    # Cycle 2: restore block 0 (it has a host copy); keep 2,3.
    plan = session.prepare(h.stamp(), [0, 2, 3], [0, 2, 3], ext.MemoryBudget(3 * bt, bt, 4 * bt, 4 * h.record_bytes))
    check(len(plan.restore) == 1 and plan.restore[0].id == 0, "cycle2 plan")
    check(advance_to_ready(session) == ext.MemoryPhase.Ready, "cycle2 ready")
    view = session.publish()
    check([b.block.id for b in view.blocks] == [0, 2, 3], "cycle2 view")
    check(h.resident() == [0, 2, 3], "cycle2 resident")
    check(np.array_equal(h.gpu[0], ref[0]), "cycle2 byte-for-byte restore")

    print("PASS vllm_trampoline_test")


if __name__ == "__main__":
    main()
