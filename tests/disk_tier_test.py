"""Disk tier: when the H budget is exhausted, pages must stage out to the NVMe tier instead of
being dropped, and read back byte-exactly."""
import sys, tempfile, torch
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from kvmem_sparse_orchestrator import KvmemHostStore

page, layers, shape = 16, 2, (2, 16, 256)     # 16 KiB/rank, 2 layers
slot_bytes = layers * 2 * 16 * 256
d = tempfile.mkdtemp()
# H holds two pages, D holds the rest
store = KvmemHostStore(2 * slot_bytes, page=page, layers=layers, record_bytes=slot_bytes,
                       tensor_shape=shape, disk_bytes=16 * slot_bytes, disk_dir=d)
assert store._disk is not None and store._disk.enabled(), "disk tier did not initialise"
payloads = {}
import numpy as np
for page_id in range(6):
    raw = torch.randint(0, 255, (layers, 2, 16, 256), dtype=torch.uint8)
    data = [raw[i].clone() for i in range(layers)]
    payloads[page_id] = [t.clone() for t in data]
    store.put(page_id, data)
print("pages in RAM:", len(store), "| on disk:", len(store._disk_slot))
assert len(store._disk_slot) > 0, "nothing staged out"
for page_id, expect in payloads.items():
    got = store.get(page_id)
    for a, b in zip(got, expect):
        assert torch.equal(a, b), f"page {page_id} mismatch"
print("roundtrip through the disk tier: OK")
print("disk slots used:", len(store._disk_slot), "of", store._disk.slot_count())
print("PASS disk_tier_test")
