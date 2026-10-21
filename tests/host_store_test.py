# The page store must be KVMem's host tier: byte-identical round trip, and the H budget must be
# enforced by KVMem (over-subscription evicts instead of silently growing the process).
import sys
import torch

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")
import vllm_kvmem_ext as ext
from kvmem_sparse_orchestrator import KvmemHostStore

PAGE, HEADS, HD, CONTENT, LAYERS = 16, 2, 256, 512, 3
RECORD = LAYERS * PAGE * HEADS * CONTENT
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def page_bytes(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 255, (HEADS, PAGE, CONTENT), generator=g, dtype=torch.uint8)
            for _ in range(LAYERS)]


def main():
    store = KvmemHostStore(capacity_bytes=RECORD * 4, page=PAGE, layers=LAYERS,
                           record_bytes=RECORD, tensor_shape=(HEADS, PAGE, CONTENT))
    check(store.capacity_bytes() == RECORD * 4, "the store reports KVMem's capacity")

    payload = page_bytes(11)
    store.put(3, payload)
    back = store.get(3)
    check(all(torch.equal(a, b) for a, b in zip(payload, back)),
          "a page round-trips byte-identically through KVMem's tier")
    check(store.charged_bytes() >= RECORD, "the page is charged against the H budget")
    check(store.has(3), "the page is present")

    # Fill past capacity: KVMem's allocate must fail and the store evicts the oldest page.
    for p in range(4, 9):
        store.put(p, page_bytes(p))
    check(store.charged_bytes() <= store.capacity_bytes(),
          "charged bytes never exceed the H budget")
    check(not store.has(3), "the oldest page was evicted under pressure")
    check(store.has(8), "the newest page is retained")

    print("PASS host_store_test" if not failed else "FAILED host_store_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
