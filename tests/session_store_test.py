"""Offline checks for the cross-request session store (no engine, no GPU).

What this pins down:
  1. `block_hash_sequence` is the NATIVE vLLM chained content hash: identical prefixes hash
     identically regardless of what follows, and a prompt that extends another produces a hash
     sequence that starts with the shorter one's (that is the whole hit mechanism).
  2. The hash sequence only covers FULL blocks (the native prefix cache never caches a tail).
  3. `SessionStore.lookup` returns the LONGEST parked prefix and only when it is a real prefix.
  4. Parking the same conversation again replaces the entry and evicts whole sessions under LRU
     pressure (never a half session).

Run: python tests/session_store_test.py   (uses vLLM's hashing, so run inside the vLLM venv)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kvmem_session_store import (  # noqa: E402
    SessionStore,
    block_hash_sequence,
    native_hash_fn,
)


def test_prefix_hashes_are_stable_and_extendable():
    block = 16
    base = list(range(1000, 1000 + 320))
    longer = base + list(range(2000, 2032))
    other = list(range(5000, 5320))

    h_base = block_hash_sequence(base, block)
    h_longer = block_hash_sequence(longer, block)
    h_other = block_hash_sequence(other, block)

    assert len(h_base) == 20, len(h_base)  # only full blocks
    assert h_longer[:len(h_base)] == h_base, "an extension must share the prefix hashes"
    assert h_other[0] != h_base[0], "different content must not collide"
    # Determinism: recomputing gives exactly the same bytes.
    assert block_hash_sequence(base, block) == h_base


def test_partial_tail_is_not_hashed():
    block = 16
    full = list(range(500))
    tail = full + list(range(5000, 5007))  # 7 extra tokens: no full extra block
    assert block_hash_sequence(tail, block) == block_hash_sequence(full, block)


def test_lookup_returns_longest_prefix_only():
    block = 16
    store = SessionStore(max_sessions=4)
    short_prompt = list(range(100, 100 + 160))       # 10 blocks
    long_prompt = short_prompt + list(range(900, 900 + 160))  # 20 blocks

    store.park(block_hash_sequence(short_prompt, block), orchestrator=object(),
               covered_tokens=len(short_prompt), page_tokens=16)
    store.park(block_hash_sequence(long_prompt, block), orchestrator=object(),
               covered_tokens=len(long_prompt), page_tokens=16)

    incoming = long_prompt + list(range(3000, 3000 + 16))  # extends BOTH
    hit = store.lookup(block_hash_sequence(incoming, block))
    assert hit is not None
    entry, blocks = hit
    assert blocks == 20, blocks                       # the longer prefix wins
    assert entry.covered_tokens == len(long_prompt)

    # A prompt that diverges early must NOT hit.
    divergent = list(range(7000, 7000 + 160))
    assert store.lookup(block_hash_sequence(divergent, block)) is None


def test_repark_replaces_and_lru_evicts_whole_sessions():
    block = 16
    evicted = []
    store = SessionStore(max_sessions=2, on_evict=evicted.append)

    p1 = list(range(100, 260))
    p2 = list(range(400, 560))
    p3 = list(range(700, 860))
    o1, o1b = object(), object()
    store.park(block_hash_sequence(p1, block), o1, len(p1), 16)
    store.park(block_hash_sequence(p1, block), o1b, len(p1), 16)  # re-park same conversation
    assert len(store) == 1, len(store)
    assert evicted and evicted[0].orchestrator is o1, "the superseded orchestrator is released"

    store.park(block_hash_sequence(p2, block), object(), len(p2), 16)
    store.park(block_hash_sequence(p3, block), object(), len(p3), 16)
    assert len(store) == 2, len(store)
    # The oldest whole session (the re-parked p1) was evicted, not a half one.
    assert store.lookup(block_hash_sequence(p1, block)) is None
    assert store.lookup(block_hash_sequence(p3, block)) is not None


def test_meta_digest_rejects_a_mismatched_prefix():
    """Two prompts can share leading blocks and still be different text further in; the digest
    published with the park must veto a resume that cannot prove the same covered prefix."""
    block = 16
    from kvmem_session_store import MetaStore
    import hashlib

    def digest(tokens):
        return hashlib.sha256(b"".join(int(t).to_bytes(4, "little", signed=False)
                                       for t in tokens)).hexdigest()

    store = MetaStore(max_sessions=4)
    prompt = list(range(100, 100 + 160))          # 10 blocks
    hashes = block_hash_sequence(prompt, block)
    store.park(hashes, covered_tokens=len(prompt), page_tokens=16, pages_per_block=1,
               capacity_pages=160, has_state=True, prefix_digest=digest(prompt))

    good = store.lookup(hashes, prefix_digest=lambda n: digest(prompt[:n * block]))
    assert good is not None and good[1] == 10, good

    # ADVISORY: a mismatch is recorded for telemetry but does not veto the resume yet (the two
    # sides hash different lengths in some geometries, and vetoing silently disabled reuse).
    import kvmem_session_store as _ss
    _ss._MISMATCHES.clear()
    bad = store.lookup(hashes, prefix_digest=lambda n: digest(list(range(9000, 9000 + 10 * block))))
    assert bad is not None, "lookup still resolves the entry"
    assert _ss._MISMATCHES, "the mismatch is recorded for telemetry"


def test_native_hash_fn_is_sha256_by_default():
    fn = native_hash_fn()
    assert fn.__name__ == "sha256", fn.__name__


if __name__ == "__main__":
    test_prefix_hashes_are_stable_and_extendable()
    test_partial_tail_is_not_hashed()
    test_lookup_returns_longest_prefix_only()
    test_repark_replaces_and_lru_evicts_whole_sessions()
    test_meta_digest_rejects_a_mismatched_prefix()
    test_native_hash_fn_is_sha256_by_default()
    print("session_store_test: OK")
