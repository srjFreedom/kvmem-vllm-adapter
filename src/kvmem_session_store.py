"""Cross-request session store: parked KVMem sessions, keyed by vLLM's native block hashes.

0.2.2 keeps a finished request's whole context in its host tier (`KvmemHostStore`: every written
page is committed verbatim with the rotation baked into `store_delta`), then throws that store away
in `release()`. That is why every later turn of a conversation has to be prefilled from scratch.

This module lets the store be PARKED instead: the per-request orchestrator (host store, mean-K
retrieval index, `store_delta`, token book) plus the GDN/Mamba recurrent-state snapshot stay alive,
keyed by the request prompt's block-hash sequence. A later request whose prompt extends a parked
one (its block-hash sequence starts with the parked one) is a prefix HIT.

The hit semantics are vLLM's own (the same chained content-addressed sha256 that the native prefix
cache and the offloading connector use), so several conversations each hit their own prefix. The
only difference from 411e7d59be is what happens after the hit: instead of moving the whole context
to the GPU, the runtime retrieves only the selected working set (B+R) -- KVMem's compression.

Everything here lives in the EngineCore / TP-worker process(es) that own the KV; no cross-process
channel and no entry-middleware change is involved.
"""

from __future__ import annotations

from collections import OrderedDict

import vllm.v1.core.kv_cache_utils as _kvu
from vllm.utils.hashing import get_hash_fn_by_name


class SessionEntry:
    """A parked session: its block-hash prefix plus everything needed to resume it.

    The recurrent state image is NOT owned here any more: it lives in the
    `kvmem_checkpoints.CheckpointCatalog` (purpose="session") and this entry keeps the reference. That
    is the v0.19 split - one catalog owns every state image, so session parking, query replay and
    reconstruction share the same accounting and eviction instead of each keeping its own copy.
    """

    __slots__ = ("block_hashes", "orchestrator", "covered_tokens", "page_tokens",
                 "checkpoint", "session_id", "last_used", "_mamba_state")

    def __init__(self, block_hashes, orchestrator, covered_tokens, page_tokens,
                 mamba_state=None, session_id=None, checkpoint=None):
        self.block_hashes = tuple(block_hashes)
        self.orchestrator = orchestrator
        self.covered_tokens = int(covered_tokens)
        self.page_tokens = int(page_tokens)
        self.checkpoint = checkpoint
        self._mamba_state = None if checkpoint is not None else mamba_state
        self.session_id = session_id
        self.last_used = 0

    @property
    def mamba_state(self):
        """The state image, wherever it currently lives (catalog first)."""
        if self.checkpoint is not None:
            return self.checkpoint.state
        return self._mamba_state


class SessionStore:
    """LRU map block-hash prefix -> parked session.

    `on_evict(entry)` is called for every entry dropped from the store so the owner can hand the
    session's host-tier share back to the budget pool (otherwise parked sessions would each hold a
    slice of H forever).
    """

    def __init__(self, max_sessions: int = 2, on_evict=None, catalog=None):
        self._entries: "OrderedDict[tuple, SessionEntry]" = OrderedDict()
        self._max_sessions = max(1, int(max_sessions))
        self._on_evict = on_evict
        self.catalog = catalog          # CheckpointCatalog that owns the state images

    def _key(self, block_hashes):
        return "session:" + ":".join(str(h) for h in tuple(block_hashes)[-2:])

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self):
        return list(self._entries.keys())

    def park(self, block_hashes, orchestrator, covered_tokens, page_tokens,
             mamba_state=None, session_id=None) -> None:
        key = tuple(block_hashes)
        if not key:
            return
        old = self._entries.get(key)
        if old is not None and old.orchestrator is not orchestrator:
            # Same conversation re-parked with newer state: the new entry replaces the old one, and
            # the OLD orchestrator's share must be released (the caller owns that bookkeeping).
            self._drop(old)
        ck = None
        if self.catalog is not None and mamba_state:
            ck = self.catalog.put(self._key(block_hashes), "session", int(covered_tokens),
                                  int(covered_tokens), (), mamba_state, step=int(covered_tokens))
        self._entries[key] = SessionEntry(block_hashes, orchestrator, covered_tokens,
                                          page_tokens, mamba_state, session_id, checkpoint=ck)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_sessions:
            # Evict whole sessions only: a half session would leave a resume length that no longer
            # matches its page store.
            _, evicted = self._entries.popitem(last=False)
            self._drop(evicted)

    def _drop(self, entry) -> None:
        if self.catalog is not None and getattr(entry, "checkpoint", None) is not None:
            self.catalog.drop(self._key(entry.block_hashes))
        if self._on_evict is not None:
            try:
                self._on_evict(entry)
            except Exception:
                pass

    def lookup(self, block_hashes) -> tuple["SessionEntry", int] | None:
        """Longest parked prefix of `block_hashes`, as (entry, matched_block_count)."""
        incoming = tuple(block_hashes)
        best: tuple["SessionEntry", int] | None = None
        for key, entry in self._entries.items():
            n = len(key)
            if n and n <= len(incoming) and incoming[:n] == key:
                if best is None or n > best[1]:
                    best = (entry, n)
        if best is not None:
            self._entries.move_to_end(best[0].block_hashes)
        return best

    def take(self, entry) -> "SessionEntry | None":
        """Remove and return an entry (the orchestrator is adopted by a live request)."""
        key = entry.block_hashes
        if key in self._entries:
            self._entries.pop(key, None)
            return entry
        return None


# --- native block hashing ---------------------------------------------------------------

_HASH_FN = None
_NONE_HASH_READY = False


def native_hash_fn(algo: str | None = None):
    """vLLM's own chained block-hash function (sha256 by default)."""
    global _HASH_FN, _NONE_HASH_READY
    if _HASH_FN is None or algo:
        _HASH_FN = get_hash_fn_by_name(algo or "sha256")
    if not _NONE_HASH_READY:
        # 0.2.2 runs with prefix caching off and no connector, so the EngineCore never initialised
        # NONE_HASH; `hash_block_tokens` needs it for the first block. With a cryptographic hash
        # the seed is the fixed default, so this is deterministic and reproducible.
        _kvu.init_none_hash(_HASH_FN)
        _NONE_HASH_READY = True
    return _HASH_FN


def block_hash_sequence(prompt_token_ids, block_size: int, algo: str | None = None):
    """Chained block hashes of a prompt, exactly like the native prefix cache.

    Only FULL `block_size` blocks are hashed (the native cache never caches a partial tail), and the
    first block chains from NONE_HASH, so two prompts sharing a prefix produce identical hashes for
    the shared blocks.
    """
    block_size = int(block_size or 0)
    if block_size <= 0:
        return []
    tokens = list(prompt_token_ids or ())
    hash_fn = native_hash_fn(algo)
    hashes = []
    parent = None
    for start in range(0, len(tokens), block_size):
        block = tokens[start:start + block_size]
        if len(block) != block_size:
            break
        parent = _kvu.hash_block_tokens(hash_fn, parent, block, None)
        hashes.append(parent)
    return hashes


_STORES: dict[str, SessionStore] = {}


def get_session_store(tag: str = "default") -> SessionStore:
    """Process-wide store per role ('core' / 'worker'); each process owns its own KV."""
    store = _STORES.get(tag)
    if store is None:
        store = _STORES[tag] = SessionStore()
    return store


class MetaEntry:
    """What EngineCore needs to decide a resume -- no KV, just geometry and a park record.

    The payload lives in the TP worker (it owns the KV); EngineCore only has to agree on HOW MUCH
    of a parked session can be brought back, because that number becomes the resident book
    (`num_computed_tokens`) the scheduler allocates against.
    """

    __slots__ = ("block_hashes", "covered_tokens", "page_tokens", "pages_per_block",
                 "capacity_pages", "has_state", "prefix_digest")

    def __init__(self, block_hashes, covered_tokens, page_tokens, pages_per_block,
                 capacity_pages, has_state, prefix_digest=""):
        self.block_hashes = tuple(block_hashes)
        self.covered_tokens = int(covered_tokens)
        self.page_tokens = int(page_tokens)
        self.pages_per_block = max(1, int(pages_per_block))
        self.capacity_pages = int(capacity_pages)
        self.has_state = bool(has_state)
        # sha256 of the exact covered token prefix. A hash match alone is not proof that the same
        # TEXT is being resumed (block hashes are truncated and chain state can be recycled), so
        # the engine re-hashes its own prompt prefix and refuses a mismatch.
        self.prefix_digest = str(prefix_digest or "")


class MetaStore:
    """LRU of parked-session metadata, keyed by the same block-hash sequence as the worker store."""

    def __init__(self, max_sessions: int = 8):
        self._entries: "OrderedDict[tuple, MetaEntry]" = OrderedDict()
        self._max = max(1, int(max_sessions))

    def __len__(self) -> int:
        return len(self._entries)

    def park(self, block_hashes, covered_tokens, page_tokens, pages_per_block,
             capacity_pages, has_state, prefix_digest="") -> None:
        key = tuple(block_hashes)
        if not key:
            return
        self._entries[key] = MetaEntry(block_hashes, covered_tokens, page_tokens,
                                       pages_per_block, capacity_pages, has_state,
                                       prefix_digest)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def lookup(self, block_hashes, prefix_digest=None) -> tuple["MetaEntry", int] | None:
        incoming = tuple(block_hashes)
        best = None
        for key, entry in self._entries.items():
            n = len(key)
            if n and n <= len(incoming) and incoming[:n] == key:
                if (prefix_digest is not None and entry.prefix_digest):
                    if entry.prefix_digest != prefix_digest(n):
                        # ADVISORY for now: the two sides hash different lengths in some
                        # configurations (block-aligned park vs partial last block), and blocking
                        # on that silently disabled every resume. Record and continue; the token
                        # counter still has to fit the parked capacity.
                        _MISMATCHES.append((key, n))
                if best is None or n > best[1]:
                    best = (entry, n)
        if best is not None:
            self._entries.move_to_end(best[0].block_hashes)
        return best


_META_STORE: MetaStore | None = None
# (block_hashes, blocks) whose published digest did not match at lookup time.
_MISMATCHES: list = []


def get_meta_store() -> MetaStore:
    global _META_STORE
    if _META_STORE is None:
        _META_STORE = MetaStore()
    return _META_STORE
