"""Opt-in exact-frontier session candidates; not qualified for production use.

Native (cache_salt, session_id) scopes a candidate, chained block hashes locate it, and a full
SHA256 of the executed token prefix independently verifies its identity. Each TP worker retains
KV/statistics and the GDN StateImage from that SAME block-aligned frontier. EngineCore publishes
metadata only after both workers acknowledge the terminal checkpoint. A hit restores the full
resident prefix and prefills the remaining prompt normally; compressed checkpoint restores and
MTP checkpoints remain unsupported.
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
                 "checkpoint", "session_id", "owner_session_id", "prefix_digest",
                 "source_request", "capacity_pages", "last_used", "_mamba_state")

    def __init__(self, block_hashes, orchestrator, covered_tokens, page_tokens,
                 mamba_state=None, session_id=None, checkpoint=None, owner_session_id=None,
                 prefix_digest="", source_request="", capacity_pages=0):
        self.block_hashes = tuple(block_hashes)
        self.orchestrator = orchestrator
        self.covered_tokens = int(covered_tokens)
        self.page_tokens = int(page_tokens)
        self.checkpoint = checkpoint
        self._mamba_state = None if checkpoint is not None else mamba_state
        self.session_id = session_id  # H-budget pool session, not the API conversation ID
        self.owner_session_id = owner_session_id
        self.prefix_digest = str(prefix_digest or "")
        self.source_request = str(source_request or "")
        self.capacity_pages = int(capacity_pages)
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

    def _key(self, block_hashes, owner_session_id=None):
        import hashlib
        owner = hashlib.sha256(str(owner_session_id or "").encode()).hexdigest()[:16]
        return "session:" + owner + ":" + ":".join(
            str(h) for h in tuple(block_hashes)[-2:])

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self):
        return list(self._entries.keys())

    def park(self, block_hashes, orchestrator, covered_tokens, page_tokens,
             mamba_state=None, session_id=None, owner_session_id=None,
             prefix_digest="", source_request="", capacity_pages=0) -> None:
        hashes = tuple(block_hashes)
        if not hashes:
            return
        key = (owner_session_id, hashes)
        old = self._entries.get(key)
        if old is not None and old.orchestrator is not orchestrator:
            # Same conversation re-parked with newer state: the new entry replaces the old one, and
            # the OLD orchestrator's share must be released (the caller owns that bookkeeping).
            self._drop(old)
        ck = None
        if self.catalog is not None and mamba_state:
            ck = self.catalog.put(self._key(hashes, owner_session_id), "session", int(covered_tokens),
                                  int(covered_tokens), (), mamba_state, step=int(covered_tokens))
        self._entries[key] = SessionEntry(hashes, orchestrator, covered_tokens,
                                          page_tokens, mamba_state, session_id, checkpoint=ck,
                                          owner_session_id=owner_session_id,
                                          prefix_digest=prefix_digest,
                                          source_request=source_request,
                                          capacity_pages=capacity_pages)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_sessions:
            # Evict whole sessions only: a half session would leave a resume length that no longer
            # matches its page store.
            _, evicted = self._entries.popitem(last=False)
            self._drop(evicted)

    def _drop(self, entry) -> None:
        if self.catalog is not None and getattr(entry, "checkpoint", None) is not None:
            self.catalog.drop(self._key(entry.block_hashes, entry.owner_session_id))
        if self._on_evict is not None:
            try:
                self._on_evict(entry)
            except Exception:
                pass

    def lookup(self, block_hashes, owner_session_id=None,
               prefix_digest=None) -> tuple["SessionEntry", int] | None:
        """Longest exact-content prefix within the requested native session namespace."""
        incoming = tuple(block_hashes)
        best: tuple["SessionEntry", int] | None = None
        best_key = None
        for key, entry in self._entries.items():
            if owner_session_id is not None and entry.owner_session_id != owner_session_id:
                continue
            prefix = entry.block_hashes
            n = len(prefix)
            if n and n <= len(incoming) and incoming[:n] == prefix:
                if (prefix_digest is not None and entry.prefix_digest
                        and entry.prefix_digest != prefix_digest(entry.covered_tokens)):
                    continue
                if best is None or n > best[1]:
                    best = (entry, n)
                    best_key = key
        if best_key is not None:
            self._entries.move_to_end(best_key)
        return best

    def evict_oldest(self):
        """Release one idle checkpoint and its H share before a cold request is admitted."""
        if not self._entries:
            return False
        _, entry = self._entries.popitem(last=False)
        self._drop(entry)
        return True

    def discard_requests(self, request_ids):
        request_ids = set(request_ids)
        removed = 0
        for key, entry in list(self._entries.items()):
            if entry.source_request in request_ids:
                self._entries.pop(key, None)
                self._drop(entry)
                removed += 1
        return removed

    def take(self, entry) -> "SessionEntry | None":
        """Remove and return an entry (the orchestrator is adopted by a live request)."""
        key = (entry.owner_session_id, entry.block_hashes)
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
                 "capacity_pages", "has_state", "prefix_digest", "owner_session_id",
                 "source_request")

    def __init__(self, block_hashes, covered_tokens, page_tokens, pages_per_block,
                 capacity_pages, has_state, prefix_digest="", owner_session_id=None,
                 source_request=""):
        self.block_hashes = tuple(block_hashes)
        self.covered_tokens = int(covered_tokens)
        self.page_tokens = int(page_tokens)
        self.pages_per_block = max(1, int(pages_per_block))
        self.capacity_pages = int(capacity_pages)
        self.has_state = bool(has_state)
        # SHA256 of the block-aligned prompt prefix, recomputed by EngineCore on a hit.
        # This independently checks the worker's published token identity.
        self.prefix_digest = str(prefix_digest or "")
        self.owner_session_id = owner_session_id
        self.source_request = str(source_request or "")


class MetaStore:
    """LRU of parked-session metadata, keyed by the same block-hash sequence as the worker store."""

    def __init__(self, max_sessions: int = 8):
        self._entries: "OrderedDict[tuple, MetaEntry]" = OrderedDict()
        self._max = max(1, int(max_sessions))

    def __len__(self) -> int:
        return len(self._entries)

    def park(self, block_hashes, covered_tokens, page_tokens, pages_per_block,
             capacity_pages, has_state, prefix_digest="", owner_session_id=None,
             source_request="") -> None:
        hashes = tuple(block_hashes)
        if not hashes:
            return
        key = (owner_session_id, hashes)
        self._entries[key] = MetaEntry(hashes, covered_tokens, page_tokens,
                                       pages_per_block, capacity_pages, has_state,
                                       prefix_digest, owner_session_id, source_request)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def discard_requests(self, request_ids):
        request_ids = set(request_ids)
        for key, entry in list(self._entries.items()):
            if entry.source_request in request_ids:
                self._entries.pop(key, None)

    def lookup(self, block_hashes, prefix_digest=None,
               owner_session_id=None) -> tuple["MetaEntry", int] | None:
        incoming = tuple(block_hashes)
        best = None
        best_key = None
        for key, entry in self._entries.items():
            if owner_session_id is not None and entry.owner_session_id != owner_session_id:
                continue
            prefix = entry.block_hashes
            n = len(prefix)
            if n and n <= len(incoming) and incoming[:n] == prefix:
                if (prefix_digest is not None and entry.prefix_digest
                        and entry.prefix_digest != prefix_digest(n)):
                    continue
                if best is None or n > best[1]:
                    best = (entry, n)
                    best_key = key
        if best_key is not None:
            self._entries.move_to_end(best_key)
        return best


def session_checkpoint_frontier(prompt_tokens, block_tokens, budget_tokens,
                                select_block_tokens=0):
    """Pre-pressure frontier aligned to both native KV and retrieval selection blocks."""
    from math import lcm
    prompt, block, budget = int(prompt_tokens), int(block_tokens), int(budget_tokens)
    selected = int(select_block_tokens or block)
    if block <= 0 or selected <= 0 or budget <= 0 or prompt <= block:
        return 0
    quantum = lcm(block, selected)
    # Leave one native KV block of headroom. The old B-aligned boundary triggered physical
    # compaction before the checkpoint could be captured at 36064 tokens in the MTP-off geometry.
    limit = max(0, budget - block)
    return max(0, min(((prompt - 1) // quantum) * quantum,
                      (limit // quantum) * quantum))


def clip_session_checkpoint_chunk(start, count, prompt, block, budget,
                                  select_block_tokens=0):
    """Stop a prefill step at its planned checkpoint without changing later chunks."""
    target = session_checkpoint_frontier(prompt, block, budget,
                                         select_block_tokens)
    if target and int(start) < target < int(start) + int(count):
        return target - int(start)
    return count


def terminal_checkpoint_requests(output_map, allowed_reasons):
    """Only normal terminal outputs may publish checkpoints; errors and aborts cannot."""
    return sorted({
        output.request_id
        for batch in (output_map or {}).values()
        for output in (getattr(batch, "outputs", None) or ())
        if getattr(output, "finish_reason", None) in allowed_reasons
    })


def verify_tp_checkpoint_records(records, tp_size, request_id):
    """Return one agreed checkpoint record, or refuse publication for all TP ranks."""
    if len(records) != tp_size or any(not isinstance(p, dict) for p in records):
        raise ValueError("TP workers did not all prepare a checkpoint")
    first = records[0]
    keys = ("hashes", "covered", "prefix_digest", "kv_block_size", "page_tokens",
            "pages_per_block", "capacity_pages", "has_state", "session_id",
            "expected_ranks", "request")
    if any(any(p.get(key) != first.get(key) for key in keys) for p in records):
        raise ValueError("TP checkpoint metadata differs")
    if {int(p.get("rank", -1)) for p in records} != set(range(tp_size)):
        raise ValueError("TP checkpoint rank set is incomplete")
    if first.get("request") != request_id or int(first.get("expected_ranks", 0)) != tp_size:
        raise ValueError("checkpoint request or TP size mismatch")
    namespace = first.get("session_id")
    if (not isinstance(namespace, (list, tuple)) or len(namespace) != 2
            or any(not isinstance(v, str) for v in namespace) or not namespace[1]):
        raise ValueError("checkpoint has no native session namespace")
    hashes = first.get("hashes")
    covered = int(first.get("covered", 0) or 0)
    block = int(first.get("kv_block_size", 0) or 0)
    page = int(first.get("page_tokens", 0) or 0)
    pages_per_block = int(first.get("pages_per_block", 0) or 0)
    capacity = int(first.get("capacity_pages", 0) or 0)
    if (not isinstance(hashes, list) or not hashes or block <= 0 or page <= 0
            or pages_per_block <= 0 or covered != len(hashes) * block
            or block != page * pages_per_block
            or capacity <= 0 or capacity % pages_per_block
            or capacity * page != covered):
        raise ValueError("checkpoint frontier or KV geometry is invalid")
    digest = first.get("prefix_digest")
    if not isinstance(digest, str) or len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest):
        raise ValueError("checkpoint token identity is missing")
    return first


_META_STORE: MetaStore | None = None


def get_meta_store() -> MetaStore:
    global _META_STORE
    if _META_STORE is None:
        _META_STORE = MetaStore()
    return _META_STORE
