"""Checkpoint catalog (v0.19 `CheckpointCatalog` in miniature).

One record = a point in a request's life that can be RETURNED to:

    position   logical token count at capture        (drives where the tail starts)
    resident   `seq_len` at capture                  (the coordinate a rewind uses)
    state      the recurrent state (GDN) snapshot    (what makes the return cheap)
    pages      the block-table row that was installed at capture
    purpose    "session" | "replay" | "reconstruct"

Upstream keeps these in the KVMem core with an on-disk image; here the images live in RAM because
that is what this adapter can validate today. The point is the same: a return is
"restore state + rewind the two token books + re-prefill the tail", which never changes the request's
token identity - the operation that made the earlier `COMPACT_REBUILD` path stall.
"""

from __future__ import annotations


class Checkpoint:
    __slots__ = ("req", "purpose", "position", "resident", "pages", "state", "bytes", "step")

    def __init__(self, req, purpose, position, resident, pages, state, step):
        self.req = req
        self.purpose = str(purpose)
        self.position = int(position)
        self.resident = int(resident)
        self.pages = tuple(int(p) for p in (pages or ()))
        self.state = state
        self.step = int(step or 0)
        self.bytes = 0
        for _key, value in (state or {}).items():
            # `_capture_state_snapshot` stores {group: (slot, [tensors])}; accept a bare list too.
            tensors = value[1] if isinstance(value, tuple) and len(value) == 2 else value
            for t in (tensors or ()):
                try:
                    self.bytes += int(t.numel()) * int(t.element_size())
                except Exception:
                    pass

    def __repr__(self):
        return (f"Checkpoint({self.req[:12]}, {self.purpose}, pos={self.position}, "
                f"res={self.resident}, pages={len(self.pages)}, {self.bytes // 1024} KiB)")


class CheckpointCatalog:
    """Per-request checkpoint list with a bound on how many are kept."""

    def __init__(self, keep=2, max_bytes=0):
        self._items: dict[str, list[Checkpoint]] = {}
        self.keep = max(1, int(keep))
        self.max_bytes = int(max_bytes or 0)
        self.puts = 0
        self.drops = 0
        self.bytes = 0

    # -- writes ---------------------------------------------------------------------------
    def put(self, req, purpose, position, resident, pages, state, step=0):
        if not state:
            return None
        ck = Checkpoint(req, purpose, position, resident, pages, state, step)
        items = self._items.setdefault(req, [])
        items.append(ck)
        self.puts += 1
        self.bytes += ck.bytes
        # Keep the newest `keep` per (request, purpose)-ish: a reconstruction only ever wants the
        # latest usable boundary, and older images are pure memory.
        while len(items) > self.keep:
            old = items.pop(0)
            self.bytes = max(0, self.bytes - old.bytes)
            self.drops += 1
        if self.max_bytes and self.bytes > self.max_bytes:
            # Global pressure: drop the oldest from whichever request has the most.
            for r, lst in sorted(self._items.items(), key=lambda kv: -len(kv[1])):
                while len(lst) > 1 and self.bytes > self.max_bytes:
                    old = lst.pop(0)
                    self.bytes = max(0, self.bytes - old.bytes)
                    self.drops += 1
                break
        return ck

    def drop(self, req):
        for old in self._items.pop(req, []) or []:
            self.bytes = max(0, self.bytes - old.bytes)
            self.drops += 1

    # -- reads ----------------------------------------------------------------------------
    def latest(self, req, purpose=None, at_most=None):
        best = None
        for ck in self._items.get(req, ()):
            if purpose is not None and ck.purpose != purpose:
                continue
            if at_most is not None and ck.resident > int(at_most):
                continue
            if best is None or ck.resident >= best.resident:
                best = ck
        return best

    def stats(self):
        return {"puts": self.puts, "drops": self.drops, "bytes": self.bytes,
                "requests": len(self._items),
                "items": sum(len(v) for v in self._items.values())}
