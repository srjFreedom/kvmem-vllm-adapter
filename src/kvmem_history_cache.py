"""Session history block cache: H (RAM) budget with a D (NVMe) tier behind it.

The entry gateway re-splits and re-scores a long history on every turn. Both are pure functions of
the history text, so the parsed blocks are cacheable. Keeping them in RAM is bounded by H; blocks
beyond H spill to the NVMe tier and are read back on demand - the same two-tier arrangement the KV
side uses (`HostKvStorage` + `nvme_kv_tier`), applied to the entry's working data.

This is what makes the disk tier reachable on the recommended path: with in-request KV compaction
unavailable on this backend, the H/D tiers otherwise have no live consumer.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


class HistoryCache:
    def __init__(self, host_bytes: int, disk_bytes: int = 0, disk_dir: str | None = None,
                 slot_bytes: int = 64 * 1024):
        self.host_bytes = max(0, int(host_bytes))
        self.slot_bytes = max(1024, int(slot_bytes))
        self._ram: OrderedDict[str, bytes] = OrderedDict()
        self._ram_bytes = 0
        self._disk = None
        self._disk_slots: dict[str, list[int]] = {}
        self._disk_len: dict[str, int] = {}
        self._disk_used: set[int] = set()
        self.hits = self.misses = self.spills = self.reads = 0
        if disk_bytes and disk_dir:
            try:
                import glob as _glob
                import sys as _sys
                if _glob.glob(os.path.join(os.path.dirname(__file__), "..", "build", "python",
                                           "vllm_kvmem_ext*.so")):
                    _sys.path.insert(0, os.path.abspath(os.path.join(
                        os.path.dirname(__file__), "..", "build", "python")))
                import vllm_kvmem_ext as ext
                os.makedirs(disk_dir, exist_ok=True)
                self._disk = ext.NvmeKvTier(disk_dir, int(disk_bytes), self.slot_bytes, False, True)
            except Exception:
                self._disk = None

    # --- RAM tier -------------------------------------------------------------------
    def _ram_put(self, key: str, payload: bytes) -> None:
        if key in self._ram:
            self._ram_bytes -= len(self._ram[key])
            del self._ram[key]
        self._ram[key] = payload
        self._ram_bytes += len(payload)
        while self._ram_bytes > self.host_bytes and self._ram:
            old_key, old = next(iter(self._ram.items()))
            del self._ram[old_key]
            self._ram_bytes -= len(old)
            self._spill(old_key, old)

    # --- disk tier ------------------------------------------------------------------
    def _spill(self, key: str, payload: bytes) -> None:
        """Write a payload of any size, chunked across slots.

        A single block is often larger than one slot (a 4,000-block history is ~230 KB), so the
        payload is split; the slot list is the key's disk address.
        """
        if self._disk is None or key in self._disk_slots:
            return
        chunks = [payload[i:i + self.slot_bytes]
                  for i in range(0, len(payload), self.slot_bytes)] or [b""]
        total = int(self._disk.slot_count())
        if len(chunks) > total:
            return
        free = [s for s in range(total) if s not in self._disk_used]
        if len(free) < len(chunks):
            return
        slots = []
        for slot, chunk in zip(free, chunks):
            try:
                self._disk.write_block(slot, chunk)
            except Exception:
                for done in slots:
                    self._disk_used.discard(done)
                return
            self._disk_used.add(slot)
            slots.append(slot)
        self._disk_slots[key] = slots
        self._disk_len[key] = len(payload)
        self.spills += 1

    def _from_disk(self, key: str):
        if self._disk is None:
            return None
        slots = self._disk_slots.get(key)
        if not slots:
            return None
        parts = []
        try:
            for slot in slots:
                parts.append(bytes(self._disk.read_block(slot, self.slot_bytes)))
        except Exception:
            return None
        self.reads += 1
        # Slots are fixed size, so the tail chunk is zero padded; trim to the stored length.
        return b"".join(parts)[: self._disk_len.get(key, sum(len(p) for p in parts))]

    # --- public ---------------------------------------------------------------------
    def put_blocks(self, text: str, blocks) -> None:
        payload = json.dumps(blocks).encode()
        self._ram_put(_key(text), payload)

    def get_blocks(self, text: str):
        key = _key(text)
        payload = None
        if key in self._ram:
            payload = self._ram.pop(key)
            self._ram[key] = payload
            self.hits += 1
        else:
            payload = self._from_disk(key)
            if payload is None:
                self.misses += 1
                return None
            self.hits += 1
            self._ram_put(key, payload)
        try:
            return json.loads(payload.decode())
        except Exception:
            return None

    def stats(self) -> dict:
        return {
            "ram_entries": len(self._ram),
            "ram_bytes": self._ram_bytes,
            "host_bytes": self.host_bytes,
            "disk_spills": self.spills,
            "disk_reads": self.reads,
            "disk_slots": len(self._disk_used),
            "disk_entries": len(self._disk_slots),
            "hits": self.hits,
            "misses": self.misses,
        }
