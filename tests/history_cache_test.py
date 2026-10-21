#!/usr/bin/env python3
"""Entry-side H/D tiers: RAM budget enforced, cold blocks read back from the NVMe tier."""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "python"))
from kvmem_history_cache import HistoryCache  # noqa: E402

checks = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    checks.append(bool(cond))


def main():
    disk = "/tmp/kvmem_history_cache_test"
    shutil.rmtree(disk, ignore_errors=True)
    cache = HistoryCache(host_bytes=8 * 1024, disk_bytes=8 * 1024 * 1024, disk_dir=disk,
                         slot_bytes=64 * 1024)
    check(cache._disk is not None, "the NVMe tier opens when a path is configured")

    texts, blocks = {}, {}
    for i in range(24):
        text = f"history-{i}-" + ("x" * 12000)      # ~12 KB each, well above the 8 KB H budget
        texts[i] = text
        # A realistic parsed history: hundreds of record-sized blocks (~55 chars each).
        blocks[i] = [f"Record {i}-{j}: the access key for record {i}-{j} is KEY{i:02d}{j:03d}."
                     for j in range(400)]
        cache.put_blocks(text, blocks[i])

    stats = cache.stats()
    check(stats["ram_bytes"] <= 8 * 1024, f"RAM stayed within H ({stats['ram_bytes']} <= 8192)")
    check(stats["disk_spills"] > 0, f"cold blocks spilled to disk ({stats['disk_spills']} blocks)")

    # Every block is still retrievable, byte-exact, even after eviction from RAM.
    ok = all(cache.get_blocks(texts[i]) == blocks[i] for i in range(24))
    check(ok, "every history is readable again (RAM or disk)")
    check(cache.stats()["disk_reads"] > 0, f"some reads came from disk ({cache.stats()['disk_reads']})")

    # A missing key is a miss, not an exception.
    check(cache.get_blocks("never-stored") is None, "unknown histories report a miss")

    shutil.rmtree(disk, ignore_errors=True)
    print(("PASS" if all(checks) else "FAILED") + " history_cache_test")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
