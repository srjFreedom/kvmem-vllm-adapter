# The restore control plane: the worker asks EngineCore for a free aligned block and gets one.
import sys
import time

sys.path.insert(0, "src")
from kvmem_block_channel import BlockChannel, BlockChannelScheduler

failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def main():
    handed = [200, 201, 202]

    def allocator(request_id, count):
        if request_id == "unknown":
            raise RuntimeError("no such request")
        return handed[:count]

    path = "/tmp/kvmem_block_channel_test.sock"
    received = []

    def compact(request_id, pages, page_tokens, visible, rebuild, block_pages, physical):
        received.append((request_id, pages, physical))
        if request_id == "refused":
            return {"ok": False, "truncated": False, "rebuild": False}
        return {"ok": True, "truncated": True, "rebuild": bool(rebuild),
                "queued": bool(rebuild), "tokens": 7}

    server = BlockChannelScheduler(allocator, path=path, compactor=compact).start()
    client = BlockChannel(path=path, timeout=2.0)
    time.sleep(0.1)

    got = client.request("req-a", 1)
    check(got == [200], f"a worker gets a free block ({got})")
    got2 = client.request("req-a", 3)
    check(got2 == handed, "the scheduler honours the requested count")

    # A failing allocator must be reported to the worker, not kill the scheduler.
    try:
        client.request("unknown", 1)
        check(False, "an allocator failure must raise in the worker")
    except RuntimeError as exc:
        check("no such request" in str(exc), "the failure reaches the worker")
    check(client.request("req-a", 1) == [200], "the scheduler keeps serving after a failure")
    check(server.requests_served == 3, f"served {server.requests_served} requests")

    reply = client.compact("rebuild", [0, 8], 4, visible_tokens=7,
                           rebuild=True, physical_pages=[40, 41])
    check(reply.get("rebuild") and reply.get("queued") and reply.get("tokens") == 7,
          "rebuild metadata survives server and client without bool coercion")
    check(received[-1] == ("rebuild", [0, 8], [40, 41]),
          "logical pages and physical slots travel independently")
    reply = client.compact("refused", [0], 4, physical_pages=[40])
    check(not reply["ok"] and not reply["truncated"],
          "a nonempty refusal dictionary remains a refusal")
    server.stop()
    print("PASS block_channel_test" if not failed else "FAILED block_channel_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
