#!/usr/bin/env python3
"""Entry policy on the parsed request: B selection, R ceiling, CTX rejection.

No ASGI middleware: the policy rides on `ChatCompletionRequest`, so streaming and CORS are never
touched. The hook is one wrapper on the chat serving class, installed by our endpoint plugin.
"""
import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "plugin"))
sys.path.insert(0, os.path.join(ROOT, "src"))

os.environ.update({
    "KVMEM_ENTRY_B": "4096",
    "KVMEM_ENTRY_R": "64",
    "KVMEM_ENTRY_CTX": "100000",
    "KVMEM_ENTRY_SINK": "256",
    "KVMEM_ENTRY_RECENT": "512",
    "KVMEM_ENTRY_HISTORY_KIB": "0",
})

import kvmem_entry_middleware as entry  # noqa: E402
from kvmem_entry_policy import ContextTooLong, EntryPolicy  # noqa: E402

checks = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    checks.append(bool(cond))


class FakeRequest:
    def __init__(self, messages, max_tokens=None, max_completion_tokens=None):
        self.messages = messages
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens


class FakeServing:
    """Stands in for OpenAIServingChat: records what the handler would have seen."""

    seen = []

    def create_error_response(self, message, err_type=None, status_code=400):
        return {"error": message, "type": err_type, "code": status_code}

    async def create_chat_completion(self, request, raw_request=None):
        FakeServing.seen.append(request)
        return {"ok": True, "max": request.max_completion_tokens}


def main():
    policy = EntryPolicy.from_env()

    # --- R ceiling -------------------------------------------------------------------
    request = FakeRequest([{"role": "user", "content": "hi"}], max_tokens=5000)
    stats = entry.apply_policy(request, policy)
    check(request.max_completion_tokens == 64, "a client budget above R is capped")
    check(stats["gen_budget"]["requested"] == 5000, "the cap is reported in the stats")

    request = FakeRequest([{"role": "user", "content": "hi"}], max_tokens=32)
    check(entry.apply_policy(request, policy) is None and request.max_completion_tokens is None,
          "a client budget below R is left alone (no field is written)")

    request = FakeRequest([{"role": "user", "content": "hi"}])
    entry.apply_policy(request, policy)
    check(request.max_completion_tokens == 64, "an omitted budget is stamped with R")

    request = FakeRequest([{"role": "user", "content": "hi"}], max_completion_tokens=8,
                          max_tokens=5000)
    entry.apply_policy(request, policy)
    check(request.max_completion_tokens == 8, "the effective field wins when both are present")

    # --- B selection ------------------------------------------------------------------
    history = "\n".join(f"Record {i}: the access key for record {i} is K{i:06d}."
                        for i in range(1, 3000))
    question = "\n\nAnswer with exactly two keys: record 250 and record 2800."
    request = FakeRequest([{"role": "user", "content": history + question}])
    stats = entry.apply_policy(request, policy)
    text = request.messages[0]["content"]
    check(bool(stats and stats.get("selection")), "a long history reports a selection")
    check("K000250" in text and "K002800" in text, "the records the question names survive")
    check(len(text) < len(history) // 3, f"the prompt shrank ({len(text)} chars)")

    # --- CTX ---------------------------------------------------------------------------
    tight = EntryPolicy(budget=4096, sink_tokens=256, recent_tokens=512, ctx=1000)
    request = FakeRequest([{"role": "user", "content": "A" * 60000}])
    try:
        entry.apply_policy(request, tight)
        check(False, "a history above CTX is rejected")
    except ContextTooLong:
        check(True, "a history above CTX is rejected")

    # --- the installed hook -------------------------------------------------------------
    FakeServing.seen = []
    os.environ["KVMEM_ENTRY_R"] = "64"
    installed = entry.install(FakeServing)
    check(installed, "install() wraps the chat serving class")
    check(getattr(FakeServing.create_chat_completion, "_kvmem_entry_patched", False),
          "the wrapper is marked so a double install is a no-op")
    check(entry.install(FakeServing), "a second install() is idempotent")

    served = asyncio.run(FakeServing().create_chat_completion(
        FakeRequest([{"role": "user", "content": "hi"}], max_tokens=9000)))
    check(served["ok"] is True, "the wrapped handler still runs")
    check(FakeServing.seen[-1].max_completion_tokens == 64,
          "the handler sees the capped budget")

    result = asyncio.run(FakeServing().create_chat_completion(
        FakeRequest([{"role": "user", "content": "A" * 600000}]), None))
    check(isinstance(result, dict) and result.get("code") == 400,
          "an over-CTX request returns a 400-style error instead of reaching the engine")

    print(("PASS" if all(checks) else "FAILED") + " entry_middleware_test")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
